import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from alpha_clip_rw import alpha_clip
from alpha_clip_rw.simple_tokenizer import SimpleTokenizer as _Tokenizer
from alpha_clip_rw.model import build_model
import os

from utils import log
_tokenizer = _Tokenizer()

CAMO_PROMPTS = [
    "A photo of the camouflaged {}.",
    "A photo of the concealed {}.",
    "A photo of the {} camouflaged in the background.",
    "A photo of the {} concealed in the background.",
    "A photo of the {} camouflaged to blend in with its surroundings.",
    "A photo of the {} concealed to blend in with its surroundings.",
]

def load_clip_to_cpu(cfg, device="cpu", alpha_vision_ckpt_pth=None):
    device = device
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = alpha_clip._MODELS[backbone_name]
    model_path = alpha_clip._download(url, os.path.expanduser("~/.cache/clip"))

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            state_dict = torch.load(opened_file, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = build_model(state_dict or model.state_dict(), lora_adapt=False, rank=-1, design_details=design_details).to(device)
    if str(device) == "cpu":
        model.float()
    if alpha_vision_ckpt_pth != None:
        model.visual.load_state_dict(torch.load(alpha_vision_ckpt_pth))
        model.eval()  # merge lora params if exists (for inference only)
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, classnames_test, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_cls_test = len(classnames_test)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = alpha_clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        log('MaPLe design: Multi-modal Prompt Learning')
        log(f'Initial context: "{prompt_prefix}"')
        log(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 1024)
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 1024)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([alpha_clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        # 补充有关test的内容
        classnames_test = [name.replace("_", " ") for name in classnames_test]
        name_lens_test = [len(_tokenizer.encode(name)) for name in classnames_test]
        prompts_test = [prompt_prefix + " " + name + "." for name in classnames_test]

        tokenized_prompts_test = torch.cat([alpha_clip.tokenize(p) for p in prompts_test])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding_test = clip_model.token_embedding(tokenized_prompts_test).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_test", embedding_test[:, :1, :])  # SOS
        self.register_buffer("token_suffix_test", embedding_test[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls_test = n_cls_test
        self.tokenized_prompts_test = tokenized_prompts_test  # torch.Tensor
        self.name_lens_test = name_lens_test

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

    def forward_test(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls_test, -1, -1)

        prefix = self.token_prefix_test
        suffix = self.token_suffix_test
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, classnames_test, clip_model):
        super(CustomCLIP, self).__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, classnames_test, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_prompts_test = self.prompt_learner.tokenized_prompts_test
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # 补充原始的相关内容，先使用原始的看看效果
        log("Using standard 6 camo templates for text embeddings")
        total_templates_to_use = CAMO_PROMPTS
        log(f"Prompt ensembling (n={len(total_templates_to_use)})")
        # self.train_text_features = torch.load("/media/estar/Data/ywb/OVCamoDataset/text-features/ViT-L-14/TrainCamoPromptsTextFeaturesViTB-14-336.pth")
        # self.test_text_features = torch.load("/media/estar/Data/ywb/OVCamoDataset/text-features/ViT-L-14/TestCamoPromptsTextFeaturesViTB-14-336.pth")
        # # self.text_features = torch.load("/media/estar/Data/ywb/OVCamoDataset/text-features/ViT-L-14/TestCamoPromptsTextFeaturesViTB-14-336.pth")
        # with torch.no_grad():
        #     zeroshot_weights = []
        #     for classname in tqdm(classnames_test):
        #         texts = [template.format(classname) for template in total_templates_to_use]  # format with class
        #         texts = alpha_clip.tokenize(texts)  # tokenize
        #         class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
        #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        #         class_embedding = class_embeddings.mean(dim=0)
        #         class_embedding /= class_embedding.norm()
        #         zeroshot_weights.append(class_embedding)
        #     camo_all = torch.stack(zeroshot_weights, dim=0).cuda()
        # self.text_features = camo_all

    def load_text_features(self, train_text_features, test_text_features):
        self.train_text_features = train_text_features
        self.test_text_features = test_text_features

    def forward(self, image, mask, label=None, train=False):
        images = image
        masks = mask
        if train:
            tokenized_prompts = self.tokenized_prompts
            logit_scale = self.logit_scale.exp()

            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            image_features = self.image_encoder(images.type(self.dtype), masks.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features + self.train_text_features
            logits = logit_scale * image_features @ text_features.t()
            pred_label_id = logits.max(1)[1]
            return image_features.unsqueeze(1), text_features[pred_label_id].unsqueeze(1), pred_label_id, logits
        else:
            tokenized_prompts = self.tokenized_prompts_test
            logit_scale = self.logit_scale.exp()

            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner.forward_test()
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            image_features = self.image_encoder(images.type(self.dtype), masks.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features + self.test_text_features
            logits = logit_scale * image_features @ text_features.t()
            pred_label_id = logits.max(1)[1]
            return image_features.unsqueeze(1), text_features[pred_label_id].unsqueeze(1), pred_label_id, logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MaPLeAlphaCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        classnames_test = self.dm.dataset.classnames_test
        log(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(
            cfg,
            device="cpu",
            alpha_vision_ckpt_pth="/media/estar/Data/ywb/AlphaCLIP-main/checkpoints/clip_l14_336_grit_20m_4xe.pth"
        )

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        log("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, classnames_test, clip_model)

        log("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        log(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            log(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_test(self, batch):
        input = batch["img"]
        mask = batch["mask"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        mask = mask.to(self.device)

        inputs = [input, mask]
        return inputs, label

    def parse_batch_train(self, batch):
        input = batch["img"]
        mask = batch["mask"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        mask = mask.to(self.device)

        inputs = [input, mask]
        return inputs, label

    def load_model(self, directory, epoch=None):
        if not directory:
            log("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            log("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

class TrainMaPLeAlphaCLIP(nn.Module):
    def __init__(self, cfg, classnames, classnames_test):
        super().__init__()
        classnames = classnames
        classnames_test = classnames_test
        log(f"--------Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})---------")
        clip_model = load_clip_to_cpu(
            cfg,
            device="cpu",
            alpha_vision_ckpt_pth="/media/estar/Data/ywb/AlphaCLIP-main/checkpoints/clip_l14_336_grit_20m_4xe.pth"
        )

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        log("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, classnames_test, clip_model)
        log("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        log(f"Parameters of maple alphaclip model to be updated: {enabled}")

class TestMaPLeAlphaCLIP(nn.Module):
    def __init__(self, cfg, classnames_train, classnames_test):
        super().__init__()
        self.classnames_train = classnames_train
        self.classnames_test = classnames_test
        log(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(
            cfg,
            device="cpu"
        )

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        log("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.classnames_train, self.classnames_test, clip_model)

    def load_model(self, directory, epoch=None):
        if not directory:
            log("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            log("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
