import torch
import torch.nn as nn
import os
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from alpha_clip import alpha_clip
from alpha_clip.model import build_model
from clip.model import convert_weights
import json
# from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from templates.imagenet_templates import IMAGENET_TEMPLATES
from templates.mapper_data import ctx_templates
from tqdm import tqdm

CUSTOM_TEMPLATES_PROTEXT = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of a texture.",
    "EuroSAT": "a photo of a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "OVcamo": "A photo of the camouflaged {}."
}

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

    model = build_model(state_dict or model.state_dict(), lora_adapt=False, rank=-1).to(device)
    if str(device) == "cpu":
        model.float()
    if alpha_vision_ckpt_pth != "None":
        model.visual.load_state_dict(torch.load(alpha_vision_ckpt_pth))
        model.eval()  # merge lora params if exists (for inference only)
    return model


@TRAINER_REGISTRY.register()
class AlphaCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(
            cfg,
            device="cpu",
            alpha_vision_ckpt_pth="/media/estar/Data/ywb/AlphaCLIP-main/checkpoints/clip_l14_336_grit_20m_4xe.pth"
        )
        # clip_model.to(self.device)
        # clip_model, _ = alpha_clip.load("ViT-L/14@336px",
        #                                 alpha_vision_ckpt_pth="/media/estar/Data/ywb/AlphaCLIP-main/checkpoints/clip_l14_336_grit_20m_4xe.pth",
        #                                 device='cpu',
        #                                 lora_adapt=False, rank=-1)
        clip_model = clip_model.float().cuda()
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Original classnames: {classnames}")
        print(f"Prompts: {prompts}")
        prompts = torch.cat([alpha_clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        use_gpt_prompts = cfg.TRAINER.PROTEXT.GPT_PATH
        use_attribute_prompts = cfg.TRAINER.PROTEXT.USE_ATTRIBUTE_DATA
        use_80_prompts = cfg.TRAINER.PROTEXT.USE_TEMPLATES
        use_camo_prompts = cfg.TRAINER.PROTEXT.CAMO
        mean_text_features = 0
        if use_80_prompts:
            print("Using standard 80 openai templates for text embeddings")
            total_templates_to_use = IMAGENET_TEMPLATES
            print(f"Prompt ensembling (n={len(total_templates_to_use)})")
            for i, temp in enumerate(total_templates_to_use):
                prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                prompts = torch.cat([alpha_clip.tokenize(p) for p in prompts]).cuda()
                with torch.no_grad():
                    text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_text_features = mean_text_features + text_features
            mean_text_features = mean_text_features / len(total_templates_to_use)
            mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
            text_features = mean_text_features
        # 第一种为 加和 的处理方法
        if use_camo_prompts and False:
            print("Using standard 6 camo templates for text embeddings")
            total_templates_to_use = CAMO_PROMPTS
            print(f"Prompt ensembling (n={len(total_templates_to_use)})")
            for i, temp in enumerate(total_templates_to_use):
                prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                prompts = torch.cat([alpha_clip.tokenize(p) for p in prompts]).cuda()
                with torch.no_grad():
                    text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                mean_text_features = mean_text_features + text_features
            mean_text_features = mean_text_features / len(total_templates_to_use)
            mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
            text_features = mean_text_features

        # 第二种 循环访问 template的方法
        if use_camo_prompts:
            camo_all = []
            print("Using standard 6 camo templates for text embeddings")
            total_templates_to_use = CAMO_PROMPTS
            print(f"Prompt ensembling (n={len(total_templates_to_use)})")
            with torch.no_grad():
                zeroshot_weights = []
                for classname in tqdm(classnames):
                    texts = [template.format(classname) for template in total_templates_to_use]  # format with class
                    texts = alpha_clip.tokenize(texts).cuda()  # tokenize
                    class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                camo_all = torch.stack(zeroshot_weights, dim=1).cuda()
            # 仿照gpt的方法：
            # for single_key in classnames:
            #     prompts = [temp.format(single_key.replace("_", " ")) for temp in total_templates_to_use]
            #     x_tokenized = torch.cat([clip.tokenize(p) for p in total_templates_to_use])
            #     with torch.no_grad():
            #         text_features = clip_model.encode_text(x_tokenized.cuda())
            #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            #     camo_all.append(text_features.mean(0).unsqueeze(0))
            # camo_all = torch.cat(camo_all, dim=0)
            # camo_all = camo_all / camo_all.norm(dim=-1, keepdim=True)

            # 仿照OVCOS的方法
            # for single_key in classnames:
            #     prompts = [temp.format(single_key.replace("_", " ")) for temp in total_templates_to_use]
            #     x_tokenized = torch.cat([alpha_clip.tokenize(p) for p in total_templates_to_use])
            #     with torch.no_grad():
            #         text_features = clip_model.encode_text(x_tokenized.cuda())
            #     camo_all.append(text_features)
            # camo_all = torch.stack(camo_all, dim=0)  # Nc,Nt,768
            # camo_all /= camo_all.norm(dim=-1, keepdim=True)
            # camo_all = camo_all.mean(1)
            # camo_all /= camo_all.norm(dim=-1, keepdim=True)

            if torch.is_tensor(mean_text_features):
                mean_text_features = torch.cat([mean_text_features.unsqueeze(0), camo_all.unsqueeze(0)], dim=0).mean(
                    0)
                mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
                text_features = mean_text_features
            else:
                text_features = camo_all
                mean_text_features = camo_all
        if use_gpt_prompts != None:
            print("Using CuPL templates for text embeddings")
            old_gpt_all = []
            file = open(use_gpt_prompts, "r")
            GPT_prompt_dict = json.load(file)
            # The order of embeddings should follow strictly order of classname variable
            # Keys name should match classnames so that we could do fetching from the dict.
            # Convert the dict to lower case
            GPT_prompt_dict = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}
            k = 0
            for single_key in classnames:
                single_class_prompts = GPT_prompt_dict[single_key.lower().replace("_", " ")]
                k += 1
                x_tokenized = torch.cat([alpha_clip.tokenize(p) for p in single_class_prompts])
                with torch.no_grad():
                    text_features = clip_model.encode_text(x_tokenized.cuda())
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                old_gpt_all.append(text_features.mean(0).unsqueeze(0))
            old_gpt_all = torch.cat(old_gpt_all, dim=0)
            old_gpt_all = old_gpt_all / old_gpt_all.norm(dim=-1, keepdim=True)
            print("Total CuPL prompt classes used for ZS evaluation: ", k)
            if torch.is_tensor(mean_text_features):
                mean_text_features = torch.cat([mean_text_features.unsqueeze(0), old_gpt_all.unsqueeze(0)], dim=0).mean(
                    0)
                mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
                text_features = mean_text_features
            else:
                text_features = old_gpt_all
                mean_text_features = old_gpt_all
        if use_attribute_prompts:
            attribute_prompt_all = 0
            print("Using attribute templates for text embeddings")
            print(f"Prompt ensembling (n={len(ctx_templates)})")
            for i, temp in enumerate(ctx_templates):
                prompts = [temp.format(c.replace("_", " ")) for c in classnames]
                prompts = torch.cat([alpha_clip.tokenize(p) for p in prompts]).cuda()
                with torch.no_grad():
                    text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                attribute_prompt_all = attribute_prompt_all + text_features
            attribute_prompt_all = attribute_prompt_all / len(ctx_templates)
            attribute_prompt_all = attribute_prompt_all / attribute_prompt_all.norm(dim=-1, keepdim=True)
            if torch.is_tensor(mean_text_features):
                mean_text_features = torch.cat([mean_text_features.unsqueeze(0), attribute_prompt_all.unsqueeze(0)],
                                               dim=0).mean(0)
                mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)
                text_features = mean_text_features
            else:
                text_features = attribute_prompt_all
                mean_text_features = attribute_prompt_all

        # If above is set to False, use single prompt dataset conditioned template
        if not torch.is_tensor(mean_text_features):
            print("Performing single 1 template zeroshot inference")
            with torch.no_grad():
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def parse_batch_test(self, batch):
        input = batch["img"]
        mask = batch["mask"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)
        mask = mask.to(self.device)

        inputs = [input, mask]
        return inputs, label

    def model_inference(self, image):
        images = image[0]
        masks = image[1]
        image_features = self.clip_model.visual(images, masks)
        # image_features = self.model.visual(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        logits = logit_scale * torch.matmul(image_features, self.text_features)
        return logits

    # @torch.no_grad()
    # def test(self, split=None):
    #     """A generic testing pipeline."""
    #     self.set_model_mode("eval")
    #     self.evaluator.reset()
    #
    #     if split is None:
    #         split = self.cfg.TEST.SPLIT
    #
    #     if split == "val" and self.val_loader is not None:
    #         data_loader = self.val_loader
    #     else:
    #         split = "test"  # in case val_loader is None
    #         data_loader = self.test_loader
    #
    #     print(f"Evaluate on the *{split}* set")
    #     temp_corr_dict = dict()
    #     for batch_idx, batch in enumerate(tqdm(data_loader)):
    #         input, target = self.parse_batch_test(batch)
    #         score = self.model_inference(input)
    #         pred = score.topk(1, dim=1)[1].squeeze(dim=1)
    #         pred_5 = score.topk(5, dim=1)[1].squeeze(dim=1)
    #         for i in range(target.shape[0]):
    #             if target[i].item() not in temp_corr_dict:
    #                 temp_corr_dict[target[i].item()] = [0, 0, 0]
    #             temp_corr_dict[target[i].item()][0] += 1
    #             if target[i].item() == pred[i].item():
    #                 temp_corr_dict[target[i].item()][1] += 1
    #             if target[i].item() in pred_5[i].tolist():
    #                 temp_corr_dict[target[i].item()][2] += 1
    #     output = [temp_corr_dict]
    #     # if self.local_rank == 0:
    #     final_dict = dict()
    #     for dic in output:
    #         for k, v in dic.items():
    #             if k not in final_dict.keys():
    #                 final_dict[k] = v
    #             else:
    #                 final_dict[k][0] += v[0]
    #                 final_dict[k][1] += v[1]
    #                 final_dict[k][2] += v[2]
    #     acc1 = 0.0
    #     acc5 = 0.0
    #     num_class = 0
    #     for v in final_dict.values():
    #         acc1 += v[1] / v[0]
    #         acc5 += v[2] / v[0]
    #         num_class += 1
    #     acc1 = acc1 / num_class
    #     acc5 = acc5 / num_class
    #     print("=====================================")
    #     print(f"test mean of per class acc-1 step 0: {acc1}")
    #     print(f"test mean of per class acc-5 step 0: {acc5}")
    #     print("=====================================")
    #
    #     # return list(results.values())[0]
