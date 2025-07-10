import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer
logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
import open_clip
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

CAMO_PROMPTS = [
    "A photo of the camouflaged {}.",
    "A photo of the concealed {}.",
    "A photo of the {} camouflaged in the background.",
    "A photo of the {} concealed in the background.",
    "A photo of the {} camouflaged to blend in with its surroundings.",
    "A photo of the {} concealed to blend in with its surroundings.",
]


def get_prompt_template_by_name(name):
    if name == "camoprompts":
        template_set = CAMO_PROMPTS
    elif name == "imagenet":
        template_set = IMAGENET_PROMPT
    elif name == "vild":
        template_set = VILD_PROMPT
    else:
        raise NotImplementedError(template_set)
    return template_set

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class ConvNeXtCLIP(nn.Module):
    def __init__(
        self,
        model_name="convnext_large_d_320",
        pretrained="laion2b_s29b_b131k_ft_soup",
        template_set="camoprompts",
    ):
        super().__init__()
        self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained="/media/estar/Data/ywb/OVCamo-main/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin"
        )
        self.mean = OPENAI_DATASET_MEAN
        self.std = OPENAI_DATASET_STD
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        self.template_set = get_prompt_template_by_name(template_set)
        logger.info(f"Create the CLIP ({model_name + '-' + pretrained}) with template_set {self.template_set}")

        model_name = model_name.lower()
        assert "convnext_" in model_name
        self.model_type = "convnext"
        if "_base" in model_name:
            self.feat_chans = [128, 128, 256, 512, 1024]
        elif "_large" in model_name:
            self.feat_chans = [192, 192, 384, 768, 1536]
        elif "_xxlarge" in model_name:
            self.feat_chans = [384, 384, 768, 1536, 3072]

        self.dim_latent = self.clip_model.text_projection.shape[-1]
        self.out_strides = {"stem": 2, "res2": 4, "res3": 8, "res4": 16, "res5": 32, "emb": -1}
        self.out_chans = {
            "stem": self.feat_chans[0],
            "res2": self.feat_chans[1],
            "res3": self.feat_chans[2],
            "res4": self.feat_chans[3],
            "res5": self.feat_chans[4],
            "emb": self.dim_latent,
        }

    def output_shape(self):
        return {
            name: dict(channels=self.out_chans[name], stride=self.out_strides[name])
            for name in ["stem", "res2", "res3", "res4", "res5", "emb"]
        }

    @property
    def device(self):
        for param in self.clip_model.parameters():
            return param.device

    @torch.no_grad()
    def get_text_embs(self, text_list, normalize=True):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
        text_tokens = self.text_tokenizer(text_list).to(self.device)

        # list -> TD
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
        #
        x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take feats from the eot embedding (eot_token is the highest number in each sequence)
        text_embs = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection

        if normalize:
            text_embs = F.normalize(text_embs, dim=-1)  # Nc,768
        return text_embs

    @torch.no_grad()
    def get_text_embs_by_template(self, text_list):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        text_embs = []
        for text in text_list:
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.text_tokenizer([template.format(text) for template in self.template_set]).to(
                self.device
            )

            # list -> TD
            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
            #
            x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

            # take feats from the eot embedding (eot_token is the highest number in each sequence) => Nc,768；这里的意思有点像从[6,77]挑出每一行最大值对应的下标；
            text_emb = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
            text_embs.append(text_emb)
        text_embs = torch.stack(text_embs, dim=0)  # Nc,Nt,768

        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.mean(1)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs  # Nc,768

    def visual_feats_to_embs(self, x, normalize: bool = True):
        """
        将图像特征转换为图像嵌入向量
        """
        self.eval()

        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return F.normalize(x, dim=-1) if normalize else x

    @torch.no_grad()
    def get_visual_feats(self, x):
        self.eval()

        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out["stem"] = x.contiguous()  # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f"res{i + 2}"] = x.contiguous()  # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        x = self.clip_model.visual.trunk.norm_pre(x)
        out["clip_vis_dense"] = x.contiguous()
        return out

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)

class PixelNormalizer(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

        Args:
            mean (tuple, optional): the mean value. Defaults to (0.485, 0.456, 0.406).
            std (tuple, optional): the std value. Defaults to (0.229, 0.224, 0.225).
        """
        super().__init__()
        # self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        self.register_buffer(name="mean", tensor=torch.Tensor(mean).reshape(3, 1, 1))
        self.register_buffer(name="std", tensor=torch.Tensor(std).reshape(3, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean.flatten()}, std={self.std.flatten()})"

    def forward(self, x):
        """normalize x by the mean and std values

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        Albumentations:

        ```
            mean = np.array(mean, dtype=np.float32)
            mean *= max_pixel_value
            std = np.array(std, dtype=np.float32)
            std *= max_pixel_value
            denominator = np.reciprocal(std, dtype=np.float32)

            img = img.astype(np.float32)
            img -= mean
            img *= denominator
        ```
        """
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

    def set_input(self, input, mask, label_id):
        self.input = input.to(self.device)
        self.gt_mask = mask.to(self.device)
        self.label_id = label_id.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def map_classifier(self, logits, image_deep, normed_class_embs):
        prob = logits.sigmoid()
        image_embs = resize_to(image_deep, tgt_hw=prob.shape[-2:])

        # image_embs (B,C)
        image_embs = (prob * image_embs).sum((-1, -2)) / prob.sum((-1, -2))
        image_embs = image_embs[..., None, None]

        # B,C => B,D
        normed_image_embs = self.clip.visual_feats_to_embs(image_embs, normalize=True)
        class_logits = normed_image_embs @ normed_class_embs.T  # B,N
        class_logits = self.clip.clip_model.logit_scale.exp() * class_logits
        return class_logits

    def get_visual_feats(self, image, mask):
        # image = self.normalizer(image)
        # image = image * torch.sigmoid(mask)
        image_feats = self.clip.get_visual_feats(image)
        image_deep = image_feats["clip_vis_dense"]
        return image_deep

    def forward(self):

        # self.train_class_embs = self.clip.get_text_embs_by_template(self.class_names)
        # normed_class_embs = self.train_class_embs  # Nc,D
        # image_deep = self.get_visual_feats(self.input)
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(self.input)
        self.image_pe = self.get_dense_pe()

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        # self.pred_cls = self.map_classifier(low_res_masks, image_deep, normed_class_embs)
        self.pred_mask = masks

    def infer(self, input):
        bs = 1
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)
        self.image_pe = self.get_dense_pe()

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)

        return masks

    def infer_feat(self, input):
        bs = 1
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)
        self.image_pe = self.get_dense_pe()

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)

        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
