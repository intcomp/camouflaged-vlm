import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer, TwoWayTransformer_MaskDecoder_Edge, MaskDecoder_Edge


from models.ovcamo_loss import edge_dice_loss

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from dassl.utils import load_checkpoint

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

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

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

@register('sam_maskdecoder_edge')
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
        self.mask_decoder = MaskDecoder_Edge(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer_MaskDecoder_Edge(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

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

        self.sam_visual_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
        )

        self.sam_text_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
        )

        self.train_text_features = torch.load(
            "./datasets/ovcamo_info/TrainCamoPromptsTextFeaturesViTB-14-336.pth").to(
            self.device)
        self.test_text_features = torch.load(
            "./datasets/ovcamo_info/TestCamoPromptsTextFeaturesViTB-14-336.pth").to(
            self.device)

    def load_mapleAlphaCLIP(self, maple_clip_model, MaPLeAlphaCLIP_checkpoint=None):
        self.clip_model = maple_clip_model
        self.clip_model = self.clip_model.float()
        for k, p in self.clip_model.named_parameters():
            p.requires_grad = False
        self.clip_model.to(self.device)
        self.clip_model.load_text_features(self.train_text_features, self.test_text_features)
        #
        if MaPLeAlphaCLIP_checkpoint != None:
            checkpoint = load_checkpoint(MaPLeAlphaCLIP_checkpoint)
            state_dict = checkpoint["state_dict"]
            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            # set strict=False
            self.clip_model.load_state_dict(state_dict, strict=False)

    def set_input(self, input, mask, label_id, clip_image, clip_mask):
        self.input = input.to(self.device)
        self.gt_mask = mask.to(self.device)
        self.label_id = label_id.to(self.device)
        self.clip_image = clip_image.to(self.device)
        self.clip_mask = clip_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def alpha_clip_process(self, image, alpha):
        if self.training:
            text_embeddings = self.train_text_features  #(14, 768)
        else:
            text_embeddings = self.test_text_features  #(61, 768)
        # text_embeddings = self.train_text_features  # (14, 768)
        image_features = self.clip_model.visual(image, alpha)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = torch.matmul(image_features, text_embeddings.permute(1, 0))
        pred_1 = score.topk(1, dim=1)[1].squeeze(dim=1)
        # 1.
        # label smoothing
        smoothing = 0.1
        confidence = 1 - smoothing
        smooth_score = torch.zeros_like(score).to(score.device)
        smooth_score.fill_(smoothing)
        smooth_score.scatter_(1, pred_1.unsqueeze(1), confidence)
        # 扩展 smooth_score 的维度
        smooth_score = smooth_score.unsqueeze(-1)  # [1, 14, 1]
        output_text_features = (smooth_score * text_embeddings).sum(dim=1)
        # 进行广播相乘并求和
        # 2.
        # score = torch.sigmoid(score)
        # output_text_features = (score.unsqueeze(-1) * text_embeddings).sum(dim=1)  # [1, 768]
        # 3.
        # get logits
        # distances = score
        # temperature = 0.5
        # top_p_logits = 0.7
        # filter_value = float('-inf')
        # soft_code = F.softmax(-distances / temperature, dim=-1)
        # # get top-p
        # sorted_logits, sorted_indices = torch.sort(soft_code, descending=True, dim=-1)
        # cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        # # Remove tokens with cumulative probability above the threshold
        # sorted_indices_to_remove = cumulative_probs > top_p_logits
        # # Shift the indices to the right to keep also the first token above the threshold
        # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # sorted_indices_to_remove[..., 0] = 0
        #
        # indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        # soft_code = soft_code.masked_fill(indices_to_remove, filter_value)
        # soft_code = F.softmax(soft_code, dim=-1)  # (b, 128)
        # output_text_features = (soft_code.unsqueeze(-1) * text_embeddings).sum(dim=1)

        return image_features.unsqueeze(1), output_text_features.unsqueeze(1), pred_1

    def maple_alpha_clip_process(self, image, alpha):
        image_features, text_features, pred_1, score = self.clip_model(image, alpha, self.training)
        return image_features, text_features, pred_1, score

    def forward(self):
        bs = 1
        # Embed prompts
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        features, interm_embeddings = self.image_encoder(self.input, interm=True)
        image_pe = self.get_dense_pe()

        #第一阶段
        image_feat_1, text_feat_1, pred_1, score = self.maple_alpha_clip_process(self.clip_image, self.clip_mask)
        image_feat_1 = self.sam_visual_proj(image_feat_1)
        text_feat_1 = self.sam_text_proj(text_feat_1)
        sparse_embeddings_1 = torch.cat((image_feat_1, text_feat_1), dim=1)

        # Predict masks
        low_res_masks, low_res_edges, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            interm_embeddings=interm_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings_1,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks1 = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        edges1 = self.postprocess_masks(low_res_edges, self.inp_size, self.inp_size)

        self.pred_mask = masks1
        self.pred_edge = edges1
        # self.score = score

    def infer(self, input, clip_image, clip_zero_mask):
        bs = 1

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        features, interm_embeddings = self.image_encoder(input, interm=True)
        image_pe = self.get_dense_pe()

        image_feat_1, text_feat_1, pred_1, score = self.maple_alpha_clip_process(clip_image, clip_zero_mask)
        image_feat_1 = self.sam_visual_proj(image_feat_1)
        text_feat_1 = self.sam_text_proj(text_feat_1)
        sparse_embeddings_1 = torch.cat((image_feat_1, text_feat_1), dim=1)

        # Predict masks
        low_res_masks, low_res_edges, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            interm_embeddings=interm_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings_1,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def infer_test(self, input, clip_image, clip_zero_mask):
        bs = input.shape[0]

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        features, interm_embeddings = self.image_encoder(input, interm=True)
        image_pe = self.get_dense_pe()

        # 第一阶段
        image_feat_1, text_feat_1, pred_1, score = self.maple_alpha_clip_process(clip_image, clip_zero_mask)
        image_feat_1 = self.sam_visual_proj(image_feat_1)
        text_feat_1 = self.sam_text_proj(text_feat_1)
        sparse_embeddings_1 = torch.cat((image_feat_1, text_feat_1), dim=1)

        # Predict masks
        low_res_masks, low_res_edges, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            interm_embeddings=interm_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings_1,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
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

    def backward_G_other(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_masks = 0.
        self.loss_fuse = 0.
        self.loss_dict = {}
        #loss masks
        self.loss_masks += self.criterionBCE(self.pred_mask['masks'], self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_masks += _iou_loss(self.pred_mask['masks'], self.gt_mask)
        self.loss_dict['masks'] = self.loss_masks
        #loss fuse
        self.loss_fuse += self.criterionBCE(self.pred_mask['masks_fuse'], self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_fuse += _iou_loss(self.pred_mask['masks_fuse'], self.gt_mask)
        self.loss_dict['masks_fuse'] = self.loss_fuse

        self.loss_G = self.loss_masks + self.loss_fuse
        self.loss_G.backward()

    def backward_G_class(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_mask1 = 0.
        self.loss_mask2 = 0.
        self.loss_dict = {}
        #loss mask1
        self.loss_mask1 += self.criterionBCE(self.pred_mask[0], self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_mask1 += _iou_loss(self.pred_mask[0], self.gt_mask)
        self.loss_dict['mask1'] = self.loss_mask1

        #loss mask2
        self.loss_mask2 += self.criterionBCE(self.pred_mask[1], self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_mask2 += _iou_loss(self.pred_mask[1], self.gt_mask)
        self.loss_dict['mask2'] = self.loss_mask2

        self.loss_G = self.loss_mask1 + self.loss_mask2
        self.loss_G.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_dict = {}
        self.loss_G = 0
        # seg loss
        self.loss_dict["loss_mask"] = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_dict["loss_mask"] += _iou_loss(self.pred_mask, self.gt_mask)

        self.loss_G += self.loss_dict["loss_mask"]
        # edge loss
        with torch.no_grad():
            edge_ks = 5
            eroded_mask = -F.max_pool2d(-self.gt_mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            dilated_mask = F.max_pool2d(self.gt_mask, kernel_size=edge_ks, stride=1, padding=edge_ks // 2)
            edge = dilated_mask - eroded_mask
            edge = edge.gt(0).float()
        self.gt_edge = edge
        self.loss_dict["loss_edge"] = edge_dice_loss(self.pred_edge, self.gt_edge)
        self.loss_G += self.loss_dict["loss_edge"]
        # cls loss
        # self.loss_dict["loss_cls"] = F.cross_entropy(self.score, self.label_id)
        # self.loss_G += self.loss_dict["loss_cls"]
        # self.loss_dict["loss_cls"] = 0

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
