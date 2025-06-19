from models import register
from models.openclip_backbone import OpenCLIPBackbone, MultiLayerTransformerNeck
from models.sam2clip_utils import load_checkpoint_with_prefix
from models.clip2sam_utils import FPN, CrossEntropyLoss, DiceLoss, seg_loss

import copy
import os
from typing import Literal, Tuple, List, Optional, Any

import torch
from mmcv.cnn import ConvModule
from torch import nn
import torch.nn.functional as F
# from mmcv.ops import point_sample
# from mmdet.models.utils import get_uncertain_point_coords_with_randomness
# from mmdet.utils import reduce_mean
from mmengine.structures import InstanceData

from ext.sam import MaskDecoder
from ext.sam.mask_decoder import MLP as SAMMLP
from ext.meta.sam_meta import meta_dict, checkpoint_dict
import numpy as np


class OVSAMHead(nn.Module):
    def __init__(
            self,
            model_name: Literal['vit_h', 'vit_l', 'vit_b'] = 'vit_h',
            with_label_token: bool = False,
            ov_classifier_name: Optional[str] = None,
            logit: Optional[float] = None,
            roi_extractor=None,
            fix: bool = True,
            init_cfg=None,
            loss_cls=None,
            loss_mask=None,
            loss_dice=None,
            cur_mask=1,
            load_roi_conv=None,
            gen_box=False,
    ):
        assert init_cfg is not None and \
               init_cfg['type'] in ['sam_pretrain', 'Pretrained'], f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__()
        self.init_cfg = init_cfg

        mask_decoder = MaskDecoder(
            num_multimask_outputs=cur_mask - 1,
            transformer_dim=meta_dict[model_name]['prompt_embed_dim'],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            with_iou=False
        )

        if self.init_cfg['type'] == 'sam_pretrain':
            checkpoint_path = checkpoint_dict[pretrained]
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix='mask_decoder')
            ckpt_mask = len(state_dict['mask_tokens.weight'])
            if cur_mask != ckpt_mask:
                for name in ['mask_tokens.weight', 'iou_prediction_head.layers.2.weight',
                             'iou_prediction_head.layers.2.bias']:
                    state_dict[name] = state_dict[name][:cur_mask]
                for prefix in ['output_hypernetworks_mlps.']:
                    for name in list(state_dict.keys()):
                        for num in range(cur_mask, ckpt_mask):
                            if name.startswith(prefix + str(num)):
                                del state_dict[name]
                for name in list(state_dict.keys()):
                    if name.startswith('iou_'):
                        del state_dict[name]
            mask_decoder.load_state_dict(state_dict, strict=True)

        self.mask_decoder = mask_decoder

        self.with_label_token = with_label_token

        if self.with_label_token:
            # ov_path = os.path.join(os.path.expanduser('~/.cache/embd'), f"{ov_classifier_name}.pth")
            ov_path = ov_classifier_name
            cls_embed = torch.load(ov_path)
            cls_embed_norm = cls_embed.norm(p=2, dim=-1)
            assert torch.allclose(cls_embed_norm, torch.ones_like(cls_embed_norm))

            _dim = cls_embed.size(2)
            # _prototypes = cls_embed.size(1)
            # back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cpu')
            # cls_embed = torch.cat([
            #     cls_embed, back_token.repeat(_prototypes, 1)[None]
            # ], dim=0)
            self.register_buffer('cls_embed', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)

            if logit is None:
                logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            else:
                logit_scale = torch.tensor(logit, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)

            transformer_dim = self.mask_decoder.mask_tokens.weight.shape[1]
            self.label_token = nn.Embedding(1, transformer_dim)
            self.label_mlp = SAMMLP(transformer_dim, transformer_dim, _dim, 3)

            if loss_cls is not None:
                _loss_cls = copy.deepcopy(loss_cls)
                _loss_cls.update(class_weight=[1.]*self.cls_embed.shape[1])
                self.loss_cls = CrossEntropyLoss(
                    use_sigmoid=_loss_cls['use_sigmoid'],
                    loss_weight=_loss_cls['loss_weight'],
                    reduction=_loss_cls['reduction'],
                    class_weight=_loss_cls['class_weight']
                )
                self.register_buffer('class_weight', torch.tensor(self.loss_cls.class_weight), persistent=False)
            else:
                self.loss_cls = None

            if loss_mask is not None:
                self.loss_mask = CrossEntropyLoss(
                    use_sigmoid=loss_mask['use_sigmoid'],
                    loss_weight=loss_mask['loss_weight'],
                    reduction=loss_mask['reduction']
                )
            else:
                self.loss_mask = None

            if loss_dice is not None:
                self.loss_dice = DiceLoss(
                    use_sigmoid=loss_dice['use_sigmoid'],
                    activate=loss_dice['activate'],
                    reduction=loss_dice['reduction'],
                    naive_dice=loss_dice['naive_dice'],
                    eps=loss_dice['eps'],
                    loss_weight=loss_dice['loss_weight']
                )
            else:
                self.loss_dice = None

            self.criterionBCE = torch.nn.BCEWithLogitsLoss()


        # meta
        # self.num_points = 12544
        # self.oversample_ratio = 3.
        # self.importance_sample_ratio = .75
        #
        # self.gen_box = gen_box
        #
        # if roi_extractor is not None:
        #     self.roi = MODELS.build(roi_extractor)
        #     self.roi_conv = nn.Sequential(*[
        #         ConvModule(in_channels=self.roi.out_channels, out_channels=_dim, kernel_size=1, bias=False)
        #     ])
        # else:
        #     self.roi = None

        if self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=self.init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)

        self.fix = fix

        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    def init_weights(self):
        self.logger.info(f"Init Config for {self.__class__.__name__}")
        self.logger.info(self.init_cfg)

    def forward_logit(self, cls_embd, normed_class_embs):
        cls_pred = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), normed_class_embs)
        cls_pred = cls_pred.max(-1).values
        cls_pred = self.logit_scale.exp() * cls_pred
        return cls_pred

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
        # masks = F.interpolate(
        #     masks,
        #     (self.image_encoder.img_size, self.image_encoder.img_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            fpn_feats: List[torch.Tensor],
            normed_class_embs: torch.Tensor,
            backbone_feature: torch.Tensor,
            backbone=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        num_instances = int(sparse_prompt_embeddings.size(0))
        # Concatenate output tokens
        output_tokens = torch.cat([
            self.label_token.weight,
            self.mask_decoder.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(num_instances, -1, -1)
        queries = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # image_embeddings = torch.repeat_interleave(image_embeddings, num_instances, dim=0)
        image_embeddings = image_embeddings + dense_prompt_embeddings
        pos_img = torch.repeat_interleave(image_pe, num_instances, dim=0)
        b, c, h, w = image_embeddings.shape

        # Run the transformer
        queries, mask_feats = self.mask_decoder.transformer(image_embeddings, pos_img, queries)
        label_query = queries[:, 0, :]
        mask_embeds = queries[:, 1:(1 + self.mask_decoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        mask_feats = mask_feats.transpose(1, 2).view(b, c, h, w)
        mask_feats = self.mask_decoder.output_upscaling(mask_feats)
        mask_queries_list: List[torch.Tensor] = []
        for i in range(self.mask_decoder.num_mask_tokens):
            mask_queries_list.append(self.mask_decoder.output_hypernetworks_mlps[i](mask_embeds[:, i, :]))
        mask_queries = torch.stack(mask_queries_list, dim=1)
        b, c, h, w = mask_feats.shape
        masks = (mask_queries @ mask_feats.view(b, c, h * w)).view(b, -1, h, w)

        # Generate class labels
        if self.with_label_token:
            cls_embed_list = []
            assert self.mask_decoder.num_mask_tokens == 1
            for i in range(self.mask_decoder.num_mask_tokens):
                cls_embed_list.append(self.label_mlp(label_query))
            cls_embed = torch.stack(cls_embed_list, dim=1)

            # if self.gen_box:
            #     bboxes = mask2bbox(masks.sigmoid()[:, 0] > 0.5) * 4
            #     roi_list = bbox2roi([bboxes])
            # roi_feats = self.roi(fpn_feats, roi_list)
            # roi_feats = self.roi_conv(roi_feats)
            # roi_feats = roi_feats.mean(dim=-1).mean(dim=-1)
            # roi_feats = roi_feats[:, None] + 0 * cls_embed
            cls_pred = self.forward_logit(cls_embed, normed_class_embs)
        else:
            cls_pred = None
        return masks, None, cls_pred

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multi_mask_output=None,
            data_samples=None,
            fpn_feats=None,
            backbone_feats=None,
            backbone=None,
            gt=None,
            label=None,
            normed_class_embs=None
    ):
        # num_prompts = len(sparse_prompt_embeddings)
        # image_embeddings = torch.repeat_interleave(image_embeddings, num_prompts, dim=0)

        masks, _, cls_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            fpn_feats=fpn_feats,
            normed_class_embs=normed_class_embs,
            backbone_feature=backbone_feats,
            backbone=backbone
        )

        # Select the correct mask or masks for output
        if multi_mask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]

        # 对输出的mask 进行 interpolate
        masks_bs, masks_c, masks_input_h, _ = masks.shape
        original_bs, original_c, original_h, _ = gt.shape
        masks_input_size = (masks_input_h, masks_input_h)
        original_size = (original_h, original_h)
        masks = self.postprocess_masks(masks, masks_input_size, original_size)

        # Prepare output
        return masks, cls_pred

    def forward_train(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            gt=None,
            label=None,
            normed_class_embs=None,
            fpn_feats=None,
            backbone_feats=None,
            backbone=None,
    ):
        image_embed = image_embeddings

        masks, _, cls_preds = self.predict_masks(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            fpn_feats=fpn_feats,
            normed_class_embs=normed_class_embs,
            backbone_feature=backbone_feats,
            backbone=backbone,
        )
        #对输出的mask 进行 interpolate
        masks_bs, masks_c, masks_input_h, _ = masks.shape
        original_bs, original_c, original_h, _ = gt.shape
        masks_input_size = (masks_input_h, masks_input_h)
        original_size = (original_h, original_h)
        masks = self.postprocess_masks(masks, masks_input_size, original_size)

        # prepare gt
        # instances = []
        # for data_sample in data_samples:
        #     if 'masks' in data_sample.gt_instances:
        #         instances.append(InstanceData(
        #             labels=data_sample.gt_instances.labels,
        #             masks=data_sample.gt_instances.masks
        #         ))
        #     else:
        #         instances.append(InstanceData(
        #             labels=data_sample.gt_instances.labels,
        #         ))
        # gt_instances = InstanceData.cat(instances)
        # assert len(gt_instances) == len(image_embed)
        #
        # device = image_embed.device
        #
        # cls_scores = cls_preds[:, 0]
        # gt_labels = gt_instances.labels
        # iou_mask = gt_labels.eq(-1)
        # score_mask = torch.logical_not(iou_mask)
        # gt_labels[iou_mask] = 0

        cls_scores = cls_preds[:, 0]
        gt_labels = label
        score_mask = torch.ones_like(gt_labels).to(gt_labels.device)
        loss_cls = self.loss_cls(
            cls_scores,
            gt_labels,
            score_mask.float(),
            avg_factor=self.class_weight[gt_labels].sum()
        )

        # pred_masks = masks[:, 0:1]
        # num_masks = len(pred_masks)
        # mask_avg_factor = reduce_mean(cls_scores.new_tensor([num_masks]))
        # mask_avg_factor = mask_avg_factor.clamp(min=1)
        # if 'masks' not in gt_instances:
        #     loss_dice = pred_masks.sum() * 0
        #     loss_mask = pred_masks.sum() * 0
        # else:
        #     gt_masks = gt_instances.masks.to_tensor(dtype=torch.float, device=device)[:, None]
        #     with torch.no_grad():
        #         uncertain_points = get_uncertain_point_coords_with_randomness(
        #             pred_masks, None, self.num_points, self.oversample_ratio, self.importance_sample_ratio)
        #     pred_masks = point_sample(pred_masks, uncertain_points)
        #     gt_masks = point_sample(gt_masks, uncertain_points)
        #     loss_dice = self.loss_dice(
        #         pred_masks, gt_masks, avg_factor=mask_avg_factor
        #     )
        #
        #     pred_masks = pred_masks.reshape(-1)
        #     gt_masks = gt_masks.reshape(-1)
        #     loss_mask = self.loss_mask(
        #         pred_masks,
        #         gt_masks,
        #         avg_factor=mask_avg_factor * self.num_points
        #     )

        loss_dice = self.criterionBCE(masks, gt)
        loss_iou = seg_loss(masks, gt)
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = loss_cls
        loss_dict['loss_dice'] = loss_dice
        loss_dict['loss_iou'] = loss_iou
        # loss_dict['loss_mask'] = loss_mask
        # sum_loss = loss_dict['loss_cls'] + 5.0 * (loss_dict['loss_dice'] + loss_dict['loss_iou'])
        sum_loss = loss_dict['loss_dice'] + loss_dict['loss_iou']
        return loss_dict, sum_loss

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

@register('CLIP2SAM')
class CLIP2SAM(nn.Module):
    MASK_THRESHOLD = 0.5

    def __init__(
            self,
            backbone: dict,
            neck: dict,
            # prompt_encoder: dict = None,
            mask_decoder: dict = None,
            fpn_neck: dict = None,
            with_box: bool = False,
            with_points: bool = False
    ) -> None:
        super().__init__()

        self.backbone = OpenCLIPBackbone(
            model_name=backbone['model_name'],
            fix=backbone['fix'],
            init_cfg=backbone['init_cfg']
        )
        self.neck = MultiLayerTransformerNeck(
            input_size=tuple(neck['input_size']),
            in_channels=neck['in_channels'],
            strides=neck['strides'],
            layer_ids=neck['layer_ids'],
            embed_channels=neck['embed_channels'],
            out_channels=neck['out_channels'],
            # embedding_path=neck['embedding_path'],
            init_cfg=neck['init_cfg'],
            fix=neck['fix']
        )

        if fpn_neck is not None:
            self.fpn_neck = FPN(
                in_channels=fpn_neck['in_channels'],
                out_channels=fpn_neck['out_channels'],
                num_outs=fpn_neck['num_outs']
            )
        else:
            self.fpn_neck = None

        # self.pe = MODELS.build(prompt_encoder)
        self.mask_decoder = OVSAMHead(
            model_name=mask_decoder['model_name'],
            with_label_token=mask_decoder['with_label_token'],
            ov_classifier_name=mask_decoder['ov_classifier_name'],
            # roi_extractor=mask_decoder['roi_extractor'],
            fix=mask_decoder['fix'],
            init_cfg=mask_decoder['init_cfg'],
            loss_cls=mask_decoder['loss_cls'],
            loss_mask=mask_decoder['loss_mask'],
            loss_dice=mask_decoder['loss_dice']
        )
        self.pe_layer = PositionEmbeddingRandom(neck['out_channels'] // 2)
        self.no_mask_embed = nn.Embedding(1, neck['out_channels'])

        self.with_box = with_box
        self.with_points = with_points

    def get_dense_pe(self, image_embedding_size):
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(image_embedding_size).unsqueeze(0)

    def forward(self, inputs, gt, label, known_class_names, mode):
        if mode == 'loss':
            normed_class_embs = self.mask_decoder.cls_embed.to(inputs.device)
            return self.loss(inputs, gt, label, normed_class_embs)
        elif mode == 'predict':
            ov_path = "/media/estar/Data/ywb/OVCamoDataset/test/ViT-B-16_ovcamo_test.pth"
            normed_class_embs = torch.load(ov_path)
            normed_class_embs = normed_class_embs.permute(2, 0, 1).contiguous().to(inputs.device)
            return self.predict(inputs, gt, normed_class_embs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def extract_feat(self, batch_inputs):
        x = self.neck(self.backbone(batch_inputs))
        return x

    def _forward(self, batch_inputs, batch_data_samples):
        raise NotImplementedError

    def predict(self, batch_inputs, gt, normed_class_embs):
        backbone_feats = self.backbone(batch_inputs)
        feats = self.neck(backbone_feats)

        if self.fpn_neck is not None:
            fpn_feats = self.fpn_neck(backbone_feats)
        else:
            fpn_feats = None

        bs, embed_dim, image_embedding_size, _ = feats.shape
        sparse_embed = torch.empty((bs, 0, embed_dim), device=batch_inputs.device)
        dense_embed = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, image_embedding_size, image_embedding_size
        )

        pred_masks, pred_cls = self.mask_decoder.forward(
                image_embeddings=feats,
                image_pe=self.get_dense_pe(image_embedding_size),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                normed_class_embs=normed_class_embs,
                fpn_feats=fpn_feats,
                gt=gt
        )
        if pred_cls is not None:
            pred_cls = pred_cls[:, 0].softmax(-1)[:, :-1].argmax(-1)
        return pred_masks, pred_cls

    def loss(self, batch_inputs, gt, label, normed_class_embs):
        backbone_feats = self.backbone(batch_inputs)
        feats = self.neck(backbone_feats)

        if self.fpn_neck is not None:
            fpn_feats = self.fpn_neck(backbone_feats)
        else:
            fpn_feats = None

        bs, embed_dim, image_embedding_size, _ = feats.shape
        sparse_embed = torch.empty((bs, 0, embed_dim), device=batch_inputs.device)
        dense_embed = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, image_embedding_size, image_embedding_size
        )

        if fpn_feats is not None:
            losses, sum_loss = self.mask_decoder.forward_train(
                image_embeddings=feats,
                image_pe=self.get_dense_pe(image_embedding_size),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                normed_class_embs=normed_class_embs,
                fpn_feats=fpn_feats,
                gt=gt,
                label=label
            )
        else:
            losses, sum_loss = self.mask_decoder.forward_train(
                image_embeddings=feats,
                image_pe=self.pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                batch_ind_list=batch_ind_list,
                data_samples=batch_data_samples,
            )
        return losses, sum_loss

    def optimize_parameters(self, inputs, gt, label, known_class_names, mode):
        losses, sum_loss = self.forward(inputs, gt, label, known_class_names, mode)
        self.sum_loss = sum_loss
        self.loss_dict = losses
        self.optimizer.zero_grad()  # set G's gradients to zero
        sum_loss.backward()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights
