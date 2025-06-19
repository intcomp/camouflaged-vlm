from typing import Tuple, List, Optional
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from timm.layers import resample_abs_pos_embed

import open_clip as open_clip
# from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix
from models import register
from functools import partial

from models.sam2clip_utils import normal_init, load_checkpoint_with_prefix, PatchEmbed, MSELoss
from ext.meta.sam_meta import checkpoint_dict
from ext.sam.common import LayerNorm2d
from ext.sam.image_encoder import Block
# from mmcv.cnn.bricks.wrappers import ConvTranspose2d
logger = logging.getLogger(__name__)

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


def flatten_permute(x):
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    return x

@register('openclipbackbone')
class OpenCLIPBackbone(nn.Module):
    """OpenCLIPBackbone,
    Please refer to:
    https://github.com/mlfoundations/open_clip/tree/5f7892b672b21e6853d0f6c11b18dda9bcf36c8d#pretrained-model-interface
    for the supported models and checkpoints.
    """
    STAGES = 4

    def __init__(
            self,
            img_size: int = 1024,
            model_name: str = '',
            fix: bool = True,
            fix_layers: Optional[List] = None,
            init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] in ['clip_pretrain', 'image_pretrain', 'Pretrained'], \
            f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__()
        self.init_cfg = init_cfg

        # Get the clip model
        if init_cfg['type'] == 'clip_pretrain':
            clip_model = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained,return_transform=False)
        elif init_cfg['type'] == 'image_pretrain':
            clip_model = open_clip.create_model(model_name, pretrained_image=True, logger=self.logger)
        elif init_cfg['type'] == 'Pretrained':
            clip_model = open_clip.create_model(model_name, pretrained_image=False, logger=self.logger)
        else:
            raise NotImplementedError
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        self.out_indices = (0, 1, 2, 3)
        model_name_lower = model_name.lower()
        if 'convnext_' in model_name_lower:
            model_type = 'convnext'
            if '_base' in model_name_lower:
                output_channels = [128, 256, 512, 1024]
                feat_size = 0
            elif '_large' in model_name_lower:
                output_channels = [192, 384, 768, 1536]
                feat_size = 0
            elif '_xxlarge' in model_name_lower:
                output_channels = [384, 768, 1536, 3072]
                feat_size = 0
            else:
                raise NotImplementedError(f"{model_name} not supported yet.")
        elif 'rn' in model_name_lower:
            model_type = 'resnet'
            if model_name_lower.replace('-quickgelu', '') in ['rn50', 'rn101']:
                output_channels = [256, 512, 1024, 2048]
                feat_size = 7
            elif model_name_lower == 'rn50x4':
                output_channels = [320, 640, 1280, 2560]
                feat_size = 9
            elif model_name_lower == 'rn50x16':
                output_channels = [384, 768, 1536, 3072]
                feat_size = 12
            elif model_name_lower == 'rn50x64':
                output_channels = [512, 1024, 2048, 4096]
                feat_size = 14
            else:
                raise NotImplementedError(f"{model_name} not supported yet.")
        elif "vit" in model_name_lower:
            model_type = 'vit'
            if model_name_lower == 'vit-l-14':
                output_channels = [1024, 1024, 1024, 1024]
                feat_size = 0
                # assert not clip_model.visual.input_patchnorm
                # assert clip_model.visual.attn_pool is None
            elif model_name_lower == 'vit-b-16':
                output_channels = [768, 768, 768, 768]
                feat_size = 0
            else:
                raise NotImplementedError(f"{model_name} not supported yet.")
        else:
            raise NotImplementedError(f"{model_name} not supported yet.")

        self.model_name = model_name
        self.fix = fix
        self.model_type = model_type
        self.output_channels = output_channels
        self.feat_size = feat_size

        # Get the visual model
        if self.model_type == 'resnet':
            self.stem = nn.Sequential(*[
                clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.act1,
                clip_model.visual.conv2, clip_model.visual.bn2, clip_model.visual.act2,
                clip_model.visual.conv3, clip_model.visual.bn3, clip_model.visual.act3,
            ])
        elif self.model_type == 'convnext':
            self.stem = clip_model.visual.trunk.stem
        elif self.model_type == 'vit':
            self.stem = clip_model.visual.conv1
        else:
            raise ValueError

        if self.model_type == 'resnet':
            self.avgpool = clip_model.visual.avgpool
        elif self.model_type == 'convnext':
            self.avgpool = nn.Identity()
        elif self.model_type == 'vit':
            self.avgpool = flatten_permute
        else:
            raise ValueError

        self.res_layers = []
        if self.model_type in ['vit']:
            self.t_class_embedding = clip_model.visual.class_embedding
            self.t_positional_embedding = clip_model.visual.positional_embedding
            self.t_ln_pre_trans = clip_model.visual.ln_pre
            self.t_transformer = clip_model.visual.transformer
        else:
            for i in range(self.STAGES):
                if self.model_type == 'resnet':
                    layer_name = f'layer{i + 1}'
                    layer = getattr(clip_model.visual, layer_name)
                elif self.model_type == 'convnext':
                    layer_name = f'layer{i + 1}'
                    layer = clip_model.visual.trunk.stages[i]
                else:
                    raise ValueError
                self.add_module(layer_name, layer)
                self.res_layers.append(layer_name)

        if self.model_type == 'resnet':
            self.norm_pre = nn.Identity()
        elif self.model_type == 'convnext':
            self.norm_pre = clip_model.visual.trunk.norm_pre
        elif self.model_type == 'vit':
            self.norm_pre = nn.Identity()

        if self.model_type == 'resnet':
            self.head = clip_model.visual.attnpool
        elif self.model_type == 'convnext':
            self.head = nn.Sequential(*[
                clip_model.visual.trunk.head,
                clip_model.visual.head,
            ])
        elif self.model_type == 'vit':
            self.head = clip_model.visual.ln_post

        self.fix_layers = fix_layers

        if not self.fix:
            self.train()
            for name, param in self.norm_pre.named_parameters():
                param.requires_grad = False
            for name, param in self.head.named_parameters():
                param.requires_grad = False
            if self.fix_layers is not None:
                for i, layer_name in enumerate(self.res_layers):
                    if i in self.fix_layers:
                        res_layer = getattr(self, layer_name)
                        for name, param in res_layer.named_parameters():
                            param.requires_grad = False
                        if i == 0:
                            for name, param in self.stem.named_parameters():
                                param.requires_grad = False

        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    def init_weights(self):
        self.logger.info(f"Init Config for {self.model_name}")
        self.logger.info(self.init_cfg)

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.fix:
            super().train(mode=False)
        else:
            super().train(mode=mode)
            if self.fix_layers is not None:
                for i, layer_name in enumerate(self.res_layers):
                    if i in self.fix_layers:
                        res_layer = getattr(self, layer_name)
                        res_layer.train(mode=False)
                        if i == 0:
                            self.stem.train(mode=False)
        return self

    def forward_func(self, x):
        x = self.stem(x)
        h, w = x.shape[-2:]
        x = self.avgpool(x)
        outs = []
        if self.model_type == 'vit':
            x = torch.cat(
                [self.t_class_embedding.to(x.dtype) +
                 torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1
            )  # shape = [*, grid ** 2 + 1, width]
            new_pos_embed = resample_abs_pos_embed(
                self.t_positional_embedding[None],
                [h, w],
                num_prefix_tokens=1
            )
            x = x + new_pos_embed.to(x.dtype)
            x = self.t_ln_pre_trans(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.t_transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x[:, 1:]
            x = x.permute(0, 2, 1).unflatten(2, (h, w))  # BCHW
            for i in range(self.STAGES):
                outs.append(
                    F.interpolate(
                        x,
                        scale_factor=2 ** (2 - i),
                        mode='bilinear',
                        align_corners=False
                    )
                )
        else:
            for i, layer_name in enumerate(self.res_layers):
                res_layer = getattr(self, layer_name)
                x = res_layer(x).contiguous()
                if i in self.out_indices:
                    outs.append(x)
        return tuple(outs)

    def get_clip_feature(self, backbone_feat):
        if self.model_type == 'resnet':
            return backbone_feat
        elif self.model_type == 'convnext':
            return self.norm_pre(backbone_feat)
        raise NotImplementedError

    def get_text_embs_by_template(self, text_list):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()
        template_set = get_prompt_template_by_name("camoprompts")
        text_embs = []
        for text in text_list:
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.text_tokenizer([template.format(text) for template in template_set]).to(
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

    def forward_feat(self, features):
        if self.model_type == 'convnext':
            batch, num_query, channel = features.shape
            features = features.reshape(batch * num_query, channel, 1, 1)
            features = self.head(features)
            return features.view(batch, num_query, features.shape[-1])
        elif self.model_type == 'resnet':
            num_query, channel, seven, seven = features.shape
            features = self.head(features)
            return features

    def forward(self, x):
        if self.fix:
            with torch.no_grad():
                outs = self.forward_func(x)
        else:
            outs = self.forward_func(x)
        return outs

    def get_text_model(self):
        return OpenCLIPBackboneText(
            self.model_name,
            init_cfg=self.init_cfg
        )

class OpenCLIPBackboneText(nn.Module):
    def __init__(
            self,
            model_name: str = '',
            init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] == 'clip_pretrain', f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__()
        self.init_cfg = init_cfg
        # Get the clip model
        clip_model = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained, return_transform=False)

        # Get the textual model
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        self.text_transformer = clip_model.transformer
        self.text_token_embedding = clip_model.token_embedding
        self.text_pe = clip_model.positional_embedding
        self.text_ln_final = clip_model.ln_final
        self.text_proj = clip_model.text_projection

        self.register_buffer('text_attn_mask', clip_model.attn_mask)

        self.param_dtype = torch.float32
        self.model_name = model_name

    def init_weights(self):
        self.logger.info(f"Init Config for {self.model_name}")
        self.logger.info(self.init_cfg)

    # Copied from
    # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L343
    @torch.no_grad()
    def forward(self, text):
        text_tokens = self.text_tokenizer(text).to(device=self.text_proj.device)
        x = self.text_token_embedding(text_tokens).to(self.param_dtype)
        x = x + self.text_pe.to(self.param_dtype)
        x = x.permute(1, 0, 2)
        x = self.text_transformer(x, attn_mask=self.text_attn_mask)
        x = x.permute(1, 0, 2)
        x = self.text_ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_proj
        return x

class MultiLayerTransformerNeck(nn.Module):
    STRIDE = 16

    def __init__(
            self,
            input_size: Tuple[int, int],
            in_channels: List[int],
            embed_channels: int,
            out_channels: int,
            layer_ids: Tuple[int] = (0, 1, 2, 3),
            strides: Tuple[int] = (4, 8, 16, 32),
            embedding_path: Optional[str] = None,
            fix=False,
            init_cfg=None
    ) -> None:
        super().__init__()

        self.transformer_size = (input_size[0] // self.STRIDE, input_size[1] // self.STRIDE)
        self.layer_ids = layer_ids

        self.patch_embeds = nn.ModuleList()
        for idx, in_ch in enumerate(in_channels):
            if idx in layer_ids:
                if strides[idx] > self.STRIDE:
                    patch_embed = PatchEmbed(
                        conv_type='ConvTranspose2d',
                        in_channels=in_ch,
                        embed_dims=embed_channels,
                        kernel_size=strides[idx] // self.STRIDE,
                        stride=strides[idx] // self.STRIDE,
                        input_size=(input_size[0] // strides[idx], input_size[1] // strides[idx])
                    )
                else:
                    patch_embed = PatchEmbed(
                        in_channels=in_ch,
                        embed_dims=embed_channels,
                        kernel_size=self.STRIDE // strides[idx],
                        stride=self.STRIDE // strides[idx],
                        input_size=(input_size[0] // strides[idx], input_size[1] // strides[idx])
                    )
                self.patch_embeds.append(patch_embed)
            else:
                self.patch_embeds.append(nn.Identity())

        if embedding_path is not None:
            assert embedding_path.startswith('sam_')
            embedding_ckpt = embedding_path.split('_', maxsplit=1)[1]
            path = checkpoint_dict[embedding_ckpt]
            state_dict = load_checkpoint_with_prefix(path, prefix='image_encoder')
            pos_embed = state_dict['pos_embed']
        else:
            # For loading from checkpoint
            pos_embed = torch.zeros(1, input_size[0] // self.STRIDE, input_size[1] // self.STRIDE, embed_channels)

        self.register_buffer('pos_embed', pos_embed)

        self.level_encoding = nn.Embedding(len(layer_ids), embed_channels)

        depth = 5
        global_attn_indexes = [4]
        window_size = 14

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_channels,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                use_rel_pos=True,
                rel_pos_zero_init=True,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=self.transformer_size,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )

        self.fix = fix
        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

        if init_cfg is not None:
            assert init_cfg['type'] == 'Pretrained'
            checkpoint_path = init_cfg['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)
            self._is_init = True

    def init_weights(self):
        normal_init(self.level_encoding, mean=0, std=1)

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.fix:
            super().train(mode=False)
        else:
            super().train(mode=mode)
        return self

    def forward(self, inputs):
        input_embeddings = []
        level_cnt = 0
        for idx, feat in enumerate(inputs):
            if idx not in self.layer_ids:
                continue
            feat, size = self.patch_embeds[idx](feat)
            feat = feat.unflatten(1, size)
            feat = feat + self.level_encoding.weight[level_cnt]
            input_embeddings.append(feat)
            level_cnt += 1

        feat = sum(input_embeddings)
        feat = feat + self.pos_embed
        for block in self.blocks:
            feat = block(feat)
        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat = self.neck(feat)
        return feat

@register('SAM2CLIP')
class BackboneDistillation(nn.Module):
    def __init__(
            self,
            backbone_teacher: dict,
            backbone_student: dict,
            # neck_teacher: ConfigType,
            neck_student: dict,
            loss_distill: dict,
            add_adapter: bool = False,
            use_cache: bool = False,
            # init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__()
        self.use_cache = use_cache
        # if not self.use_cache:
        #     self.backbone_teacher = MODELS.build(backbone_teacher)
        #     self.neck_teacher = MODELS.build(neck_teacher)
        # else:
        #     self.cache_suffix = f'_{backbone_teacher.model_name}_cache.pth'

        self.backbone_student = OpenCLIPBackbone(
            model_name=backbone_student['model_name'],
            fix=backbone_student['fix'],
            init_cfg=backbone_student['init_cfg']
        )
        self.neck_student = MultiLayerTransformerNeck(
            input_size=tuple(neck_student['input_size']),
            in_channels=neck_student['in_channels'],
            strides=neck_student['strides'],
            layer_ids=neck_student['layer_ids'],
            embed_channels=neck_student['embed_channels'],
            out_channels=neck_student['out_channels'],
            embedding_path=neck_student['embedding_path']
        )

        self.loss_distill = MSELoss(
            reduction=loss_distill['reduction'],
            loss_weight=loss_distill['loss_weight']
        )

        self.add_adapter = add_adapter
        # if self.add_adapter:
        #     STRIDE = 16
        #     self.patch_embeds = nn.ModuleList()
        #     for stride in [4, 8, 16, 32]:
        #         if stride > 16:
        #             patch_embed = build_conv_layer(
        #                 dict(type=nn.ConvTranspose2d),
        #                 in_channels=256,
        #                 out_channels=256,
        #                 kernel_size=stride // STRIDE,
        #                 stride=stride // STRIDE,
        #                 padding=0,
        #                 dilation=1,
        #                 bias=0
        #             )
        #         else:
        #             patch_embed = build_conv_layer(
        #                 dict(type=nn.Conv2d),
        #                 in_channels=256,
        #                 out_channels=256,
        #                 kernel_size=STRIDE // stride,
        #                 stride=STRIDE // stride,
        #                 padding=0,
        #                 dilation=1,
        #                 bias=0
        #             )
        #         self.patch_embeds.append(patch_embed)

        logger.info(
            f"teacher: {backbone_teacher['model_name']}; "
            f"student: {self.backbone_student.__class__.__name__}",
        )

    def forward(self, inputs, data_samples, mode):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def extract_feat(self, batch_inputs, batch_data_samples):
        if self.use_cache:
            # feat_list = []
            # for data_samples in batch_data_samples:
            #     feat_list.append(data_samples.gt_feats)
            # feat_teacher = torch.stack(feat_list)
            feat_teacher = batch_data_samples
        else:
            feat_teacher = self.neck_teacher(self.backbone_teacher(batch_inputs))
        feat_student = self.neck_student(self.backbone_student(batch_inputs))
        if self.add_adapter:
            feat_list = []
            for idx in range(4):
                feat_list.append(self.patch_embeds[idx](feat_student[idx]))
            feat_student = torch.stack(feat_list, dim=0).mean(dim=0)
        return feat_teacher, feat_student

    def loss(self, batch_inputs, batch_data_samples):
        feat_teacher, feat_student = self.extract_feat(batch_inputs, batch_data_samples)
        return {
            "loss_distillation": self.loss_distill(feat_teacher, feat_student)
        }

    def optimize_parameters(self, inputs, data_samples, mode):
        self.loss_distillation = self.forward(inputs, data_samples, mode)['loss_distillation']
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.loss_distillation.backward()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

