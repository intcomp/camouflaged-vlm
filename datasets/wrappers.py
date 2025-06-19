
import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import os
from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode
from datasets.transform_custom import *
import torch.nn.functional as F
import alpha_clip
from alpha_clip.alpha_clip import mask_transform as alpha_mask_transform
import datasets.boundary_modification

def load_clip_preprocess():
    _, clip_preprocess = alpha_clip.load("ViT-L/14@336px",
                                               alpha_vision_ckpt_pth="/media/estar/Data/ywb/AlphaCLIP-main/checkpoints/clip_l14_336_grit_20m_4xe.pth",
                                               device='cpu'
                                               )
    return clip_preprocess
def crop_center(img, croph, cropw):
    h, w = img.shape[:2]
    starth = h//2 - (croph//2)
    startw = w//2 - (cropw//2)
    return img[starth:starth+croph, startw:startw+cropw, :]

def random_modified(gt, iou_max=1.0, iou_min=0.8):
    iou_target = np.random.rand() * (iou_max - iou_min) + iou_min
    seg = datasets.boundary_modification.modify_boundary((np.array(gt) > 0.5).astype('uint8') * 255, iou_target=iou_target)
    return seg

@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.clip_img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.clip_mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((336, 336)),
            transforms.Normalize(0.5, 0.26)
        ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

        self.pred_mask_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

        self.alphaclip_preprocess = load_clip_preprocess()
        self.alphaclip_mask_preprocess = alpha_mask_transform()
        self.choice = "center_crop"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, label_id, file = self.dataset[idx]
        #加载第一阶段的图像
        pred_first_mask_dir = "/media/estar/Data/ywb/SAM-Adapter-PyTorch-main/save/_cod-sam-vit-h/2025-03-17/cod_val_save_test_img"
        pred_first_mask_path = os.path.join(pred_first_mask_dir, file[2].split('/')[-1])
        pred_first_mask = Image.open(pred_first_mask_path).convert('L')

        image_torch = self.alphaclip_preprocess(img)
        mask_torch = self.alphaclip_mask_preprocess(Image.fromarray(np.ones_like(mask) * 255))
        # mask_torch = self.alphaclip_mask_preprocess(pred_first_mask)
        gt_torch = self.alphaclip_mask_preprocess(mask)

        if img.size != mask.size:
            img = np.asarray(img)
            img = np.rot90(img)
            img = Image.fromarray(img)

        return {
            'inp': self.img_transform(img),
            'clip_image': image_torch,
            'clip_zero_mask': mask_torch,
            "gt_torch": gt_torch,
            # 'pred_clip_first_mask': pred_clip_first_mask,
            'pred_first_mask': self.pred_mask_transform(pred_first_mask),
            'gt': self.mask_transform(mask),
            'label_id': torch.tensor(label_id),
            'label_name': file[0],
            'image_path': file[1],
            'mask_path': file[2],
            'caption_feat_index': file[-1],
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size

        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.clip_img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.clip_mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((336, 336)),
            transforms.Normalize(0.5, 0.26)
        ])

        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

        self.pred_mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

        self.alphaclip_preprocess = load_clip_preprocess()
        self.alphaclip_mask_preprocess = alpha_mask_transform()
        self.choice = "center_crop"

    def __len__(self):
        return len(self.dataset)

    def ycbcr_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('YCbCr')

    def __getitem__(self, idx):
        img, mask, label_id, file = self.dataset[idx]
        #加载第一阶段的图像
        pred_first_mask_dir = "/media/estar/Data/ywb/SAM-Adapter-PyTorch-main/save/_cod-sam-vit-h/2025-03-17/cod_val_save_train_img"
        pred_first_mask_path = os.path.join(pred_first_mask_dir, file[2].split('/')[-1])
        pred_first_mask = Image.open(pred_first_mask_path).convert('L')
        # pred_first_mask = random_modified(mask)
        # pred_first_mask = Image.fromarray(pred_first_mask)

        image_torch = self.alphaclip_preprocess(img)
        if torch.rand(1).item() < 0.5:
            mask_torch = self.alphaclip_mask_preprocess(Image.fromarray(np.ones_like(mask) * 255))
        else:
            mask_torch = self.alphaclip_mask_preprocess(pred_first_mask)

        if img.size != mask.size:
            # 旋转img
            img = np.asarray(img)
            img = np.rot90(img)
            img = Image.fromarray(img)
        # random filp
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            pred_first_mask = pred_first_mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
        # pred_clip_first_mask = self.alphaclip_mask_preprocess(pred_first_mask)
        pred_first_mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(pred_first_mask)


        return {
            'inp': self.img_transform(img),
            'clip_image': image_torch,
            'clip_zero_mask': mask_torch,
            # 'pred_clip_first_mask': pred_clip_first_mask,
            'pred_first_mask': self.pred_mask_transform(pred_first_mask),
            'gt': self.mask_transform(mask),
            'label_id': torch.tensor(label_id),
            'label_name': file[0],
            'mask_path': file[2],
            'caption_feat_index': file[-1],
        }
