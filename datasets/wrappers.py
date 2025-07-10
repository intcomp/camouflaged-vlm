import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register

from torchvision.transforms import InterpolationMode
from datasets.transform_custom import *
from alpha_clip_rw.alpha_clip import mask_transform as alpha_mask_transform
from alpha_clip_rw.alpha_clip import _transform as alpha_img_transform

# from 
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
        
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

        self.alphaclip_img_preprocess = alpha_img_transform(n_px=336)
        self.alphaclip_mask_preprocess = alpha_mask_transform(n_px=336)
        self.choice = "center_crop"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        class_label
        class_id
        img_pth
        mask_pth
        image
        mask
        '''
        data_dict = self.dataset[idx]
        img = data_dict["image"]
        mask = data_dict["mask"]
        label_id = data_dict["class_id"]
        label_name = data_dict["class_label"]
        
        if img.size != mask.size:
            img = np.asarray(img)
            img = np.rot90(img)
            img = Image.fromarray(img)
            
        clip_img_torch = self.alphaclip_img_preprocess(img)
        clip_mask_torch = self.alphaclip_mask_preprocess(Image.fromarray(np.ones_like(mask) * 255))
        
        input = self.img_transform(img)
        gt = self.mask_transform(mask)

        return {
            'clip_image': clip_img_torch,
            'clip_mask': clip_mask_torch,
            'inp': input,
            "gt": gt,
            'label_id': torch.tensor(label_id),
            'label_name': label_name,
            'image_path': data_dict["img_pth"],
            'mask_path': data_dict["mask_pth"],
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, 
                 dataset, 
                 size_min=None, 
                 size_max=None, 
                 inp_size=None,
                 augment=False, 
                 gt_resize=None
                 ):
        
        self.dataset = dataset
        # not used at now
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

        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

        self.alphaclip_img_preprocess = alpha_img_transform(n_px=336)
        self.alphaclip_mask_preprocess = alpha_mask_transform(n_px=336)
        self.choice = "center_crop"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        class_label
        class_id
        img_pth
        mask_pth
        image
        mask
        '''
        data_dict = self.dataset[idx]
        img = data_dict["image"]
        mask = data_dict["mask"]
        label_id = data_dict["class_id"]
        label_name = data_dict["class_label"]
        
        if img.size != mask.size:
            # 旋转img
            img = np.asarray(img)
            img = np.rot90(img)
            img = Image.fromarray(img)
            
        clip_img_torch = self.alphaclip_img_preprocess(img)
        clip_mask_torch = self.alphaclip_mask_preprocess(Image.fromarray(np.ones_like(mask) * 255))

        # random filp
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)

        input = self.img_transform(img)
        gt = self.mask_transform(mask)
        return {
            'clip_image': clip_img_torch,
            'clip_mask': clip_mask_torch,
            'inp': input,
            'gt': gt,
            'label_id': torch.tensor(label_id),
            'label_name': label_name,
        }
