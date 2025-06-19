import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register
import pandas as pd

# llava_caption = "/media/estar/Data/ywb/LLaVA-NeXT-main/output_object_1.xlsx"
# if llava_caption is not None:
#     df = pd.read_excel(llava_caption, engine='openpyxl')
#     # 将第一列（图像地址列）转换为包含索引的字典
#     index_image_dict = df.iloc[:, 0].to_dict()
#
# all_llava_text_features = torch.load(
#     "/media/estar/Data/ywb/OVCamoDataset/text-features/llava_no_label/AllCamoPromptsTextFeaturesLLaVA.pth")
#
# img_textfeat_dict = {}
# for index, img_path in index_image_dict.items():
#     img_textfeat_dict[img_path] = all_llava_text_features[index]

def process_total_paths(dataset_info, class_infos, sample_infos, split_key):
    classes = []
    total_data_paths = []
    for class_info in class_infos:
        if class_info["split"] == split_key:
            classes.append(class_info["name"])

    for sample_info in sample_infos:
        class_name = sample_info["base_class"]
        if class_name not in classes:
            continue

        unique_id = sample_info["unique_id"]
        image_suffix = os.path.splitext(sample_info["image"])[1]
        mask_suffix = os.path.splitext(sample_info["mask"])[1]
        if split_key == 'train':
            image_path = os.path.join(dataset_info['OVCamo_TR_IMAGE_DIR'], unique_id + image_suffix)
            mask_path = os.path.join(dataset_info['OVCamo_TR_MASK_DIR'], unique_id + mask_suffix)
            feat_path = mask_path.replace('mask', 'sam_feats').replace('.png', '_vit_h_cache.pth')
            # depth_path = os.path.join(dataset_info['OVCamo_TR_DEPTH_DIR'], unique_id + mask_suffix)
            total_data_paths.append((class_name, image_path, mask_path, feat_path))
        else:
            image_path = os.path.join(dataset_info['OVCamo_TE_IMAGE_DIR'], unique_id + image_suffix)
            mask_path = os.path.join(dataset_info['OVCamo_TE_MASK_DIR'], unique_id + mask_suffix)
            feat_path = mask_path.replace('mask', 'sam_feats').replace('.png', '_vit_h_cache.pth')
            total_data_paths.append((class_name, image_path, mask_path, feat_path))

    return classes, total_data_paths

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, dataset_info, split_key=None, size=None,
                 repeat=1, cache='none', mask=False):
        # 读取使用llava生成的caption信息
        llava_caption = "/media/estar/Data/ywb/LLaVA-NeXT-main/output_object_1.xlsx"
        if llava_caption is not None:
            df = pd.read_excel(llava_caption, engine='openpyxl')
            # 将第一列（图像地址列）转换为包含索引的字典
            index_image_list = df.iloc[:, 0].tolist()
        #
        # all_llava_text_features = torch.load(
        #     "/media/estar/Data/ywb/OVCamoDataset/text-features/llava_no_label/AllCamoPromptsTextFeaturesLLaVA.pth")
        #
        # img_textfeat_dict = {}
        # for index, img_path in index_image_dict.items():
        #     img_textfeat_dict[img_path] = all_llava_text_features[index]

        self.repeat = repeat
        self.cache = cache
        self.Train = False
        if split_key == 'train':
            self.split_key = split_key
        elif split_key == 'test':
            self.split_key = 'test'

        self.size = size
        self.mask = mask
        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        with open(dataset_info['OVCamo_CLASS_JSON_PATH'], mode="r", encoding="utf-8") as f:
            class_infos = json.load(f)
        with open(dataset_info['OVCamo_SAMPLE_JSON_PATH'], mode="r", encoding="utf-8") as f:
            sample_infos = json.load(f)

        self.total_data_paths = []
        self.classes = []

        for class_info in class_infos:
            if class_info["split"] == self.split_key:
                self.classes.append(class_info["name"])
        for sample_info in sample_infos:
            class_name = sample_info["base_class"]
            if class_name not in self.classes:
                continue

            unique_id = sample_info["unique_id"]
            image_suffix = os.path.splitext(sample_info["image"])[1]
            mask_suffix = os.path.splitext(sample_info["mask"])[1]
            if self.split_key == 'train':
                image_path = os.path.join(dataset_info['OVCamo_TR_IMAGE_DIR'], unique_id + image_suffix)
                mask_path = os.path.join(dataset_info['OVCamo_TR_MASK_DIR'], unique_id + mask_suffix)
                caption_feat_index = index_image_list.index(image_path)
                # image_caption_feat = img_textfeat_dict[image_path]
                # image_caption_feat = None
                self.total_data_paths.append((class_name, image_path, mask_path, caption_feat_index))
            else:
                image_path = os.path.join(dataset_info['OVCamo_TE_IMAGE_DIR'], unique_id + image_suffix)
                mask_path = os.path.join(dataset_info['OVCamo_TE_MASK_DIR'], unique_id + mask_suffix)
                caption_feat_index = index_image_list.index(image_path)
                # image_caption_feat = img_textfeat_dict[image_path]
                # image_caption_feat = None
                self.total_data_paths.append((class_name, image_path, mask_path, caption_feat_index))
        # self.total_data_paths = self.total_data_paths[:10]
        print(f"[{self.split_key}Set] {len(self.total_data_paths)} Samples, {len(self.classes)} Classes")

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.total_data_paths) * self.repeat

    def __getitem__(self, idx):
        file = self.total_data_paths[idx % len(self.total_data_paths)]
        return Image.open(file[1]).convert('RGB'), Image.open(file[2]).convert('L'), self.classes.index(file[0]), file


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_info, **kwargs):
        self.dataset = ImageFolder(root_info, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

