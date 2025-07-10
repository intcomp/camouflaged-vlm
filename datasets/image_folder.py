import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register
from tqdm import tqdm
from utils import log

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, 
                 dataset_info, 
                 split_key=None, 
                 size=None,
                 repeat=1, 
                 cache='none', 
                 mask=False):
        self.repeat = repeat
        self.cache = cache
        self.Train = False
        self.split_key = split_key
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

        total_data_paths = []
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
                total_data_paths.append((class_name, image_path, mask_path))
            else:
                image_path = os.path.join(dataset_info['OVCamo_TE_IMAGE_DIR'], unique_id + image_suffix)
                mask_path = os.path.join(dataset_info['OVCamo_TE_MASK_DIR'], unique_id + mask_suffix)
                total_data_paths.append((class_name, image_path, mask_path))
                
        # log(f"[{self.split_key}Set] {len(total_data_paths)} Samples, {len(self.classes)} Classes")

        self.files = []

        for filename in total_data_paths:
            file=dict()
            file["class_label"] = filename[0]
            file["class_id"] = self.classes.index(filename[0])
            file["img_pth"] = filename[1]
            file["mask_pth"] = filename[2]
            
            if self.cache == 'none':
                file["image"] = filename[1]
                file["mask"] = filename[2]
            elif self.cache == 'in_memory':
                file["image"] = Image.open(filename[1]).convert('RGB')
                file["mask"] = Image.open(filename[2]).convert('L')
            self.files.append(file)

        # self.files = self.files[:10]
    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        file = self.files[idx % len(self.files)]
        '''
        file[0]: class label
        file[1]: RGB image path
        file[2]: gray mask path
        '''
        if self.cache == 'none':
            file["image"] = Image.open(file["img_pth"]).convert('RGB')
            file["mask"] = Image.open(file["mask_pth"]).convert('L')

        return file
    