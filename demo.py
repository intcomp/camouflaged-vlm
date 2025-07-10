import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from PIL import Image
import yaml
import torch
import models
import numpy as np
import cv2
import numpy as np
from cocotrainers.mapleAlphaCLIP import TestMaPLeAlphaCLIP
import torch.nn.functional as F
from datasets.ovcamo_info.class_names import TRAIN_CLASS_NAMES, TEST_CLASS_NAMES

from torchvision import transforms
from alpha_clip_rw.alpha_clip import mask_transform as alpha_mask_transform
from alpha_clip_rw.alpha_clip import _transform as alpha_img_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)  # 递归嵌套字典
            setattr(self, key, value)

    def __getitem__(self, item):  # 允许按键访问
        return getattr(self, item)


def resize(image_array: np.ndarray, height, width, interpolation=cv2.INTER_LINEAR):
    h, w = image_array.shape[:2]
    if h == height and w == width:
        return image_array

    resized_image_array = cv2.resize(image_array, dsize=(width, height), interpolation=interpolation)
    return resized_image_array

def save_array_as_image(image: np.ndarray, data_array: np.ndarray, class_name: str,  save_name: str, save_dir: str, to_minmax: bool = False):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    if data_array.dtype != np.uint8:
        if data_array.max() > 1:
            raise Exception("the range of data_array has smoe errors")
        data_array = (data_array * 255).astype(np.uint8)

        # 创建一个颜色遮罩
    overlay = image.copy()
    overlay[data_array > 0] = (0, 255, 0)

    # 使用加权方式融合图像与遮罩
    alpha = 0.5  # 透明度
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # 在图像左上角写入类名
    h, w = image.shape[:2]
    text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size
    center_x = (w - text_w) // 2
    center_y = (h + text_h) // 2
    cv2.putText(blended, class_name, (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(save_path, blended)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', default="./demo_img/scorpionfish.jpg")
    parser.add_argument('--output-dir', default="./demo_img")
    parser.add_argument('--config', default="./configs/demo.yaml")
    parser.add_argument('--model', default="./best_model_pth/model_epoch_best.pth")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    test_clip_cfg = DotDict(config['MAPLE_ALPHA_CLIP'])
    tr_dataset_class_names = TRAIN_CLASS_NAMES
    te_dataset_class_names = TEST_CLASS_NAMES
    maple_clip_model = TestMaPLeAlphaCLIP(test_clip_cfg, tr_dataset_class_names, te_dataset_class_names).model

    model = models.make(config['model']).cuda()
    model.load_mapleAlphaCLIP(maple_clip_model)
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    print(f"model load checkpoints:{args.model}")

    ## process image
    img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    image_rgb = Image.open(args.img_path).convert('RGB')
    image_w, image_h = image_rgb.size
    image_name = args.img_path.split('/')[-1]
    input_alpha_mask = Image.fromarray(np.ones(image_rgb.size) * 255)
    clip_img_torch = alpha_img_transform(n_px=336)(image_rgb)
    clip_mask_torch = alpha_mask_transform(n_px=336)(input_alpha_mask)

    input_image = img_transform(image_rgb)
    
    #### start test image ####
    with torch.no_grad():

        inp = input_image.unsqueeze(0).cuda()
        clip_image = clip_img_torch.unsqueeze(0).cuda()
        clip_mask = clip_mask_torch.unsqueeze(0).cuda()
        #### inference result ####
        pred_mask = model.infer_test(inp, clip_image, clip_mask)
        pred_mask = torch.sigmoid(pred_mask)

        #### classification results ####
        alpha = F.interpolate(pred_mask, (336, 336), mode="bilinear", align_corners=False)
        image = clip_image
        _, _, pred_1, score = model.clip_model(image, alpha, train=False)

        #### OVCOS metric ####
        pred_mask_array = pred_mask.squeeze(0).squeeze(0).cpu().detach().numpy()  # B,H,W
        
        pred = resize(pred_mask_array, height=image_h, width=image_w)
        pre_cls = te_dataset_class_names[pred_1]

        if args.output_dir is not None:
            save_array_as_image(np.asarray(image_rgb), pred, class_name=pre_cls, save_name=f"[{pre_cls}]{image_name}", save_dir=args.output_dir)





    