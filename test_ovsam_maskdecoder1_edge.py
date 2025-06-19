import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint
import numpy as np
import cv2
from PIL import Image
import recorder
from pathlib import Path
import numpy as np
from cocotrainers.mapleAlphaCLIP import TestMaPLeAlphaCLIP
from dassl.utils import load_checkpoint
import torch.nn.functional as F
import alpha_clip
from new_evaluator import Classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

res_mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])
def crop_center(img, croph, cropw):
    h, w = img.shape[:2]
    starth = h//2 - (croph//2)
    startw = w//2 - (cropw//2)
    return img[starth:starth+croph, startw:startw+cropw, :]

def second_process_img_mask(image, mask):
    image = np.asarray(image)
    if mask.shape != image.shape[:2]:
        image = np.rot90(image)
    rgba = np.concatenate((image, np.expand_dims(mask, axis=-1)), axis=-1)
    h, w = rgba.shape[:2]

    if min(h, w) == h:
        rgba = crop_center(rgba, h, h)
    else:
        rgba = crop_center(rgba, w, w)

    rgb = rgba[:, :, :-1]
    mask = rgba[:, :, -1]
    image_torch = res_clip_standard_transform(rgb)
    mask_torch = res_mask_transform(mask)

    image_torch = image_torch.unsqueeze(0).to(device)
    mask_torch = mask_torch.unsqueeze(0).to(device)
    return image_torch, mask_torch

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)  # 递归嵌套字典
            setattr(self, key, value)

    def __getitem__(self, item):  # 允许按键访问
        return getattr(self, item)

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds

def resize(image_array: np.ndarray, height, width, interpolation=cv2.INTER_LINEAR):
    h, w = image_array.shape[:2]
    if h == height and w == width:
        return image_array

    resized_image_array = cv2.resize(image_array, dsize=(width, height), interpolation=interpolation)
    return resized_image_array


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_array_as_image(data_array: np.ndarray, save_name: str, save_dir: str, to_minmax: bool = False):
    """
    save the ndarray as a image

    Args:
        data_array: np.float32 the max value is less than or equal to 1
        save_name: with special suffix
        save_dir: the dirname of the image path
        to_minmax: minmax the array
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    if data_array.dtype != np.uint8:
        if data_array.max() > 1:
            raise Exception("the range of data_array has smoe errors")
        data_array = (data_array * 255).astype(np.uint8)
    # if to_minmax:
    #     data_array = minmax(data_array, up_bound=255)
    #     data_array = (data_array * 255).astype(np.uint8)
    cv2.imwrite(save_path, data_array)

def eval_psnr_ovcamo(loader, model, save_img_path=None, mode=None):
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    clip, preprocess = alpha_clip.load("ViT-L/14@336px",
                                       alpha_vision_ckpt_pth="/media/estar/Data/ywb/AlphaCLIP-main/checkpoints/clip_l14_336_grit_20m_4xe.pth",
                                       device='cpu'
                                       )
    clip = clip.float().cuda()
    cnt = 0
    te_dataset_class_names = loader.dataset.dataset.dataset.classes
    test_lal2cname = dict()
    for i in range(len(te_dataset_class_names)):
        test_lal2cname[i] = te_dataset_class_names[i]
    evaluator = Classification(lab2cname=test_lal2cname, per_class_result=True)
    evaluator.reset()
    metric_names = ("sm", "wfm", "mae", "fm", "em", "iou")
    metricer = recorder.OVCOSMetricer(class_names=te_dataset_class_names, metric_names=metric_names)
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()

            inp = batch['inp']
            gt = batch['gt']
            label_id = batch['label_id']
            label_name = batch['label_name']

            clip_image = batch['clip_image']
            clip_zero_mask = batch['clip_zero_mask']
            pred_mask = model.infer_test(inp, clip_image, clip_zero_mask)
            pred_mask = torch.sigmoid(pred_mask)
            cnt += pred_mask.shape[0]
            if pbar is not None:
                pbar.update(1)
            # 方式一：
            # alpha = F.interpolate(pred_mask, (336, 336), mode="bilinear", align_corners=False)
            # text_embeddings = model.test_text_features.permute(1, 0)
            # image_features = clip.visual(clip_image, alpha)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # score = torch.matmul(image_features, text_embeddings)
            # pred_1 = score.topk(1, dim=1)[1].squeeze(dim=1)

            # 方式二：
            text_embeddings = model.test_text_features.permute(1, 0)
            # text_embeddings = model.train_text_features.permute(1, 0)
            alpha = F.interpolate(pred_mask, (336, 336), mode="bilinear", align_corners=False)
            image = Image.open(batch['image_path'][0]).convert('RGB')
            image = preprocess(image).unsqueeze(0).cuda()
            image_features = clip.visual(image, alpha)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score = torch.matmul(image_features, text_embeddings)
            evaluator.process(score, label_id)
            pred_1 = score.topk(1, dim=1)[1].squeeze(dim=1)

            #方式三：
            # alpha = pred_mask.squeeze(1).cpu().detach().numpy()
            # mask_h, mask_w = cv2.imread(Path(batch["mask_path"][0]).as_posix(), cv2.IMREAD_GRAYSCALE).shape
            # alpha = (resize(alpha[0], height=mask_h, width=mask_w) * 255).astype(np.uint8)
            # image = Image.open(batch['image_path'][0]).convert('RGB')
            # # image = preprocess(image).unsqueeze(0).cuda()
            # image, alpha = second_process_img_mask(image, alpha)
            # #利用image 和 alpha求 对应的image_feat text_feat 以及预测的pred_cls
            # text_embeddings = model.test_text_features.permute(1, 0)
            # image_features = clip.visual(image, alpha)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # score = torch.matmul(image_features, text_embeddings)
            # pred_1 = score.topk(1, dim=1)[1].squeeze(dim=1)

            probs = pred_mask.squeeze(1).cpu().detach().numpy()  # B,H,W
            mask_paths = batch["mask_path"]
            for idx_in_batch, pred in enumerate(probs):
                mask_path = Path(mask_paths[idx_in_batch])
                mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                mask_h, mask_w = mask.shape
                gt_cls = batch['label_name'][idx_in_batch]

                pred = resize(pred, height=mask_h, width=mask_w)
                # pre_cls = label_name[idx_in_batch]
                pre_cls = te_dataset_class_names[pred_1]

                if save_img_path is not None:
                    save_array_as_image(pred, save_name=f"[{pre_cls}]{mask_path.name}", save_dir=save_img_path)
                    # save_array_as_image(pred, save_name=f"{mask_path.name}", save_dir=save_img_path)
                metricer.step(
                    pre=(pred * 255).astype(np.uint8),
                    gt=mask,
                    pre_cls=pre_cls,
                    gt_cls=gt_cls,
                    gt_path=mask_path.as_posix(),
                )
        avg_ovcos_results = metricer.show()

    evaluator.evaluate()
    if pbar is not None:
        pbar.close()
    print(str(avg_ovcos_results))

    return avg_ovcos_results['sm'], avg_ovcos_results['wfm'], avg_ovcos_results['mae'], avg_ovcos_results['avgfm'], avg_ovcos_results['avgem'], avg_ovcos_results['avgiou']

def eval_psnr_ovcamo_both(loader, model, save_img_path=None, mode=None):
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    metric_fn = utils.calc_cod
    metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    cnt = 0
    te_dataset_class_names = loader.dataset.dataset.dataset.classes
    #### build evaluator ####
    test_lal2cname = dict()
    for i in range(len(te_dataset_class_names)):
        test_lal2cname[i] = te_dataset_class_names[i]
    evaluator = Classification(lab2cname=test_lal2cname, per_class_result=False)
    evaluator.reset()
    #### build metricer ####
    metric_names = ("sm", "wfm", "mae", "fm", "em", "iou")
    metricer = recorder.OVCOSMetricer(class_names=te_dataset_class_names, metric_names=metric_names)
    #### start testing ####
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()

            inp = batch['inp']
            gt = batch['gt']
            label_id = batch['label_id']
            label_name = batch['label_name']
            clip_image = batch['clip_image']
            clip_zero_mask = batch['clip_zero_mask']
            #### inference result ####
            pred_mask = model.infer_test(inp, clip_image, clip_zero_mask)
            pred_mask = torch.sigmoid(pred_mask)
            #### COS metric ####
            result1, result2, result3, result4 = metric_fn(pred_mask, batch['gt'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])
            #### classification results ####
            alpha = F.interpolate(pred_mask, (336, 336), mode="bilinear", align_corners=False)
            image = batch['clip_image']
            _, _, pred_1, score = model.clip_model(image, alpha, train=False)
            evaluator.process(score, label_id)
            #### OVCOS metric ####
            probs = pred_mask.squeeze(1).cpu().detach().numpy()  # B,H,W
            mask_paths = batch["mask_path"]
            for idx_in_batch, pred in enumerate(probs):
                mask_path = Path(mask_paths[idx_in_batch])
                mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                mask_h, mask_w = mask.shape
                gt_cls = batch['label_name'][idx_in_batch]

                pred = resize(pred, height=mask_h, width=mask_w)
                pre_cls = te_dataset_class_names[pred_1]

                if save_img_path is not None:
                    save_array_as_image(pred, save_name=f"[{pre_cls}]{mask_path.name}", save_dir=save_img_path)
                    # save_array_as_image(pred, save_name=f"{mask_path.name}", save_dir=save_img_path)
                metricer.step(
                    pre=(pred * 255).astype(np.uint8),
                    gt=mask,
                    pre_cls=pre_cls,
                    gt_cls=gt_cls,
                    gt_path=mask_path.as_posix(),
                )

            cnt += pred_mask.shape[0]
            if pbar is not None:
                pbar.update(1)
        avg_ovcos_results = metricer.show()

    evaluator.evaluate()
    if pbar is not None:
        pbar.close()
    print(str(avg_ovcos_results))

    return avg_ovcos_results['sm'], avg_ovcos_results['wfm'], avg_ovcos_results['mae'], avg_ovcos_results['avgfm'], avg_ovcos_results['avgem'], avg_ovcos_results['avgiou'], val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()

def eval_psnr_ovcamo_both_iteration(loader, model, save_img_path=None, mode=None):
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    metric_fn = utils.calc_cod
    metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    cnt = 0
    te_dataset_class_names = loader.dataset.dataset.dataset.classes
    #### build evaluator ####
    test_lal2cname = dict()
    for i in range(len(te_dataset_class_names)):
        test_lal2cname[i] = te_dataset_class_names[i]
    evaluator = Classification(lab2cname=test_lal2cname, per_class_result=False)
    evaluator.reset()
    #### build metricer ####
    metric_names = ("sm", "wfm", "mae", "fm", "em", "iou")
    metricer = recorder.OVCOSMetricer(class_names=te_dataset_class_names, metric_names=metric_names)
    #### start testing ####
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()

            inp = batch['inp']
            gt = batch['gt']
            label_id = batch['label_id']
            label_name = batch['label_name']
            clip_image = batch['clip_image']
            clip_zero_mask = batch['clip_zero_mask']
            for iter in range(2):
                if iter != 0:
                    clip_zero_mask = alpha
                #### inference result ####
                pred_mask = model.infer_test(inp, clip_image, clip_zero_mask)
                pred_mask = torch.sigmoid(pred_mask)
                #### classification results ####
                alpha = F.interpolate(pred_mask, (336, 336), mode="bilinear", align_corners=False)
                image = batch['clip_image']
                _, _, pred_1, score = model.clip_model(image, alpha, train=False)

            #### COS metric ####
            result1, result2, result3, result4 = metric_fn(pred_mask, batch['gt'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])
            evaluator.process(score, label_id)
            #### OVCOS metric ####
            probs = pred_mask.squeeze(1).cpu().detach().numpy()  # B,H,W
            mask_paths = batch["mask_path"]
            for idx_in_batch, pred in enumerate(probs):
                mask_path = Path(mask_paths[idx_in_batch])
                mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                mask_h, mask_w = mask.shape
                gt_cls = batch['label_name'][idx_in_batch]

                pred = resize(pred, height=mask_h, width=mask_w)
                pre_cls = te_dataset_class_names[pred_1]

                if save_img_path is not None:
                    save_array_as_image(pred, save_name=f"[{pre_cls}]{mask_path.name}", save_dir=save_img_path)
                    # save_array_as_image(pred, save_name=f"{mask_path.name}", save_dir=save_img_path)
                metricer.step(
                    pre=(pred * 255).astype(np.uint8),
                    gt=mask,
                    pre_cls=pre_cls,
                    gt_cls=gt_cls,
                    gt_path=mask_path.as_posix(),
                )

            cnt += pred_mask.shape[0]
            if pbar is not None:
                pbar.update(1)
        avg_ovcos_results = metricer.show()

    evaluator.evaluate()
    if pbar is not None:
        pbar.close()
    print(str(avg_ovcos_results))

    return avg_ovcos_results['sm'], avg_ovcos_results['wfm'], avg_ovcos_results['mae'], avg_ovcos_results['avgfm'], avg_ovcos_results['avgem'], avg_ovcos_results['avgiou'], val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, verbose=False, save_img_path=None):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    with torch.no_grad():
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            inp = batch['inp']

            pred = torch.sigmoid(model.infer(inp))
            # 保存图像的一些操作
            mask_path = batch['mask_path'][0]
            mask = Image.open(mask_path)
            mask_name = mask_path.split('/')[-1]
            mask_w, mask_h = mask.size
            pred_save = pred.squeeze(0).squeeze(0).cpu().detach().numpy()  # B,H,W
            pred_save = resize(pred_save, height=mask_h, width=mask_w)
            if save_img_path is not None:
                save_array_as_image(pred_save, save_name=mask_name, save_dir=save_img_path)

            result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_name', default="ovcamo_both")
    parser.add_argument('--config', default="./configs/cod-sam-vit-h-maskdecoder1-edge.yaml")
    parser.add_argument("--root-info", default="./dataset_yamls/splitted_ovcamo.yaml", type=str)
    parser.add_argument('--model', default="./best_model_pth/model_epoch_last.pth")
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.root_info, mode="r") as f:
        root_info = yaml.safe_load(f)
        config['train_dataset']['dataset']['args']['root_info'] = root_info
        config['val_dataset']['dataset']['args']['root_info'] = root_info
        config['test_dataset']['dataset']['args']['root_info'] = root_info
        config['train_test_dataset']['dataset']['args']['root_info'] = root_info

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    test_loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8)

    spec = config['train_test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    train_loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8)

    test_clip_cfg = DotDict(config['maple_alpha_clip'])
    tr_dataset_class_names = train_loader.dataset.dataset.dataset.classes
    te_dataset_class_names = test_loader.dataset.dataset.dataset.classes
    maple_clip_model = TestMaPLeAlphaCLIP(test_clip_cfg, tr_dataset_class_names, te_dataset_class_names).model

    model = models.make(config['model']).cuda()
    model.load_mapleAlphaCLIP(maple_clip_model, test_clip_cfg.MODEL.CHECKPPOINT_BEST)
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)

    metric_name = args.metric_name

    if metric_name == "ovcamo":
        print("metric_name:", metric_name)
        mode = "test_alphaclip_pred"
        save_img_path = os.path.dirname(args.model) + '/' + metric_name + '_' + mode
        result_sm, result_wfm, result_mae, result_avgfm, result_avgem, result_avgiou = eval_psnr_ovcamo(train_loader, model, save_img_path=save_img_path, mode=mode)

    elif metric_name == "ovcamo_both":
        print("metric_name:", metric_name)
        mode = "train_dataset_first_iter"
        save_img_path = os.path.dirname(args.model) + '/' + metric_name + '_' + mode
        result_sm, result_wfm, result_mae, result_avgfm, result_avgem, result_avgiou, ori_sm, ori_em, ori_wfm, ori_mae = eval_psnr_ovcamo_both(test_loader, model, save_img_path=save_img_path, mode=mode)
        print('ori_sm: {:.4f}'.format(ori_sm))
        print('ori_em: {:.4f}'.format(ori_em))
        print('ori_wfm: {:.4f}'.format(ori_wfm))
        print('ori_mae: {:.4f}'.format(ori_mae))
    else:
        print("metric_name:", metric_name)
        save_img_path = os.path.dirname(args.model) + '/' + metric_name + '_save_test_img_1'
        metric1, metric2, metric3, metric4 = eval_psnr(test_loader, model,
                                                       data_norm=config.get('data_norm'),
                                                       eval_type=config.get('eval_type'),
                                                       eval_bsize=config.get('eval_bsize'),
                                                       verbose=True,
                                                       save_img_path=save_img_path)
        print('metric1: {:.4f}'.format(metric1))
        print('metric2: {:.4f}'.format(metric2))
        print('metric3: {:.4f}'.format(metric3))
        print('metric4: {:.4f}'.format(metric4))
