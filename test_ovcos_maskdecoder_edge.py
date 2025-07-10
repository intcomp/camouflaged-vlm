import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import models
import utils
import numpy as np
import cv2
import recorder
from pathlib import Path
import numpy as np
from cocotrainers.mapleAlphaCLIP import TestMaPLeAlphaCLIP
import torch.nn.functional as F
from recorder.new_evaluator import Classification
from datasets.ovcamo_info.class_names import TRAIN_CLASS_NAMES, TEST_CLASS_NAMES

from utils import set_log_path, log
from datetime import datetime

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


def eval_psnr_ovcamo_both(loader, model, save_img_path=None):
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    metric_fn = utils.calc_cod
    metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    cnt = 0
    te_dataset_class_names = loader.dataset.dataset.classes
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
            clip_mask = batch['clip_mask']
            #### inference result ####
            pred_mask = model.infer_test(inp, clip_image, clip_mask)
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
    log(str(avg_ovcos_results))

    return avg_ovcos_results['sm'], avg_ovcos_results['wfm'], avg_ovcos_results['mae'], avg_ovcos_results['avgfm'], avg_ovcos_results['avgem'], avg_ovcos_results['avgiou'], val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric-name', default="ovcamo_both")
    parser.add_argument('--output-dir', default="./eval_results")
    parser.add_argument('--config', default="./configs/ovcos-sam-vit-h-maskdecoder-edge.yaml")
    parser.add_argument("--dataset-info", default="./datasets/ovcamo_info/splitted_ovcamo.yaml", type=str)
    parser.add_argument('--model', default="./save/_ovcos-sam-vit-h-maskdecoder-edge/202507081743/model_epoch_last.pth")
    args = parser.parse_args()

    model_pth_list =  args.model.split('/') # relative path
    now = datetime.now()
    time_str = now.strftime("%Y%m%d%H%M")
    args.output_dir = args.output_dir + '/' + model_pth_list[2] +'/' + model_pth_list[3] + '/' + 'test_' + time_str
    os.makedirs(args.output_dir, exist_ok=True)
    set_log_path(args.output_dir)
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.dataset_info, mode="r") as f:
        dataset_info = yaml.safe_load(f)
        config['train_dataset']['dataset']['args']['dataset_info'] = dataset_info
        config['val_dataset']['dataset']['args']['dataset_info'] = dataset_info
        config['test_dataset']['dataset']['args']['dataset_info'] = dataset_info

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    test_loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8)

    test_clip_cfg = DotDict(config['MAPLE_ALPHA_CLIP'])
    tr_dataset_class_names = TRAIN_CLASS_NAMES
    te_dataset_class_names = TEST_CLASS_NAMES
    maple_clip_model = TestMaPLeAlphaCLIP(test_clip_cfg, tr_dataset_class_names, te_dataset_class_names).model

    model = models.make(config['model']).cuda()
    model.load_mapleAlphaCLIP(maple_clip_model)
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    log(f"model load checkpoints:{args.model}")

    log(f"metric_name: {args.metric_name}")
    save_img_path = args.output_dir + '/' + "result_image"
    result_sm, result_wfm, result_mae, result_avgfm, result_avgem, result_avgiou, ori_sm, ori_em, ori_wfm, ori_mae = eval_psnr_ovcamo_both(test_loader, model, save_img_path=save_img_path)
    log('ori_sm: {:.4f}'.format(ori_sm))
    log('ori_em: {:.4f}'.format(ori_em))
    log('ori_wfm: {:.4f}'.format(ori_wfm))
    log('ori_mae: {:.4f}'.format(ori_mae))
