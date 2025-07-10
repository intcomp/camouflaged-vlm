import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import recorder
from pathlib import Path
import numpy as np
import cv2
from cocotrainers.mapleAlphaCLIP import TestMaPLeAlphaCLIP
from dassl.utils import load_checkpoint

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)  # 递归嵌套字典
            setattr(self, key, value)

    def __getitem__(self, item):  # 允许按键访问
        return getattr(self, item)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize(image_array: np.ndarray, height, width, interpolation=cv2.INTER_LINEAR):
    h, w = image_array.shape[:2]
    if h == height and w == width:
        return image_array

    resized_image_array = cv2.resize(image_array, dsize=(width, height), interpolation=interpolation)
    return resized_image_array


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if torch.is_tensor(v):
            log('  {}: shape={}'.format(k, tuple(v.shape)))
        else:
            log('  {}: {}'.format(k, v))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def eval_psnr_ovcamo(loader, model):
    model.eval()
    pbar = tqdm(total=len(loader), leave=False, desc='val')

    cnt = 0
    te_dataset_class_names = loader.dataset.dataset.classes
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
            clip_mask = batch['clip_mask']
            pred_mask = model.infer(inp, clip_image, clip_mask)
            pred_mask = torch.sigmoid(pred_mask)

            cnt += pred_mask.shape[0]
            if pbar is not None:
                pbar.update(1)

            probs = pred_mask.squeeze(1).cpu().detach().numpy()  # B,H,W
            mask_paths = batch["mask_path"]
            for idx_in_batch, pred in enumerate(probs):
                mask_path = Path(mask_paths[idx_in_batch])
                mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                mask_h, mask_w = mask.shape
                gt_cls = batch['label_name'][idx_in_batch]

                pred = resize(pred, height=mask_h, width=mask_w)
                # pre_cls = te_dataset_class_names[pred_cls]
                pre_cls = gt_cls
                # if save_path:
                #     ops.save_array_as_image(pred, save_name=f"[{pre_cls}]{mask_path.name}", save_dir=save_path)
                metricer.step(
                    pre=(pred * 255).astype(np.uint8),
                    gt=mask,
                    pre_cls=pre_cls,
                    gt_cls=gt_cls,
                    gt_path=mask_path.as_posix(),
                )
                numerical_results_step = metricer.get_step_results()
        avg_ovcos_results = metricer.show()

    if pbar is not None:
        pbar.close()

    return avg_ovcos_results['sm'], \
           avg_ovcos_results['wfm'], \
           avg_ovcos_results['mae'], \
           avg_ovcos_results['avgfm'], \
           avg_ovcos_results['avgem'], \
           avg_ovcos_results['avgiou']


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()
    pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        label_id = batch['label_id']
        clip_image = batch['clip_image']
        clip_mask = batch['clip_mask']
        model.set_input(inp, gt, label_id, clip_image, clip_mask)
        model.optimize_parameters()
        batch_loss = model.loss_G.item()
        loss_list.append(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = mean(loss_list)
    return loss

def main(config_, save_path):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    log("------------------------ build data loaders ------------------------")
    train_loader, val_loader = make_data_loaders()
    
    
    log("------------------------ build model ------------------------")
    # 构建 mapleAlphaCLIP
    maple_alpha_clip_cfg = DotDict(config['MAPLE_ALPHA_CLIP'])
    tr_dataset_class_names = train_loader.dataset.dataset.classes
    te_dataset_class_names = val_loader.dataset.dataset.classes
    maple_clip_model = TestMaPLeAlphaCLIP(maple_alpha_clip_cfg, tr_dataset_class_names, te_dataset_class_names).model

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    log(f"optimizer:{config['optimizer']['name']}, lr:{config['optimizer']['args']['lr']}") 
    log(f"lr_scheduler:CosineAnnealingLR, lr_min:{config['lr_min']}, epoch_max:{config['epoch_max']}")
    
    log("------------------------ load checkpoints ------------------------")
    model.load_mapleAlphaCLIP(maple_clip_model, maple_alpha_clip_cfg.MODEL.CHECKPPOINT_BEST)
    log(f"load maple alpha clip checkpoints:{maple_alpha_clip_cfg.MODEL.CHECKPPOINT_BEST}")
    
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    log(f"load sam checkpoints:{config['sam_checkpoint']}")

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    log(f"trainable parameter nums: {len(trainable_params)}")
    log("trainable parameter name:")
    log(trainable_params)
    # for name in trainable_params:
    #     log(name)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log('model_grad_params: {:.1f}M'.format(model_grad_params / 1e6))
    log('nmodel_total_params:{:.1f}M'.format(model_total_params / 1e6))

    log("------------------------ start training ------------------------")
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_mae_v = 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        # lr_scheduler.step()
        #
        train_loss_G = 1.00
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result_sm, result_wfm, result_mae, result_fm, result_em, result_iou = eval_psnr_ovcamo(val_loader, model)

            log_info.append('val: result_sm={:.4f}'.format(result_sm))
            writer.add_scalars("result_sm", {'val': result_sm}, epoch)

            log_info.append('val: result_wfm={:.4f}'.format(result_wfm))
            writer.add_scalars("result_wfm", {'val': result_wfm}, epoch)

            log_info.append('val: result_mae={:.4f}'.format(result_mae))
            writer.add_scalars("result_mae", {'val': result_mae}, epoch)

            log_info.append('val: result_fm={:.4f}'.format(result_fm))
            writer.add_scalars("result_fm", {'val': result_fm}, epoch)

            log_info.append('val: result_em={:.4f}'.format(result_em))
            writer.add_scalars("result_em", {'val': result_em}, epoch)

            log_info.append('val: result_iou={:.4f}'.format(result_iou))
            writer.add_scalars("result_iou", {'val': result_iou}, epoch)

            if result_mae < max_mae_v:
                max_mae_v = result_mae
                save(config, model, save_path, 'best')

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()

def save(config, model, save_path, name):
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/ovcos-sam-vit-h-maskdecoder-edge.yaml")
    parser.add_argument("--dataset-info", default="./datasets/ovcamo_info/splitted_ovcamo.yaml", type=str)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default="debug")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    with open(args.dataset_info, mode="r") as f:
        dataset_info = yaml.safe_load(f)
        config['train_dataset']['dataset']['args']['dataset_info'] = dataset_info
        config['val_dataset']['dataset']['args']['dataset_info'] = dataset_info
        config['test_dataset']['dataset']['args']['dataset_info'] = dataset_info

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    
    from datetime import datetime
    # 获取当前时间
    now = datetime.now()
    # 将时间格式化为字符串
    time_str = now.strftime("%Y%m%d%H")
    save_path = os.path.join(save_path, time_str)
    main(config, save_path)
