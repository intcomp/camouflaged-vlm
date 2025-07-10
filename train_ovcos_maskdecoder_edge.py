import argparse
import os
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
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
from recorder.ovcos_metricer import calc_ovcamo
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import cv2
from cocotrainers.mapleAlphaCLIP import TestMaPLeAlphaCLIP

# 获取 local_rank
local_rank = int(os.environ["LOCAL_RANK"])
torch.distributed.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
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


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log(f"[{tag}Set] {len(dataset)} Samples, {len(dataset.dataset.classes)} Classes")
        for k, v in dataset[0].items():
            if torch.is_tensor(v):
                log('  {}: shape={}'.format(k, tuple(v.shape)))
            else:
                log('  {}: {}'.format(k, v))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    if tag == "train":
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def eval_psnr_ovcamo_new(loader, model):
    model.eval()
    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None
    val_sm = 0
    val_wfm = 0
    val_mae = 0
    val_avgfm = 0
    val_avgem = 0
    val_avgiou = 0
    cnt = 0
    te_dataset_class_names = loader.dataset.dataset.classes

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

            # 分割结果
            pred_mask = model.infer(inp, clip_image, clip_mask)
            pred_mask = torch.sigmoid(pred_mask)

            # 测试时的二阶段处理：将分割结果替代all_one_mask
            alpha = F.interpolate(pred_mask, (336, 336), mode="bilinear", align_corners=False)
            image = batch['clip_image']
            _, _, pred_1, score = model.clip_model(image, alpha, train=False)

            batch_pred = [torch.zeros_like(pred_mask) for _ in range(dist.get_world_size())]
            batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]
            cnt += pred_mask.shape[0]
            if pbar is not None:
                pbar.update(1)

            probs = pred_mask.squeeze(1).cpu().detach().numpy()  # B,H,W
            mask_paths = batch["mask_path"]
            for idx_in_batch, pred in enumerate(probs):
                mask_path = Path(mask_paths[idx_in_batch])
                mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                mask_h, mask_w = mask.shape

                pred = resize(pred, height=mask_h, width=mask_w)

                pre_cls = te_dataset_class_names[pred_1]
                gt_cls = batch['label_name'][idx_in_batch]
                result_dict = calc_ovcamo(
                    pre=(pred * 255).astype(np.uint8),
                        gt=mask,
                        pre_cls=pre_cls,
                        gt_cls=gt_cls,
                        gt_path=mask_path.as_posix(),
                )
                val_sm += result_dict['sm']
                val_wfm += result_dict['wfm']
                val_mae += result_dict['mae']
                val_avgfm += result_dict['avgfm']
                val_avgem += result_dict['avgem']
                val_avgiou += result_dict['avgiou']

    val_sm = torch.tensor(val_sm).cuda()
    val_wfm = torch.tensor(val_wfm).cuda()
    val_mae = torch.tensor(val_mae).cuda()
    val_avgfm = torch.tensor(val_avgfm).cuda()
    val_avgem = torch.tensor(val_avgem).cuda()
    val_avgiou = torch.tensor(val_avgiou).cuda()
    cnt = torch.tensor(cnt).cuda()
    dist.all_reduce(val_sm)
    dist.all_reduce(val_wfm)
    dist.all_reduce(val_mae)
    dist.all_reduce(val_avgfm)
    dist.all_reduce(val_avgem)
    dist.all_reduce(val_avgiou)
    dist.all_reduce(cnt)
    if pbar is not None:
        pbar.close()

    return val_sm.item() / cnt, val_wfm.item() / cnt, val_mae.item() / cnt, val_avgfm.item() / cnt, val_avgem.item() / cnt, val_avgiou.item() / cnt

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

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None
    loss_list = []
    loss_mask_lsit = []
    loss_edge_list = []

    cnt = 0
    for batch in train_loader:
        cnt += 1
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
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)

        batch_loss_mask = [torch.zeros_like(model.loss_dict["loss_mask"]) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss_mask, model.loss_dict["loss_mask"])
        loss_mask_lsit.extend(batch_loss_mask)

        batch_loss_edge = [torch.zeros_like(model.loss_dict["loss_edge"]) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss_edge, model.loss_dict["loss_edge"])
        loss_edge_list.extend(batch_loss_edge)

        # if local_rank == 0 and cnt == 10:
        #     loss_tmp = [i.item() for i in loss_list]
        #     loss_mask_tmp = [i.item() for i in loss_mask_lsit]
        #     loss_edge_tmp = [i.item() for i in loss_edge_list]
        #     loss_cls_tmp = [i.item() for i in loss_cls_list]
        #     print("loss:{}, loss_mask:{}, loss_edge::{}, loss_cls:{}".format(mean(loss_tmp), mean(loss_mask_tmp), mean(loss_edge_tmp), mean(loss_cls_tmp)))
        #     cnt = 0

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    loss_mask = [i.item() for i in loss_mask_lsit]
    loss_edge = [i.item() for i in loss_edge_list]
    return mean(loss), mean(loss_mask), mean(loss_edge)


def main(config_, save_path):
    global config, log, writer, log_info
    config = config_

    #### create log and writer ####
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    #### load dataset ####
    if local_rank == 0:
        log("------------------------ build data loaders ------------------------")
    
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
        
    if local_rank == 0:
        log("------------------------ build model ------------------------")
    #### build MaPLe Alpha CLIP ####
    maple_alpha_clip_cfg = DotDict(config['MAPLE_ALPHA_CLIP'])
    tr_dataset_class_names = train_loader.dataset.dataset.classes
    te_dataset_class_names = val_loader.dataset.dataset.classes
    maple_clip_model = TestMaPLeAlphaCLIP(maple_alpha_clip_cfg, tr_dataset_class_names, te_dataset_class_names).model

    #### build model optim, lr_scheduler ####
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    
    if local_rank == 0:
        log("------------------------ build optim, lr_scheduler ------------------------")
        log(f"optimizer:{config['optimizer']['name']}, lr:{config['optimizer']['args']['lr']}") 
        log(f"lr_scheduler:CosineAnnealingLR, lr_min:{config['lr_min']}, epoch_max:{config['epoch_max']}")

    if local_rank == 0:
        log("------------------------ load checkpoints ------------------------")
        
    #### loading maple_clip_model into model ####
    model.load_mapleAlphaCLIP(maple_clip_model, maple_alpha_clip_cfg.MODEL.CHECKPPOINT_BEST)
    if local_rank == 0:
        log(f"load maple alpha clip checkpoints:{maple_alpha_clip_cfg.MODEL.CHECKPPOINT_BEST}")
        
    #### build distributed training ####
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    #### loading checkpoint ####
    model = model.module
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    if local_rank == 0:
        log(f"load sam checkpoints:{config['sam_checkpoint']}")
    model.load_state_dict(sam_checkpoint, strict=False)
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
            
    #### print training param ####
    if local_rank == 0:
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        log(f"trainable parameter nums: {len(trainable_params)}")
        log("trainable parameter name:")
        log(trainable_params)
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log('model_total_params: {:.1f}M'.format(model_total_params / 1e6))
        log('model_grad_params: {:.1f}M'.format(model_grad_params / 1e6))

    if local_rank == 0:
        log("------------------------ start training ------------------------")
    #### start training ####
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_mae_v = 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G,  train_loss_mask, train_loss_edge = train(train_loader, model)
        # train_loss_G = 1.0
        lr_scheduler.step()

        if local_rank == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log('epoch {}/{}: train G: loss={:.4f}, loss_mask={:.4f}, loss_edge={:.4f}'.format(epoch, epoch_max, train_loss_G, train_loss_mask, train_loss_edge))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
            writer.add_scalars('loss_mask', {'train mask': train_loss_mask}, epoch)
            writer.add_scalars('loss_edge', {'train edge': train_loss_edge}, epoch)
            
            save(model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result_sm, result_wfm, result_mae, result_fm, result_em, result_iou = eval_psnr_ovcamo_new(val_loader, model)
            if local_rank == 0:
                log_info = ['val epoch {}'.format(epoch)]
                log_info.append('result_sm={:.4f}'.format(result_sm))
                writer.add_scalars("result_sm", {'val': result_sm}, epoch)

                log_info.append('result_wfm={:.4f}'.format(result_wfm))
                writer.add_scalars("result_wfm", {'val': result_wfm}, epoch)

                log_info.append('result_mae={:.4f}'.format(result_mae))
                writer.add_scalars("result_mae", {'val': result_mae}, epoch)

                log_info.append('result_fm={:.4f}'.format(result_fm))
                writer.add_scalars("result_fm", {'val': result_fm}, epoch)

                log_info.append('result_em={:.4f}'.format(result_em))
                writer.add_scalars("result_em", {'val': result_em}, epoch)

                log_info.append('result_iou={:.4f}'.format(result_iou))
                writer.add_scalars("result_iou", {'val': result_iou}, epoch)

                if result_mae < max_mae_v:
                    max_mae_v = result_mae
                    save(model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()

def save(model, save_path, name):
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/ovcos-sam-vit-h-maskdecoder-edge.yaml")
    parser.add_argument("--dataset-info", default="./datasets/ovcamo_info/splitted_ovcamo.yaml", type=str)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
    # time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    time_str = now.strftime("%Y%m%d%H%M")
    save_path = os.path.join(save_path, time_str)
    main(config, save_path)
