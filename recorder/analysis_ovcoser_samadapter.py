"""

"""
import os
from recorder.new_evaluator import Classification
import numpy as np
import torch
from py_sod_metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from py_sod_metrics.utils import TYPE, get_adaptive_threshold, prepare_data
import cv2

class MAE(MAE):
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = prepare_data(pred, gt)
        mae = self.cal_mae(pred, gt)

        mae = np.asarray(mae).reshape(1, 1)
        # if pre_cls != gt_cls:
        #     mae.fill(1)

        self.maes.append(mae)

    def get_results(self):
        mae = np.concatenate(self.maes, axis=0, dtype=TYPE)  # N,1
        return dict(mae=mae)

test_class_list = ['owlfly larva', 'grouse', 'frogmouth', 'bat', 'bee', 'bittern', 'mockingbird', 'dragonfly', 'heron', 'egyptian nightjar', 'potoo', 'cicada', 'butterfly', 'moth', 'slug', 'reccoon', 'monkey', 'kangaroo', 'mongoose', 'lion', 'elephant', 'jerboa', 'snail', 'duck', 'cheetah', 'giraffe', 'ant', 'beetle', 'wolf', 'rabbit', 'tiger', 'squirrel', 'polar bear', 'deer', 'dog', 'scorpion', 'arctic fox', 'goat', 'hedgehog', 'chameleon', 'leopard', 'worm', 'stick insect', 'cat', 'crocodilefish', 'batfish', 'clownfish', 'frogfish', 'seadragon', 'stingaree', 'crocodile', 'starfish', 'hermit crab', 'cuttlefish', 'shrimp', 'seal', 'crab', 'octopus', 'turtle', 'scorpionfish', 'non-succulent plant']
SAMadapter_test_image_dir = "/media/estar/Data/ywb/SAM-Adapter-PyTorch-main/save/_cod-sam-vit-h-maskdecoder1-alphaclip/2024-12-04 19:55:46/ovcamo_test_alphaclip_pred"
OVCoser_test_image_dir = "/media/estar/Data/ywb/OVCamo-main/ovcoser-ovcamo-te"
gt_dir = "/media/estar/Data/ywb/OVCamoDataset/test/mask"
#分类问题
# test_lal2cname = dict()
# for i in range(len(test_class_list)):
#     test_lal2cname[i] = test_class_list[i]
# evaluator = Classification(lab2cname=test_lal2cname, per_class_result=True)
# evaluator.reset()
# for file in os.listdir(OVCoser_test_image_dir):
#     pred_name = file.split(']')[0][1:]
#     pred_index = test_class_list.index(pred_name)
#     score = torch.zeros(len(test_class_list))
#     score[pred_index] = 1
#     score = score.unsqueeze(0)
#     ###
#     label = file.split('_')[1].split('.')[0]
#     label_index = test_class_list.index(label)
#     label_id = torch.tensor([label_index])
#     evaluator.process(score, label_id)
# evaluator.evaluate()

#分割mae
test_mae = MAE()
for image_index in os.listdir(OVCoser_test_image_dir):
    gt_image_name = image_index.split(']')[1]
    gt_image_file = os.path.join(gt_dir, gt_image_name)
    gt = cv2.imread(gt_image_file)[:,:,0]
    image_file = os.path.join(OVCoser_test_image_dir, image_index)
    image = cv2.imread(image_file)[:,:,0]
    test_mae.step(image, gt)
mae = test_mae.get_results()
tmp=1

