import os, glob
import numpy as np
from PIL import Image
import pandas as pd


def IOU(gt, pred, eps=1e-5):
    intersect = np.sum(np.logical_and(gt > 0, pred > 0)) + eps
    union = np.sum(np.logical_or(gt > 0, pred > 0)) + eps
    iou = float(intersect / union)
    return iou


anns = sorted(glob.glob('/versa/dyy/dataset/ADE/panoptic/val_seg/*.png'))
pred_dir = 'v2_sem'

metric = []
eval_dict = {}
for i in range(22):
    eval_dict.setdefault(i, []).append(0)
    eval_dict.setdefault(i, []).append(0.)

for i, ann in enumerate(anns):
    name = ann.split('/')[-1]
    print(i, name)
    gt_mask = np.array(Image.open(ann))
    pred_mask = np.array(Image.open(os.path.join(pred_dir, name.replace('.png', '.png'))))
    gts = np.unique(gt_mask)
    for j in gts:
        if j == 255:
            continue
        gt = (gt_mask == j)
        pred = (pred_mask == j)
        if np.any(pred):
            iou = IOU(gt, pred)
            if iou > 0.5:
                eval_dict[j][0] += 1
                eval_dict[j][1] += iou

pd.DataFrame(eval_dict).to_csv('v2_sem.csv')

miou = 0.
metric = []
for k, v in eval_dict.items():
    if v[0] == 0:
        continue
    avg_iou = v[1] / v[0]
    print(k, v, avg_iou)
    miou += avg_iou
print(miou / 22)