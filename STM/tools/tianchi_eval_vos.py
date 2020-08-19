import os, glob, csv
import numpy as np
from PIL import Image

# TODO: modify to calculate vos miou. Match instances -> mean iou.

def IOU(gt, pred, eps=1e-5):
    intersect = np.sum(np.logical_and(gt > 0, pred > 0)) + eps
    union = np.sum(np.logical_or(gt > 0, pred > 0)) + eps
    iou = float(intersect / union)
    return iou


def AvgIOU(gt_mask, pred_mask, thresh=0.5):
    gts = np.unique(gt_mask)[1:]
    n_gt = len(gts)
    preds = np.unique(pred_mask)[1:]
    n_pred = len(preds)
    n_tp = 0
    iou_total = 0.
    for i in gts:
        gt = (gt_mask == i)
        ioui = 0.
        for j in preds:
            pred = (pred_mask == j)
            iouij = IOU(gt, pred)
            ioui = max(ioui, iouij)
        if ioui > thresh:
            n_tp += 1
        iou_total += ioui
    n_fp = n_pred - n_tp
    n_fn = n_gt - n_tp
    iou_avg = iou_total / n_gt
    recall = n_tp / (n_tp + n_fn)
    precision = n_tp / (n_tp + n_fp)
    return iou_avg, recall, precision


root = '/versa/dyy/dataset/TIANCHI'
anns = sorted(glob.glob(os.path.join(root, 'val_total/Annotations/*/*.png')))
pred_dir = os.path.join(root, 'v3_out')

metric = []
for i, ann in enumerate(anns):
    name = ann.split('/')[-1]
    video_id = ann.split('/')[-2]
    whole_name = video_id + '_' + name
    print(i, whole_name)
    gt_mask = np.array(Image.open(ann).convert('P'))
    try:
        pred_mask = np.array(Image.open(os.path.join(pred_dir, video_id, name)).convert('P'))
        iou_avg, recall, precision = AvgIOU(gt_mask, pred_mask, thresh=0.5)
        metric.append([whole_name, iou_avg, recall, precision])
    except:
        print('============ has no preds!!')
        metric.append([whole_name, 0., 0., 0.])

with open(os.path.join(root, 'v3_out.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['FileName', 'MIOU', 'Recall', 'Precision'])
    writer.writerows(metric)

