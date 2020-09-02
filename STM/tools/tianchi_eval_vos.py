import os, glob, csv
import numpy as np
from PIL import Image
import time


def calculate_videos_miou(pred_dir, ann_dir):
    print('Calculating video miou...')
    preds = os.listdir(pred_dir)
    anns = os.listdir(ann_dir)
    ious = []
    ins_ious = []
    for i, pred in enumerate(preds):
        if pred not in anns:
            print('{} have no groundtruth!'.format(pred))
        else:
            st = time.time()
            iou, result = get_video_miou(os.path.join(pred_dir, pred), os.path.join(ann_dir, pred))
            ed = time.time()
            print('{}, video: {}, iou: {:.2f}, cost time: {:.2f}s'.format(i, pred, iou, ed - st))
            print()
            ious.append(iou)
            ins_ious.extend(result)
    miou = np.mean(ious)
    miou_2 = np.mean(ins_ious)
    num_cal = len(ious)
    return miou, num_cal, miou_2


def get_video_miou(video_dir, gt_dir):
    # assert os.listdir(video_dir) == os.listdir(gt_dir)
    pred_labels = set([])
    gt_labels = set([])
    iou_d = {}
    for f in os.listdir(video_dir):
        pred = os.path.join(video_dir, f)
        ann = os.path.join(gt_dir, f)
        pred_img = np.array(Image.open(pred).convert('P')).astype(np.uint8)
        ann_img = np.array(Image.open(ann).convert('P')).astype(np.uint8)
        pl = np.unique(pred_img)
        gl = np.unique(ann_img)
        pred_labels = pred_labels | set(pl)
        gt_labels = gt_labels | set(gl)

        for g in gl:
            for p in pl:
                if g == 0 or p == 0:
                    continue
                iou_d.setdefault(g, {})
                iou_d[g].setdefault(p, [])
                iou = IOU(ann_img == g, pred_img == p)
                iou_d[g][p].append(iou)

    pred_labels = list(pred_labels)
    pred_labels.remove(0)
    gt_labels = list(gt_labels)
    gt_labels.remove(0)
    for g in gt_labels:
        for p in pred_labels:
            try:
                iou_d[g][p] = np.mean(iou_d[g][p])
            except Exception as e:
                pass

    checked = []
    result = []
    for g in gt_labels:
        for c in checked:
            if c in iou_d[g].keys():
                iou_d[g].pop(c)
        try:
            best_iou = max(iou_d.get(g).values())
        except Exception:
            best_iou = 0
        for k, v in iou_d[g].items():
            if v == best_iou:
                checked.append(k)
                print(g, k, best_iou)
        result.append(best_iou)

    return np.mean(result), result


def IOU(gt, pred, eps=1e-5):
    intersect = np.sum(np.logical_and(gt > 0, pred > 0)) + eps
    union = np.sum(np.logical_or(gt > 0, pred > 0)) + eps
    iou = float(intersect / union)
    return iou


if __name__ == '__main__':
    pred_dir = '/workspace/solo/code/user_data/merge_data'
    ann_dir = r'/workspace/dataset/VOS/fusai_train/Annotations/'
    result, num = calculate_videos_miou(pred_dir, ann_dir)
    print(result, num)
