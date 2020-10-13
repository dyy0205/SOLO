from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import numpy as np
from PIL import Image
import os, glob, time

def IOU(gt, pred, eps=1e-5):
    intersect = np.sum(np.logical_and(gt > 0, pred > 0)) + eps
    union = np.sum(np.logical_or(gt > 0, pred > 0)) + eps
    iou = float(intersect / union)
    iof = float(intersect / (np.sum(gt > 0) + eps))
    return iou, iof


root = '/versa/dyy/dataset/TIANCHI'
preds = sorted(glob.glob(os.path.join(root, 'v3_out/639838/*.png')))

first_frame = Image.open(preds[0]).convert('P')
palette = first_frame.getpalette()
w, h = first_frame.size
inst = [(np.array(first_frame) == i) for i in np.unique(first_frame)[1:]] if len(np.unique(first_frame)) > 1 else []
print(len(inst))

save_dir = os.path.join(root, '639838')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
first_frame.save(os.path.join(save_dir, preds[0].split('/')[-1]))

for i in range(1, len(preds)):
    pred_i = np.array(Image.open(preds[i]).convert('P'))
    masks = np.unique(first_frame)[1:]
    blend = np.zeros((h, w), dtype=np.uint8)
    if len(masks) > 0:
        if len(inst) > 0 and len(inst) == len(masks):
            for j in masks:
                mask = (pred_i == j)
                ious = []
                for ins in inst:
                    iou, iof = IOU(mask, ins)
                    ious.append(iof)
                ind = np.argmax(ious)
                blend[mask > 0] = ind + 1
        else:
             blend = pred_i
        blend = Image.fromarray(blend)
        blend.putpalette(palette)
    else:
        blend = Image.fromarray(blend)
    blend.save(os.path.join(save_dir, preds[i].split('/')[-1]))
    inst = [(np.array(blend) == i) for i in np.unique(blend)[1:]] if len(np.unique(blend)) > 1 else []