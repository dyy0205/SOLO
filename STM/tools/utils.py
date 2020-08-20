import numpy as np
from STM.models.loss.smooth_cross_entropy_loss import SmoothCrossEntropyLoss
from STM.models.loss.dice_loss import DiceLoss
import torch


_ce_loss = SmoothCrossEntropyLoss(eps=1e-3)
_dice_loss = DiceLoss(ignore_index=0)


def _loss(x, y):
    return _ce_loss(x, y) + _dice_loss(x, y)


def get_video_mIoU(predn, all_Mn):  # [c,t,h,w]
    pred = predn.squeeze().cpu().data.numpy()
    # np.save('blackswan.npy', pred)
    if isinstance(all_Mn, torch.Tensor):
        gt = all_Mn.squeeze().cpu().data.numpy()  # [t,h,w]
    else:
        gt = all_Mn.squeeze()
    agg = pred + gt
    i = float(np.sum(agg == 2))
    u = float(np.sum(agg > 0))
    return i / (u + 1e-6)