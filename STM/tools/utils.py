import numpy as np
from STM.models.loss.smooth_cross_entropy_loss import SmoothCrossEntropyLoss
from STM.models.loss.dice_loss import DiceLoss
import torch


_ce_loss = SmoothCrossEntropyLoss(eps=1e-3)
_dice_loss = DiceLoss(ignore_index=0)


def _loss(x, y):
    return _ce_loss(x, y) + _dice_loss(x, y)


def get_video_mIoU(predn, all_Mn):  # [c,t,h,w]
    # use cuda
    if isinstance(predn, np.ndarray):
        pred = torch.from_numpy(predn.squeeze()).cuda()
    elif isinstance(predn, torch.Tensor):
        pred = predn.squeeze().cuda().detach()
    # np.save('blackswan.npy', pred)
    if isinstance(all_Mn, torch.Tensor):
        gt = all_Mn.squeeze().cuda().detach()  # [t,h,w]
    elif isinstance(all_Mn, np.ndarray):
        gt = torch.from_numpy(all_Mn.squeeze()).cuda()
    agg = pred + gt
    i = float(torch.sum(agg == 2))
    u = float(torch.sum(agg > 0))
    return i / (u + 1e-6)