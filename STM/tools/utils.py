import numpy as np
from STM.models.loss.smooth_cross_entropy_loss import SmoothCrossEntropyLoss
from STM.models.loss.dice_loss import DiceLoss
import math
import torch
from torch import nn
from torch.nn import functional as F


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


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight.to(x.device), self.bias.to(x.device),
                        self.stride, self.padding, self.dilation, self.groups)