#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        x: A tensor of shape [N, *]
        target: A tensor of shape same with x
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, power=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.p = power
        self.reduction = reduction

    def forward(self, x, target):
        assert x.shape[0] == target.shape[0], "x & target batch size don't match"
        x = x.contiguous().view(x.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(x, target))
        den = torch.sum(x.pow(self.p) + target.pow(self.p))
        try:
            if den.item() == 0:
                return 0
        except Exception as e:
            print('!!!![ERROR]:',e)
            return 0
        loss = - torch.log(2. * num / den + 1e-5)
        return loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index set to ignore
        x: A tensor of shape [N, C, *]
        target: A tensor of same shape with x
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, power=2, reduction='mean', weight=None, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.weight = weight
        if isinstance(ignore_index, set):
            self.ignore_index = ignore_index
        elif isinstance(ignore_index, list):
            self.ignore_index = set(ignore_index)
        elif isinstance(ignore_index, tuple):
            self.ignore_index = set(ignore_index)
        elif isinstance(ignore_index, int):
            self.ignore_index = set([ignore_index])
        self.dice = BinaryDiceLoss(power, reduction)

    def forward(self, x, target):
        target_logit = torch.zeros(x.size(), device=x.device)
        target_logit.scatter_(1, target.unsqueeze(1), 1)
        target = target_logit
        assert x.shape == target.shape, 'x & target shape do not match'
        x = F.softmax(x, 1)
        total_loss = 0
        for i in range(target.size(1)):
            if i not in self.ignore_index:
                dice_loss = self.dice(x[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        return total_loss / (target.size(1) - len(self.ignore_index))

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.tensor([[
            [[0, 1],
             [1, 0]],
            [[1, 0],
             [0, 1.]],
        ]],
        device=device) * 1000.
    ylabel1 = torch.tensor([
            [[0, 1],
             [1, 0]],
        ],
        device=device)
    ylabel2 = torch.tensor([
            [[1, 0],
             [0, 1]],
        ],
        device=device)
    x3 = torch.tensor([[
            [[0, 0],
             [0, 0]],
            [[1, 1],
             [1, 1.]],
        ]],
        device=device) * 1000.
    ylabel3 = torch.tensor([
            [[1, 1],
             [1, 1]],
        ],
        device=device)

    print(DiceLoss(ignore_index=0)(x, ylabel1))
    print(DiceLoss(ignore_index=0)(x, ylabel2))
    print(DiceLoss(ignore_index=0)(x3, ylabel3))

    y = torch.zeros(x.size(), device=x.device)
    y.scatter_(1, ylabel1.unsqueeze(1), 1)
    print(BinaryDiceLoss()(F.softmax(x, 1)[:,1], y[:,1]))
    y = torch.zeros(x.size(), device=x.device)
    y.scatter_(1, ylabel2.unsqueeze(1), 1)
    print(BinaryDiceLoss()(F.softmax(x, 1)[:,1], y[:,1]))
    y = torch.zeros(x.size(), device=x.device)
    y.scatter_(1, ylabel3.unsqueeze(1), 1)
    print(BinaryDiceLoss()(F.softmax(x3, 1)[:,1], y[:,1]))
