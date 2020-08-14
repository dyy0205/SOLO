from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

print('Space-time Memory Networks: initialized.')


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m):
        f = (in_f - self.mean) / self.std
        o = torch.zeros_like(in_m).float()
        # in_m.shape [1,1,480,864],[b,c,,h,w]
        x = self.conv1(f) + self.conv1_m(in_m) #+ self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=True)
        return p  # , p2, p3, p4


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class GCM(nn.Module):
    def __init__(self):
        super(GCM, self).__init__()

    def forward(self, g, key_q, value_q):
        '''
        :param keys_m: [B,C,T,H,W], c = 128
        :param values_m: [B,C,T,H,W], c = 512
        :param key_q: [B,C,H,W], c = 128
        :param value_q: [B,C,H,W], c = 512
        :return: final_value [B, C, H, W]
        '''
        B, C_key, H, W = key_q.size()
        key_q = key_q.view(B, C_key, H * W)
        # g = F.softmax(g, dim=1)
        val_temp = torch.bmm(g, key_q)  # [b, c_val, hw]
        val_temp = val_temp / C_key
        val_temp = val_temp.view(B, 4 * C_key, H, W)
        val_final = torch.cat([val_temp, value_q], dim=1)

        return val_final


class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.GCM = GCM()
        self.Decoder = Decoder(256)

    def segment(self, frame, g):
        '''
        :param frame: 当前需要分割的image；[B,C,H,W]
        :param key: 当前memory的key；[B,C,,T,H,W]
        :param value: 当前memory的value; [B,C,T,H,W]
        :return: logits []
        '''
        #encode
        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        curKey, curValue = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16

        # memory select
        final_value = self.GCM(g, curKey, curValue)
        logits = self.Decoder(final_value, r3, r2) #[b,2,h,w]
        ps = F.softmax(logits, dim=1)[:, 1]  # B h w
        B, H, W = ps.shape
        ps_tmp = torch.unsqueeze(ps, dim=1)  # B,1,H,W
        em = torch.zeros(B, 2, H, W).cuda()
        em[:, 0] = torch.prod(1-ps_tmp, dim=1)
        em[:, 1] = ps
        em = torch.clamp(em, 1e-7, 1- 1e-7)
        logit = torch.log((em / (1-em)))



        return logit


    def memorize(self, curFrame, curMask):
        '''
        将当前帧编码
        :param curFrame: [b,c,h,w]
        :param curMask: [b,c,h,w]
        :return: 编码后的key与value
        '''
        # print('&&&&&&&&&', curMask.shape, curFrame.shape)
        r4, _, _, _, _ = self.Encoder_M(curFrame, curMask)
        # print('******r4', r4.device)
        k4, v4 = self.KV_M_r4(r4)  # num_objects, 128 and 512, H/16, W/16

        # calculate Ct
        [b, c_key, h, w] = k4.shape
        [_, c_val, _, _] = v4.shape
        k4 = k4.view(b, c_key, h*w)
        v4 = v4.view(b, c_val, h*w)
        k4 = torch.transpose(k4, 2, 1)
        Ct = torch.bmm(v4, k4)
        Ct = Ct / (h*w)
        return Ct


    def forward(self, args, mode=''):
        #args: Fs[:,:,t-1]
        #kwargs: Es[:,:,t-1]
        if mode=='segment': # keys
            return self.segment(args[0], args[1])
        else:
            return self.memorize(args[0], args[1])

