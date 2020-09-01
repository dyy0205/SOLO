from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

# draw
import numpy as np
import matplotlib.pyplot as plt

print('Space-time Memory Networks: initialized.')


class ASPP(nn.Module):
    def __init__(self, in_channel=1024, depth=256):
        super(ASPP, self).__init__()
        # self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        # self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        # self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 3, 1, padding=1, dilation=1)
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block8 = nn.Conv2d(in_channel, depth, 3, 1, padding=8, dilation=8)
        # self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth, 1, 1)

    def forward(self, x):
        # size = x.shape[2:]
        #
        # image_features = self.mean(x)
        # image_features = self.conv(image_features)
        # image_features = F.upsample(image_features, size=size, mode='bilinear')

        # atrous_block1 = self.atrous_block1(x)
        # atrous_block6 = self.atrous_block6(x)
        # atrous_block12 = self.atrous_block12(x)
        # atrous_block18 = self.atrous_block18(x)

        # net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
        #                                       atrous_block12, atrous_block18], dim=1))
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block8 = self.atrous_block8(x)

        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block2, atrous_block4, atrous_block8], dim=1))

        return net


class Spatial_prior(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_prior, self).__init__()
        self.conv = nn.Conv2d(in_channel, 1, 3, stride=1, padding=1)
        self.conv_b = nn.Conv2d(in_channel, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, values, prev_mask):
        b, c, h, w = values.shape
        pooling = nn.AdaptiveAvgPool2d((h, w))
        prev_mask = pooling(prev_mask)
        prev_mask = torch.round(prev_mask)
        vc = torch.cat([values, prev_mask], dim=1)
        m = self.conv(vc)
        m = self.sigmoid(m)
        b = self.sigmoid(self.conv_b(vc))
        values = values * (m + b)
        return values


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
        in_m = self.conv1_m(in_m)
        x = self.conv1(f) + in_m  # + self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f, x


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
        return r4, r3, r2, c1, f, x


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.Aspp = ASPP(in_channel=1024, depth=256)
        # self.SP = Spatial_prior(257)

        # self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        r4 = self.Aspp(r4)
        # r4 = self.SP(r4, prev_mask)

        m4 = self.ResMM(r4)
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred2(F.relu(m3))
        p4 = self.pred2(F.relu(m4))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, scale_factor=16, mode='bilinear', align_corners=False)
        return p, p3, p4  # , p2, p3, p4


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


# class Memory(nn.Module):
#     def __init__(self):
#         super(Memory, self).__init__()
#
#     def forward(self, keys_m, values_m, key_q, value_q):
#         '''
#         :param keys_m: [B,C,T,H,W], c = 128
#         :param values_m: [B,C,T,H,W], c = 512
#         :param key_q: [B,C,H,W], c = 128
#         :param value_q: [B,C,H,W], c = 512
#         :return: final_value [B, C, H, W]
#         '''
#         B, C_key, T, H, W = keys_m.size()
#         # print('#####', B, C_key, T, H, W)
#         _, C_value, _, _, _ = values_m.size()
#
#         keys_m_temp = keys_m.view(B, C_key, T * H * W)
#         keys_m_temp = torch.transpose(keys_m_temp, 1, 2)  # [b,thw,c]
#
#         key_q_temp = key_q.view(B, C_key, H * W)  # [b,c,hw]
#
#         p = torch.bmm(keys_m_temp, key_q_temp)  # [b, thw, hw]
#         p = p / math.sqrt(C_key)
#         p = F.softmax(p, dim=1)  # b, thw, hw
#
#         mo = values_m.view(B, C_value, T * H * W)  # [b,c,thw]
#         mem = torch.bmm(mo, p)  # Weighted-sum B, c, hw
#         mem = mem.view(B, C_value, H, W)
#
#         final_value = torch.cat([mem, value_q], dim=1)
#         # print('mem:', torch.max(mem), torch.min(mem))
#         # print('value_q:', torch.max(value_q), torch.min(value_q))
#
#         return final_value


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.conv_w = nn.Conv2d(129, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.conv_b = nn.Conv2d(129, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, keys_m, values_m, key_q, value_q, mask):
        '''
        :param keys_m: [B,C,T,H,W], c = 128
        :param values_m: [B,C,T,H,W], c = 512
        :param key_q: [B,C,H,W], c = 128
        :param value_q: [B,C,H,W], c = 512
        :return: final_value [B, C, H, W]
        '''
        B, C_key, T, H, W = keys_m.size()
        # print('#####', B, C_key, T, H, W)
        _, C_value, _, _, _ = values_m.size()

        p_mask = self.process_mask(mask, (H, W))  # B, 1, H, W
        key_q_c = torch.cat([key_q, p_mask], dim=1)  # B, Ck+1, H, W
        w = self.conv_w(key_q_c)
        w = self.sigmoid(w)
        b = self.conv_b(key_q_c)
        b = self.sigmoid(b)
        p_mask = w * p_mask + b  # B, 1, H, W
        p_mask = p_mask.view(B, 1, H * W)  # b, 1, h*w
        p_mask = p_mask.expand(-1, T * H * W, -1)

        keys_m_temp = keys_m.view(B, C_key, T * H * W)
        keys_m_temp = torch.transpose(keys_m_temp, 1, 2)  # [b,thw,c]

        key_q_temp = key_q.view(B, C_key, H * W)  # [b,c,hw]

        p = torch.bmm(keys_m_temp, key_q_temp)  # [b, thw, hw]
        p = p / math.sqrt(C_key)
        p = F.softmax(p, dim=1)  # b, thw, hw

        km = p * p_mask  # b, thw, hw

        mo = values_m.view(B, C_value, T * H * W)  # [b,c,thw]
        mem = torch.bmm(mo, km)  # Weighted-sum B, c, hw
        mem = mem.view(B, C_value, H, W)

        final_value = torch.cat([mem, value_q], dim=1)
        # print('mem:', torch.max(mem), torch.min(mem))
        # print('value_q:', torch.max(value_q), torch.min(value_q))

        return final_value

    @staticmethod
    def process_mask(mask, shape):
        pool = nn.AdaptiveAvgPool2d(shape)
        mask = pool(mask)
        b, c, h, w = mask.shape
        p_mask = torch.zeros_like(mask)
        for i in range(b):
            m_ = mask[i].squeeze(0)
            m_ = m_.cpu().numpy()
            if not np.any(m_):
                m_ = np.ones_like(m_)
            m_ *= 255
            m_ = cv2.GaussianBlur(m_, (7, 7), 0, 0)
            m_ = torch.from_numpy(m_).unsqueeze(0)
            m_ /= 255
            p_mask[i] = m_

        return p_mask


class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)

    def segment(self, frame, key, value, prev_mask):
        '''
        :param frame: 当前需要分割的image；[B,C,H,W]
        :param key: 当前memory的key；[B,C,T,H,W]
        :param value: 当前memory的value; [B,C,T,H,W]
        :return: logits []
        '''
        # encode
        r4, r3, r2, _, _, x = self.Encoder_Q(frame)
        curKey, curValue = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16

        # memory select
        final_value = self.Memory(key, value, curKey, curValue, prev_mask)
        logits, p3, p4 = self.Decoder(final_value, r3, r2)  # [b,2,h,w]
        logits = self.get_logit(logits)
        p3 = self.get_logit(p3)
        p4 = self.get_logit(p4)

        return logits, p3, p4

    @staticmethod
    def get_logit(logits):
        ps = F.softmax(logits, dim=1)[:, 1]  # B h w
        B, H, W = ps.shape
        ps_tmp = torch.unsqueeze(ps, dim=1)  # B,1,H,W
        em = torch.zeros(B, 2, H, W).cuda()
        em[:, 0] = torch.prod(1 - ps_tmp, dim=1)
        em[:, 1] = ps
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def memorize(self, curFrame, curMask):
        '''
        将当前帧编码
        :param curFrame: [b,c,h,w]
        :param curMask: [b,c,h,w]
        :return: 编码后的key与value
        '''
        # print('&&&&&&&&&', curMask.shape, curFrame.shape)
        r4, _, _, _, _, x = self.Encoder_M(curFrame, curMask)
        k4, v4 = self.KV_M_r4(r4)  # num_objects, 128 and 512, H/16, W/16
        return k4, v4

    def forward(self, args):
        # args: Fs[:,:,t-1]
        # kwargs: Es[:,:,t-1]
        if len(args) > 2:  # keys
            return self.segment(*args)
        else:
            return self.memorize(*args)
