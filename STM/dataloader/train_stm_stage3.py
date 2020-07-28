from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import random
import argparse
import glob
import imgaug as ia
import imgaug.augmenters as iaa

# constants

PALETTE = [
    0, 0, 0,
    31, 119, 180,
    174, 199, 232,
    255, 127, 14,
    255, 187, 120,
    44, 160, 44,
    152, 223, 138,
    214, 39, 40,
    255, 152, 150,
    148, 103, 189,
    197, 176, 213,
    140, 86, 75,
    196, 156, 148,
    227, 119, 194,
    247, 182, 210,
    127, 127, 127,
    199, 199, 199,
    188, 189, 34,
    219, 219, 141,
    23, 190, 207,
    158, 218, 229
]


class font:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class DAVIS3(data.Dataset):
    '''
    Dataset for YOUTUBE to train
    '''

    def __init__(self, root, phase, imset='2016/val.txt', resolution='480p', separate_instance=False, only_single=False,
                 target_size=(864, 480), clip_size=None):
        assert phase in ['train']
        self.phase = phase
        self.root = root
        self.clip_size = clip_size
        self.target_size = target_size
        self.SI = separate_instance  # 一个instance算一个视频
        if self.SI:
            assert not only_single
        self.OS = only_single  # 只统计只有一个instance的视频

        if imset[0] != '2':
            self.mask_dir = os.path.join(root, 'Annotations')
            self.image_dir = os.path.join(root, 'JPEGImages')
        else:
            self.mask_dir = os.path.join(root, 'Annotations', resolution)
            self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        # print(_imset_dir)
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        # self.num_objects = {}
        self.shape = {}
        self.frame_list = {}
        self.mask_list = {}
        # print(_imset_f)
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                temp_img = os.listdir(os.path.join(self.image_dir, _video))
                temp_img.sort()

                temp_mask = os.listdir(os.path.join(self.mask_dir, _video))
                temp_mask.sort()
                _mask = np.array(
                    Image.open(os.path.join(self.mask_dir, _video, temp_mask[0])).convert("P").resize(self.target_size,
                                                                                                      Image.NEAREST))

                if self.SI:
                    temp_label = np.unique(_mask)
                    temp_label.sort()
                    # print(_video,temp_label)
                    for i in temp_label:
                        if i != 0:
                            self.videos.append(_video + '_{}'.format(i))
                            self.num_frames[_video + '_{}'.format(i)] = len(
                                glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                            self.mask_list[_video + '_{}'.format(i)] = temp_mask
                            self.frame_list[_video + '_{}'.format(i)] = temp_img
                            # self.num_objects[_video + '_{}'.format(i)] = 1
                            self.shape[_video + '_{}'.format(i)] = np.shape(_mask)
                else:
                    if self.OS and np.max(_mask) > 1.1:
                        continue
                    self.videos.append(_video)
                    self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                    self.mask_list[_video] = temp_mask
                    self.frame_list[_video] = temp_img
                    # self.num_objects[_video] = np.max(_mask)
                    self.shape[_video] = np.shape(_mask)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        # print(self.videos[index])
        video = self.videos[index]
        if self.SI:
            video_true_name, object_label = video.split('_')
            object_label = int(object_label)
        else:
            video_true_name = video
            object_label = 1

        # print('phase',self.phase,self.clip_size)
        if isinstance(self.clip_size, int) and self.phase == 'train':
            final_clip_size = self.clip_size
            # final_clip_size = min(self.clip_size,self.num_frames[video])
        elif self.phase == 'val' and (self.clip_size is None):
            final_clip_size = self.num_frames[video]
        else:
            print(f'wrong clip_size, should be an Integer but got {self.clip_size} and phase {self.phase}')
            raise ValueError

        info = {}
        info['name'] = video
        info['num_frames'] = final_clip_size

        N_frames = np.empty((final_clip_size,) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((final_clip_size,) + self.shape[video], dtype=np.uint8)


        frame_first = random.randint(0,35)
        frame_second = random.randint(36, 60)
        frame_third = random.randint(61, 84)
        if frame_third > self.num_frames[video]-1:
            frame_first = random.randint(0,15)
            frame_second = random.randint(16,40)
            frame_third = random.randint(40,64)
            if frame_third > self.num_frames[video]-1:
                frame_first = random.randint(0, 10)
                frame_second = random.randint(11, 20)
                frame_third = random.randint(21, 34)
                if frame_third > self.num_frames[video]-1:
                    frame_first = random.randint(0, 5)
                    frame_second = random.randint(6, 12)
                    frame_third = random.randint(13, 19)

        frames_num = [frame_first, frame_second, frame_third]


        for f in range(final_clip_size):
            img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][frames_num[f]])
            N_frames[f] = np.array(
                Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS)) / 255.

            mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][frames_num[f]])
            temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
            if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                print(np.unique(temp))
            temp_mask = np.zeros(temp.shape)
            if self.SI:
                temp_mask[temp == object_label] = 1
            else:
                temp_mask[temp > 0] = 1
            N_masks[f] = (temp_mask != 0).astype(np.uint8)

        Fs = torch.from_numpy(N_frames).permute(3, 0, 1, 2).float()
        Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()

        sample = {
            'Fs': Fs, 'Ms': Ms, 'info': info
        }
        return sample


