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


class TIANCHI(data.Dataset):
    '''
    Dataset for YOUTUBE to train
    '''

    def __init__(self, root, phase, imset='2016/val.txt', separate_instance=False, only_single=False,
                 target_size=(864, 480), clip_size=None, only_multiple=False, mode='sample', interval=1):
        assert phase in ['train']
        self.phase = phase
        self.root = root
        self.clip_size = clip_size
        self.target_size = target_size
        self.SI = separate_instance  # 一个instance算一个视频
        if self.SI:
            assert not only_single
        self.OS = only_single  # 只统计只有一个instance的视频
        self.OM = only_multiple
        self.mode = mode
        self.interval = interval
        assert not (self.OM and self.OS)
        assert mode in ('sample', 'sequence')

        self.mask_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')
        _imset_dir = os.path.join(root, 'ImageSets')
        # print(_imset_dir)
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        # self.num_objects = {}
        self.shape = {}
        self.frame_list = {}
        self.mask_list = {}
        self.not_empty_frames = {}
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
                                                                                                      Image.BILINEAR))
                num_objects = _mask.max()
                if self.OM and num_objects == 1:
                    continue
                if self.OS and num_objects > 1:
                    continue
                if len(temp_img) != len(temp_mask):
                    continue
                if len(temp_img) < clip_size:
                    continue

                self.not_empty_frames.setdefault(_video, {})

                if self.SI:
                    temp_label = np.unique(_mask)
                    temp_label.sort()
                    # print(_video,temp_label)
                    for i in temp_label:
                        if i != 0:
                            self.not_empty_frames[_video].setdefault(i, {})

                            self.videos.append(_video + '_{}'.format(i))
                            self.num_frames[_video + '_{}'.format(i)] = len(
                                glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) + len(
                                glob.glob(os.path.join(self.image_dir, _video, '*.png')))
                            self.mask_list[_video + '_{}'.format(i)] = temp_mask
                            self.frame_list[_video + '_{}'.format(i)] = temp_img
                            self.shape[_video + '_{}'.format(i)] = np.shape(_mask)

                else:
                    if self.OS and np.max(_mask) > 1.1:
                        continue
                    self.videos.append(_video)
                    self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) + len(
                        glob.glob(os.path.join(self.image_dir, _video, '*.png')))
                    self.mask_list[_video] = temp_mask
                    self.frame_list[_video] = temp_img
                    # self.num_objects[_video] = np.max(_mask)
                    self.shape[_video] = np.shape(_mask)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        # print(self.videos[index])
        video = self.videos[index]
        frames = self.num_frames[video]
        if self.SI:
            video_true_name, object_label = video.split('_')
            object_label = int(object_label)
        else:
            video_true_name = video
            object_label = 1

        # print('phase',self.phase,self.clip_size)
        if isinstance(self.clip_size, int) and self.phase == 'train':
            # final_clip_size = self.clip_size
            final_clip_size = min(self.clip_size,self.num_frames[video])
        elif self.clip_size == 'all' and self.phase == 'train':
            final_clip_size = self.num_frames[video]
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

        if self.phase == 'train' and final_clip_size < self.num_frames[video] and self.mode == 'sample':
            p = [int(x / final_clip_size * frames) for x in range(1, final_clip_size)]
            p.insert(0, 0)
            p.append(frames - 1)
            frames_num = []
            for i in range(final_clip_size):
                frames_num.append(random.randint(p[i], p[i+1]))
        elif self.phase == 'train' and final_clip_size < self.num_frames[video] and self.mode == 'sequence':
            start_frame = random.randint(0, self.num_frames[video] - final_clip_size * self.interval - 1)
            frames_num = [start_frame + i * self.interval for i in range(final_clip_size)]
        else:
            frames_num = list(range(final_clip_size))


        for f in range(final_clip_size):
            img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][frames_num[f]])
            N_frames[f] = np.array(
                Image.open(img_file).convert('RGB').resize(self.target_size, Image.BILINEAR)) / 255.

            mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][frames_num[f]])
            temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.BILINEAR), dtype=np.uint8)

            temp_mask = np.zeros(temp.shape)
            if self.SI:
                temp_mask[temp == object_label] = 1
            else:
                temp_mask[temp > 0] = 1
            N_masks[f] = (temp_mask != 0).astype(np.uint8)

        # need_au = self.phase == 'train' and (np.random.rand() >= 0.3)
        # if need_au:
        #     seed = np.random.randint(99999)
        #     # print('seed:',seed)
        #     input_frames = (N_frames * 255).astype(np.uint8)
        #     for t in range(len(N_frames)):
        #         img_au, mask_au = self.aug(image=input_frames[t, np.newaxis, :, :, :].astype(np.uint8),
        #                                    mask=N_masks[t, np.newaxis, :, :, np.newaxis], seed=seed)
        #         N_frames[t] = img_au[0] / 255.
        #         N_masks[t] = mask_au[0, :, :, 0]

        Fs = torch.from_numpy(N_frames).permute(3, 0, 1, 2).float()
        Ms = torch.from_numpy(N_masks[np.newaxis, :, :, :]).long()

        sample = {
            'Fs': Fs, 'Ms': Ms, 'info': info
        }
        return sample

    def aug(self, image, mask, seed):
        ia.seed(seed)

        # Example batch of images.
        # The array has shape (32, 64, 64, 3) and dtype uint8.
        images = image  # B,H,W,C
        masks = mask  # B,H,W,C

        # print('In Aug',images.shape,masks.shape)
        combo = np.concatenate((images, masks), axis=3)
        # print('COMBO: ',combo.shape)

        seq_all = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=False)  # apply augmenters in random order

        seq_f = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.01))
                          ),
            # iaa.contrast.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=False)

        combo_aug = seq_all.augment_images(images=combo)
        # print('combo_au: ',combo_aug.shape)
        images_aug = combo_aug[:, :, :, :3]
        masks_aug = combo_aug[:, :, :, 3:]
        images_aug = seq_f.augment_images(images=images_aug)

        return images_aug, masks_aug


class TIANCHI_Stage1(data.Dataset):
    '''
    Dataset for YOUTUBE to train
    '''

    def __init__(self, root, phase, imset='2016/val.txt', resolution='480p', separate_instance=False, only_single=False,
                 target_size=(864, 480), clip_size=None, interval=1):
        assert phase in ['train']
        self.phase = phase
        self.root = root
        self.clip_size = clip_size
        self.target_size = target_size
        self.interval = interval
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
                                glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) + len(
                                glob.glob(os.path.join(self.image_dir, _video, '*.png')))
                            self.mask_list[_video + '_{}'.format(i)] = temp_mask
                            self.frame_list[_video + '_{}'.format(i)] = temp_img
                            # self.num_objects[_video + '_{}'.format(i)] = 1
                            self.shape[_video + '_{}'.format(i)] = np.shape(_mask)
                else:
                    if self.OS and np.max(_mask) > 1.1:
                        continue
                    self.videos.append(_video)
                    self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) + len(
                        glob.glob(os.path.join(self.image_dir, _video, '*.png')))
                    self.mask_list[_video] = temp_mask
                    self.frame_list[_video] = temp_img
                    # self.num_objects[_video] = np.max(_mask)
                    self.shape[_video] = np.shape(_mask)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        # print(self.videos[index])
        video = self.videos[index]
        frames = self.num_frames[video]
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

        # p1 = int(1/3 * frames)
        # p2 = int(2/3 * frames)
        frame_1 = random.randint(0, frames - self.clip_size - 1)
        # frame_2 = random.randint(p1, p2-1)
        # frame_3 = random.randint(p2, frames-1)
        frame_2 = frame_1 + 1
        frame_3 = frame_1 + 2
        info['interval'] = [frame_2 - frame_1, frame_3 - frame_2]

        frames_num = [frame_1, frame_2, frame_3]

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
