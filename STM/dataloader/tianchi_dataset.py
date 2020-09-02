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
    Dataset for TIANCHI to train
    '''

    def __init__(self, root, phase='train', imset='2016/val.txt', separate_instance=False, only_single=False,
                 target_size=(864, 480), crop_size=(480, 480), clip_size=3, only_multiple=False, mode='sample',
                 interval=1, same_frames=False, train_aug=False, keep_one_prev=False, add_prev_mask=False):
        assert phase in ['train', 'test', 'val']
        self.phase = phase
        self.root = root
        self.clip_size = clip_size
        self.target_size = target_size
        self.crop_size = crop_size
        self.SI = separate_instance  # 一个instance算一个视频
        if self.SI:
            assert not only_single
        self.OS = only_single  # 只统计只有一个instance的视频
        self.OM = only_multiple
        self.mode = mode
        self.interval = interval
        self.train_aug = train_aug
        assert not (self.OM and self.OS)
        assert mode in ('sample', 'sequence')
        self.same_frames = same_frames
        self.keep_one_prev = keep_one_prev
        self.add_prev_mask = add_prev_mask

        self.mask_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.shape = {}
        self.frame_list = {}
        self.mask_list = {}
        self.not_empty_frames = {}
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
                if len(temp_img) < clip_size * interval - 1:
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

        if same_frames:
            self.max_frames = max(self.num_frames.values())
        else:
            self.max_frames = 0

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
            final_clip_size = min(self.clip_size, self.num_frames[video])
        elif self.clip_size == 'all' and self.phase == 'train':
            final_clip_size = self.num_frames[video]
        elif (self.phase == 'val' or self.phase == 'test') and not self.same_frames:
            final_clip_size = self.num_frames[video]
        elif (self.phase == 'val' or self.phase == 'test') and self.same_frames:
            final_clip_size = self.max_frames
        else:
            print(f'wrong clip_size, should be an Integer but got {self.clip_size} and phase {self.phase}')
            raise ValueError

        info = {}
        info['name'] = video
        info['num_frames'] = final_clip_size
        info['valid_frames'] = np.zeros(final_clip_size)
        info['target_shape'] = self.target_size

        N_frames = np.empty((final_clip_size,) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((final_clip_size,) + self.shape[video], dtype=np.uint8)
        if self.add_prev_mask:
            N_prevs = np.empty((final_clip_size-1,) + self.shape[video], dtype=np.uint8)

        # generate frame numbers
        if self.phase == 'train' and final_clip_size < self.num_frames[video] and self.mode == 'sample':
            p = [int(x / final_clip_size * frames) for x in range(1, final_clip_size)]
            p.insert(0, 0)
            p.append(frames - 1)
            frames_num = []
            for i in range(final_clip_size):
                frames_num.append(random.randint(p[i], p[i + 1]))
        elif self.phase == 'train' and final_clip_size <= self.num_frames[video] and self.mode == 'sequence':
            ed = max(self.num_frames[video] - (final_clip_size - 1) * self.interval - 1, 0)
            while True:
                start_frame = random.randint(0, ed)
                mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][start_frame])
                temp = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                if (temp == object_label).sum() > 0:
                    break
            if not self.keep_one_prev:
                frames_num = [start_frame + i * self.interval for i in range(final_clip_size)]
            else:
                frames_num = [start_frame + i * self.interval for i in range(final_clip_size - 1)]
                frames_num.append(frames_num[-1] + 1)
            frames_num = [min(self.num_frames[video] - 1, fn) for fn in frames_num]
        elif self.same_frames:
            frames_num = list(range(self.num_frames[video]))
        else:
            frames_num = list(range(final_clip_size))

        intervals = [0]
        for i in range(1, final_clip_size):
            intervals.append(frames_num[i] - frames_num[i - 1])
        info['frames'] = frames_num
        info['intervals'] = intervals

        for f in range(len(frames_num)):
            info['valid_frames'][f] = 1
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

        if self.add_prev_mask:
            prevs_num = [frames_num[i] - 1 for i in range(1, len(frames_num))]
            for f in range(len(prevs_num)):
                mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][prevs_num[f]])
                temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.BILINEAR),
                                dtype=np.uint8)

                temp_mask = np.zeros(temp.shape)
                if self.SI:
                    temp_mask[temp == object_label] = 1
                else:
                    temp_mask[temp > 0] = 1
                N_prevs[f] = (temp_mask != 0).astype(np.uint8)

        # first frame should not be empty
        assert np.any(N_masks[0])

        # augmentation
        N_frames_ = []
        N_masks_ = []
        N_prevs_ = []
        if self.phase == 'train' and self.train_aug:
            # seed = np.random.randint(99999)
            crop_size = random.choice([(384, 384), (416, 416), (448, 448), (480, 480)])
            input_frames = (N_frames * 255).astype(np.uint8)
            for t in range(len(N_frames)):
                count = 0
                tmp_result = []
                ious = []
                while count < 100:
                    if t > 0 and self.add_prev_mask:
                        masks = np.stack((N_masks[t], N_prevs[t-1]), axis=2)
                        img_au, masks_au = self.aug(image=input_frames[t, np.newaxis, :, :, :].astype(np.uint8),
                                                   mask=masks[np.newaxis, :, :, :], crop_size=crop_size)
                        mask_au = masks_au[0, :, :, 0]
                        prev_au = masks_au[0, :, :, 1]
                        tmp_result.append((img_au, mask_au, prev_au))
                    else:
                        img_au, mask_au = self.aug(image=input_frames[t, np.newaxis, :, :, :].astype(np.uint8),
                                                   mask=N_masks[t, np.newaxis, :, :, np.newaxis], crop_size=crop_size)
                        mask_au = mask_au[0, :, :, 0]
                        tmp_result.append((img_au, mask_au))
                    iou = float(np.sum(mask_au)) / float(np.sum(N_masks[t]) + 1e-6)
                    ious.append(iou)
                    if np.sum(N_masks[t]) == 0 or (np.sum(N_masks[t]) > 0 and iou > 0.5):
                        N_frames_.append(np.array(Image.fromarray(img_au[0]).resize(self.crop_size)) / 255.)
                        N_masks_.append(np.array(Image.fromarray(mask_au).resize(self.crop_size)))
                        if t > 0 and self.add_prev_mask:
                            N_prevs_.append(np.array(Image.fromarray(prev_au).resize(self.crop_size)))
                        break
                    count += 1
                if count >= 100:
                    print(video)
                    idx = np.argmax(ious)
                    # assert ious[idx] > 0
                    if t > 0 and self.add_prev_mask:
                        img_au, mask_au, prev_au = tmp_result[idx]
                    else:
                        img_au, mask_au = tmp_result[idx]
                    N_frames_.append(np.array(Image.fromarray(img_au[0]).resize(self.crop_size)) / 255.)
                    N_masks_.append(np.array(Image.fromarray(mask_au).resize(self.crop_size)))
                    if t > 0 and self.add_prev_mask:
                        N_prevs_.append(np.array(Image.fromarray(prev_au).resize(self.crop_size)))

            assert len(N_frames_) == final_clip_size
            Fs = torch.from_numpy(np.array(N_frames_)).permute(3, 0, 1, 2).float()
            Ms = torch.from_numpy(np.array(N_masks_)[np.newaxis, :, :, :]).long()
            if self.add_prev_mask:
                Ps = torch.from_numpy(np.array(N_prevs_)[np.newaxis, :, :, :]).long()
        else:
            Fs = torch.from_numpy(N_frames).permute(3, 0, 1, 2).float()
            Ms = torch.from_numpy(N_masks[np.newaxis, :, :, :]).long()
            if self.add_prev_mask:
                Ps = torch.from_numpy(N_prevs[np.newaxis, :, :, :]).long()

        if self.add_prev_mask:
            sample = {
                'Fs': Fs, 'Ms': Ms, 'Ps': Ps, 'info': info
            }
        else:
            sample = {
                'Fs': Fs, 'Ms': Ms, 'info': info
            }
        return sample

    def aug(self, image, mask, crop_size, seed=None):
        # ia.seed(seed)

        # Example batch of images.
        # The array has shape (32, 64, 64, 3) and dtype uint8.
        images = image  # B,H,W,C
        masks = mask  # B,H,W,C

        # print('In Aug',images.shape,masks.shape)
        combo = np.concatenate((images, masks), axis=3)
        # print('COMBO: ',combo.shape)

        seq_all = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.PadToFixedSize(width=crop_size[0], height=crop_size[1]),
            iaa.CropToFixedSize(width=crop_size[0], height=crop_size[1]),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 90-110% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                # translate by -10 to +10 percent (per axis)
                rotate=(-5, 5),  # rotate by -5 to +5 degrees
                shear=(-3, 3),  # shear by -3 to +3 degrees
            )
        ], random_order=False)  # apply augmenters in random order

        seq_f = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.OneOf([
                              iaa.GaussianBlur((0.0, 3.0)),
                              iaa.MotionBlur(k=(3, 7))
                          ]),
                          ),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.0))),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
        ], random_order=False)

        combo_aug = np.array(seq_all.augment_images(images=combo))
        # print('combo_au: ', combo_aug.shape)
        images_aug = combo_aug[:, :, :, :3]
        masks_aug = combo_aug[:, :, :, 3:]
        images_aug = seq_f.augment_images(images=images_aug)

        return images_aug, masks_aug

# class TIANCHI_Stage1(data.Dataset):
#     '''
#     Dataset for YOUTUBE to train
#     '''
#
#     def __init__(self, root, phase, imset='2016/val.txt', resolution='480p', separate_instance=False, only_single=False,
#                  target_size=(864, 480), clip_size=None, interval=1):
#         assert phase in ['train']
#         self.phase = phase
#         self.root = root
#         self.clip_size = clip_size
#         self.target_size = target_size
#         self.interval = interval
#         self.SI = separate_instance  # 一个instance算一个视频
#         if self.SI:
#             assert not only_single
#         self.OS = only_single  # 只统计只有一个instance的视频
#
#         if imset[0] != '2':
#             self.mask_dir = os.path.join(root, 'Annotations')
#             self.image_dir = os.path.join(root, 'JPEGImages')
#         else:
#             self.mask_dir = os.path.join(root, 'Annotations', resolution)
#             self.image_dir = os.path.join(root, 'JPEGImages', resolution)
#         _imset_dir = os.path.join(root, 'ImageSets')
#         # print(_imset_dir)
#         _imset_f = os.path.join(_imset_dir, imset)
#
#         self.videos = []
#         self.num_frames = {}
#         # self.num_objects = {}
#         self.shape = {}
#         self.frame_list = {}
#         self.mask_list = {}
#         # print(_imset_f)
#         with open(os.path.join(_imset_f), "r") as lines:
#             for line in lines:
#                 _video = line.rstrip('\n')
#                 temp_img = os.listdir(os.path.join(self.image_dir, _video))
#                 temp_img.sort()
#
#                 temp_mask = os.listdir(os.path.join(self.mask_dir, _video))
#                 temp_mask.sort()
#                 _mask = np.array(
#                     Image.open(os.path.join(self.mask_dir, _video, temp_mask[0])).convert("P").resize(self.target_size,
#                                                                                                       Image.NEAREST))
#
#                 if self.SI:
#                     temp_label = np.unique(_mask)
#                     temp_label.sort()
#                     # print(_video,temp_label)
#                     for i in temp_label:
#                         if i != 0:
#                             self.videos.append(_video + '_{}'.format(i))
#                             self.num_frames[_video + '_{}'.format(i)] = len(
#                                 glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) + len(
#                                 glob.glob(os.path.join(self.image_dir, _video, '*.png')))
#                             self.mask_list[_video + '_{}'.format(i)] = temp_mask
#                             self.frame_list[_video + '_{}'.format(i)] = temp_img
#                             # self.num_objects[_video + '_{}'.format(i)] = 1
#                             self.shape[_video + '_{}'.format(i)] = np.shape(_mask)
#                 else:
#                     if self.OS and np.max(_mask) > 1.1:
#                         continue
#                     self.videos.append(_video)
#                     self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) + len(
#                         glob.glob(os.path.join(self.image_dir, _video, '*.png')))
#                     self.mask_list[_video] = temp_mask
#                     self.frame_list[_video] = temp_img
#                     # self.num_objects[_video] = np.max(_mask)
#                     self.shape[_video] = np.shape(_mask)
#
#     def __len__(self):
#         return len(self.videos)
#
#     def __getitem__(self, index):
#         # print(self.videos[index])
#         video = self.videos[index]
#         frames = self.num_frames[video]
#         if self.SI:
#             video_true_name, object_label = video.split('_')
#             object_label = int(object_label)
#         else:
#             video_true_name = video
#             object_label = 1
#
#         # print('phase',self.phase,self.clip_size)
#         if isinstance(self.clip_size, int) and self.phase == 'train':
#             final_clip_size = self.clip_size
#             # final_clip_size = min(self.clip_size,self.num_frames[video])
#         elif self.phase == 'val' and (self.clip_size is None):
#             final_clip_size = self.num_frames[video]
#         else:
#             print(f'wrong clip_size, should be an Integer but got {self.clip_size} and phase {self.phase}')
#             raise ValueError
#
#         info = {}
#         info['name'] = video
#         info['num_frames'] = final_clip_size
#
#         N_frames = np.empty((final_clip_size,) + self.shape[video] + (3,), dtype=np.float32)
#         N_masks = np.empty((final_clip_size,) + self.shape[video], dtype=np.uint8)
#
#         # p1 = int(1/3 * frames)
#         # p2 = int(2/3 * frames)
#         frame_1 = random.randint(0, frames - self.clip_size - 1)
#         # frame_2 = random.randint(p1, p2-1)
#         # frame_3 = random.randint(p2, frames-1)
#         frame_2 = frame_1 + 1
#         frame_3 = frame_1 + 2
#         info['interval'] = [frame_2 - frame_1, frame_3 - frame_2]
#
#         frames_num = [frame_1, frame_2, frame_3]
#
#         for f in range(final_clip_size):
#             img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][frames_num[f]])
#             N_frames[f] = np.array(
#                 Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS)) / 255.
#
#             mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][frames_num[f]])
#             temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
#             if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#                 print(np.unique(temp))
#             temp_mask = np.zeros(temp.shape)
#             if self.SI:
#                 temp_mask[temp == object_label] = 1
#             else:
#                 temp_mask[temp > 0] = 1
#             N_masks[f] = (temp_mask != 0).astype(np.uint8)
#
#         Fs = torch.from_numpy(N_frames).permute(3, 0, 1, 2).float()
#         Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()
#
#         sample = {
#             'Fs': Fs, 'Ms': Ms, 'info': info
#         }
#         return sample
