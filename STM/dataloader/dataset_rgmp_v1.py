# #######
# This version plugs in some data augmentation operations from imgaug
# #######
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


class DAVIS(data.Dataset):
    '''
    Dataset for DAVIS
    '''

    def __init__(self, root, phase, imset='2016/val.txt', resolution='480p', separate_instance=False, only_single=False,
                 target_size=(864, 480), clip_size=None):
        assert phase in ['train', 'val']
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
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        # self.num_objects = {}
        self.shape = {}
        self.frame_list = {}
        self.mask_list = {}
        self.num_objects = {}
        # self.object_label={}
        # print('SI:',self.SI)
        # print('OS:',self.OS)
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
                self.num_objects[_video] = np.max(_mask)

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
            num_objects = 1
        else:
            num_objects = int(self.num_objects[video])
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

        if self.phase == 'train':
            if self.num_frames[video] > final_clip_size:
                start_frame = np.random.randint(0, (self.num_frames[video] - final_clip_size) + 1)
            else:
                start_frame = 0
            mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][start_frame])
            temp = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            if self.SI:

                while object_label not in list(np.unique(temp)):
                    if self.num_frames[video] > final_clip_size:
                        start_frame = np.random.randint(0, (self.num_frames[video] - final_clip_size) + 1)
                    else:
                        start_frame = 0
                    mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][start_frame])
                    temp = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

            else:

                while len(list(np.unique(temp))) <= 1:
                    if self.num_frames[video] > final_clip_size:
                        start_frame = np.random.randint(0, (self.num_frames[video] - final_clip_size) + 1)
                    else:
                        start_frame = 0
                    mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][start_frame])
                    temp = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        elif self.phase == 'val':
            start_frame = 0
        else:
            print(f'wrong phase: {self.phase}')
            raise ValueError

        # print(video_true_name,start_frame,self.mask_list[video][start_frame],final_clip_size)
        for f in range(final_clip_size):
            target = f + start_frame
            if target >= self.num_frames[video]:
                target = 2 * self.num_frames[video] - target - 2
            if target < 0:
                target = target * (-1)
            try:
                img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][target])
                N_frames[f] = np.array(
                    Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS)) / 255.
            except:
                print(video, self.num_frames[video], target)

            mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][target])
            temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
            if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                print(np.unique(temp))
            temp_mask = np.zeros(temp.shape)
            if self.SI:
                temp_mask[temp == object_label] = 1
            else:
                temp_mask[temp > 0] = 1
            N_masks[f] = (temp_mask != 0).astype(np.uint8)

        need_au = self.phase == 'train' and (np.random.randint(5) >= 1)
        if need_au:
            seed = np.random.randint(99999)
            # print('seed:',seed)
            input_frames = (N_frames * 255).astype(np.uint8)
            for t in range(len(N_frames)):
                # img_au, mask_au = self.aug(image=input_frames[t,np.newaxis,:,:,:].astype(np.uint8),mask=N_masks[t,np.newaxis,:,:,np.newaxis],seed=45229)
                img_au, mask_au = self.aug(image=input_frames[t, np.newaxis, :, :, :].astype(np.uint8),
                                           mask=N_masks[t, np.newaxis, :, :, np.newaxis], seed=seed)
                N_frames[t] = img_au[0] / 255.
                N_masks[t] = mask_au[0, :, :, 0]
        # print('need_au',need_au,'N_frames',N_frames.shape,'N_masks',N_masks.shape,np.unique(N_masks))
        Fs = torch.from_numpy(N_frames).permute(3, 0, 1, 2).float()
        Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()
        # print("*********", torch.min(Ms))
        # if len(Ms[0]) != 7:
        #     print(video,Ms.shape)
        # # print('Fs',Fs.shape,'Ms',Ms.shape)
        sample = {
            'Fs': Fs, 'Ms': Ms, 'num_objects': num_objects, 'info': info
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

        combo_aug = seq_all(images=combo)
        # print('combo_au: ',combo_aug.shape)
        images_aug = combo_aug[:, :, :, :3]
        masks_aug = combo_aug[:, :, :, 3:]
        images_aug = seq_f(images=images_aug)

        return images_aug, masks_aug


if __name__ == "__main__":
    DAVIS_ROOT = '/cfs/mazhongke/databases/DAVIS2017/'
    YOUTUBE_ROOT = '/cfs/dataset/youtube_complete/'
    clip_size = 4
    from torch.utils.data import DataLoader

    dataset = DAVIS(DAVIS_ROOT, phase='val', imset='2016/val.txt', resolution='480p', separate_instance=False,
                    only_single=True, target_size=(864, 480), clip_size=5)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    # dataset = DAVIS(YOUTUBE_ROOT, phase='train', imset='train.txt', resolution='480p', separate_instance=True, only_single=False, target_size=(864, 480), clip_size=5)
    # data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    from tqdm import tqdm

    for i, batch in enumerate(tqdm(data_loader)):
        all_F, all_M, info = batch['Fs'], batch['Ms'], batch['info']
        print('all_F', all_F.shape, 'all_M', all_M.shape, 'info', info)
        # image1, image2, label, pos, neg = batch['image'],batch['image2'],batch['label'],batch['pos'],batch['neg']

    # train_dataset = DAVIS(YOUTUBE_ROOT, resolution='480p', imset='train.txt', single_object=False, clip_size=clip_size)
    # # train_dataset = DAVIS(DAVIS_ROOT, resolution='480p', imset='2017/train.txt', single_object=False, clip_size=clip_size)
    # train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # # val_dataset = DAVIS(YOUTUBE_ROOT, resolution='480p', imset='simple_test.txt', single_object=False, clip_size=clip_size)
    # val_dataset = DAVIS(DAVIS_ROOT, resolution='480p', imset='2016/val.txt', single_object=True, clip_size='val')
    # val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
