import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


class TIANCHI_FUSAI(data.Dataset):
    '''
    Dataset for DAVIS
    '''

    def __init__(self, root, imset='2017/train.txt', target_size=(864, 480), test_aug=False):
        self.root = root
        # self.mask_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.target_size = target_size
        self.test_aug = test_aug

        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.shape = {}
        self.frame_list = {}
        self.mask_list = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                temp_img = os.listdir(os.path.join(self.image_dir, _video))
                temp_img.sort()
                _img = np.array(Image.open(os.path.join(self.image_dir, _video, temp_img[0])).convert("P"))

                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                self.frame_list[_video] = temp_img
                # self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_img)[:2]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['ori_shape'] = self.shape[video]
        info['target_shape'] = self.target_size

        video_true_name = video

        N_frames = np.empty((self.num_frames[video],) + self.target_size[::-1] + (3,), dtype=np.float32)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][f])
            frame_image = np.array(
                Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS))
            if self.test_aug:
                frame_image = self.test_augmentation(frame_image)
            N_frames[f] = frame_image / 255.

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        return Fs, info


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
            iaa.contrast.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=False)

        combo_aug = seq_all(images=combo)
        # print('combo_au: ',combo_aug.shape)
        images_aug = combo_aug[:, :, :, :3]
        masks_aug = combo_aug[:, :, :, 3:]
        images_aug = seq_f(images=images_aug)

        return images_aug, masks_aug

    def test_augmentation(self, src):
        dst = np.zeros_like(src)
        for i in range(3):
            channel = src[:, :, i]
            eh = cv2.equalizeHist(channel)
            dst[:, :, i] = eh
        return dst


if __name__ == '__main__':
    pass
