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


class TIANCHI(data.Dataset):
    '''
    Dataset for DAVIS
    '''

    def __init__(self, root, imset='2017/train.txt', single_object=True, target_size=(864, 480), test_aug=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.target_size = target_size
        self.test_aug = test_aug

        self.single_object = single_object
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.shape = {}
        self.frame_list = {}
        self.mask_list = {}
        self.start_index = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                temp_img = os.listdir(os.path.join(self.image_dir, _video))
                temp_img.sort()

                temp_mask = os.listdir(os.path.join(self.mask_dir, _video))
                temp_mask.sort()
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, temp_mask[0])).convert("P"))

                if self.single_object:
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
                            self.shape[_video + '_{}'.format(i)] = np.shape(_mask)
                            self.start_index[_video + '_{}'.format(i)] = temp_img.index(
                                temp_mask[0].replace('png', 'jpg'))
                else:
                    self.videos.append(_video)
                    self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) + len(
                        glob.glob(os.path.join(self.image_dir, _video, '*.png')))
                    self.mask_list[_video] = temp_mask
                    self.frame_list[_video] = temp_img
                    # self.num_objects[_video] = np.max(_mask)
                    self.shape[_video] = np.shape(_mask)

        self.K = 9

    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K + 1, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K + 1):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K + 1, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        start_index = self.start_index.get(video)
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['ori_shape'] = self.shape[video]
        info['start_index'] = start_index

        if start_index == 0:
            info['mode'] = 0
            video_true_name, object_label = video.split('_')
            object_label = int(object_label)

            N_frames = np.empty((self.num_frames[video],) + self.target_size[::-1] + (3,), dtype=np.float32)
            N_masks = np.empty((1,) + self.target_size[::-1], dtype=np.uint8)
            for f in range(self.num_frames[video]):
                img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][f])
                frame_image = np.array(
                    Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS))
                if self.test_aug:
                    frame_image = self.test_augmentation(frame_image)
                N_frames[f] = frame_image / 255.

            mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][0])
            temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
            if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                print(np.unique(temp))
            temp_mask = np.zeros(temp.shape)
            temp_mask[temp == object_label] = 1
            N_masks[0] = (temp_mask != 0).astype(np.uint8)

            Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
            Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()
            return Fs, Ms, info

        elif start_index == self.num_frames[video] - 1:
            info['mode'] = 1
            video_true_name, object_label = video.split('_')
            object_label = int(object_label)

            N_frames = np.empty((self.num_frames[video],) + self.target_size[::-1] + (3,), dtype=np.float32)
            N_masks = np.empty((1,) + self.target_size[::-1], dtype=np.uint8)
            for f in range(self.num_frames[video]):
                img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][::-1][f])
                frame_image = np.array(
                    Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS))
                if self.test_aug:
                    frame_image = self.test_augmentation(frame_image)
                N_frames[f] = frame_image / 255.

            mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][0])
            temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
            if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                print(np.unique(temp))
            temp_mask = np.zeros(temp.shape)
            temp_mask[temp == object_label] = 1
            N_masks[0] = (temp_mask != 0).astype(np.uint8)

            Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
            Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()
            return Fs, Ms, info

        else:
            info['mode'] = 2
            video_true_name, object_label = video.split('_')
            object_label = int(object_label)
            prev_frames = self.frame_list[video][:start_index + 1]

            N_frames = np.empty((len(prev_frames),) + self.target_size[::-1] + (3,), dtype=np.float32)
            N_masks = np.empty((1,) + self.target_size[::-1], dtype=np.uint8)
            for f in range(len(prev_frames)):
                img_file = os.path.join(self.image_dir, video_true_name, prev_frames[::-1][f])
                frame_image = np.array(
                    Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS))
                if self.test_aug:
                    frame_image = self.test_augmentation(frame_image)
                N_frames[f] = frame_image / 255.

            mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][0])
            temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
            if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                print(np.unique(temp))
            temp_mask = np.zeros(temp.shape)
            temp_mask[temp == object_label] = 1
            N_masks[0] = (temp_mask != 0).astype(np.uint8)

            Fs_p = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
            Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()

            rear_frames = self.frame_list[video][start_index:]
            N_frames = np.empty((len(rear_frames),) + self.target_size[::-1] + (3,), dtype=np.float32)
            for f in range(len(rear_frames)):
                img_file = os.path.join(self.image_dir, video_true_name, rear_frames[f])
                frame_image = np.array(
                    Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS))
                if self.test_aug:
                    frame_image = self.test_augmentation(frame_image)
                N_frames[f] = frame_image / 255.

            Fs_r = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()

            return Fs_p, Fs_r, Ms, info



        # if self.single_object:
        #     video_true_name, object_label = video.split('_')
        #     object_label = int(object_label)
        # else:
        #     video_true_name = video
        #     object_label = 1
        #
        # N_frames = np.empty((self.num_frames[video],) + self.target_size[::-1] + (3,), dtype=np.float32)
        # N_masks = np.empty((1,) + self.target_size[::-1], dtype=np.uint8)
        # for f in range(self.num_frames[video]):
        #     img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][f])
        #     frame_image = np.array(
        #         Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS))
        #     if self.test_aug:
        #         frame_image = self.test_augmentation(frame_image)
        #     N_frames[f] = frame_image / 255.
        #
        # mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][0])
        # temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
        # if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        #     print(np.unique(temp))
        # temp_mask = np.zeros(temp.shape)
        # if self.single_object:
        #     temp_mask[temp == object_label] = 1
        # else:
        #     temp_mask[temp > 0] = 1
        # N_masks[0] = (temp_mask != 0).astype(np.uint8)
        #
        # Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        # if self.single_object:
        #     Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()
        #     return Fs, Ms, info
        # else:
        #     Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        #     return Fs, Ms, info

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