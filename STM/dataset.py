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



class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                # print('a')
                N_masks[f] = 255
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info


class TIANCHI(data.Dataset):
    '''
    Dataset for DAVIS
    '''

    def __init__(self, root, imset='2017/train.txt', single_object=False, target_size=(832, 448)):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.target_size = target_size

        self.single_object = single_object
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

                temp_mask = os.listdir(os.path.join(self.mask_dir, _video))
                temp_mask.sort()
                _mask = np.array(
                    Image.open(os.path.join(self.mask_dir, _video, temp_mask[0])).convert("P").resize(self.target_size,
                                                                                                      Image.NEAREST))

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
                            # self.num_objects[_video + '_{}'.format(i)] = 1
                            self.shape[_video + '_{}'.format(i)] = np.shape(_mask)
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
        M = np.zeros((self.K+1, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K+1):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K+1, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['ori_shape'] = self.shape[video]

        if self.single_object:
            video_true_name, object_label = video.split('_')
            object_label = int(object_label)
        else:
            video_true_name = video
            object_label = 1

        N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
        N_masks = np.empty((1,) + self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video_true_name, self.frame_list[video][f])
            N_frames[f] = np.array(
                Image.open(img_file).convert('RGB').resize(self.target_size, Image.ANTIALIAS)) / 255.

        mask_file = os.path.join(self.mask_dir, video_true_name, self.mask_list[video][0])
        temp = np.array(Image.open(mask_file).convert('P').resize(self.target_size, Image.NEAREST), dtype=np.uint8)
        if np.unique(temp).any() not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            print(np.unique(temp))
        temp_mask = np.zeros(temp.shape)
        if self.single_object:
            temp_mask[temp == object_label] = 1
        else:
            temp_mask[temp > 0] = 1
        N_masks[0] = (temp_mask != 0).astype(np.uint8)

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            Ms = torch.from_numpy(N_masks[:, :, :, np.newaxis]).permute(3, 0, 1, 2).long()
            return Fs, Ms, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            return Fs, Ms, info

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

if __name__ == '__main__':
    pass
