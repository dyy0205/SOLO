from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os, glob, time

import datetime
import os
import glob
from PIL import Image
import cv2
import numpy as np
import zipfile
from functools import wraps

### My libs
from STM.models.model import STM
from STM.dataset import TIANCHI


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[finished {func_name} in {time:.2f}s]'.format(func_name=function.__name__, time=t1 - t0))
        return result

    return function_timer


@fn_timer
def mask_inference(data_dir, config, ckpt, out_dir):
    # build the model from a config file and a checkpoint file
    print('Generating first frame mask...')
    model = init_detector(config, ckpt, device='cuda:0')

    # test a single image
    imgs = glob.glob(os.path.join(data_dir, 'JPEGImages/*/*.jpg'))
    print('Total images: {}'.format(len(imgs)))
    generate_imagesets(data_dir, imgs)
    for img in imgs:
        # img = '/workspace/solo/test/00001.jpg'
        result, cost_time = inference_detector(model, img)
        result = filter_result(result, max_num=3)
        save_mask(img, result, MASK_THR, out_dir)


def generate_imagesets(data_dir, imgs):
    print('Generating image set file...')
    videos = [img.split('/')[-2] for img in imgs]
    videos = np.unique(videos)
    with open(os.path.join(data_dir, 'ImageSets/test.txt'), 'w') as f:
        for v in videos:
            f.write(v)
            f.write('\n')


def filter_result(result, index=0, max_num=3):
    assert isinstance(result, list)
    rr = []
    for r in result:
        if r is None:
            return rr
        mask, cate, score = r
        idxs = cate == index
        score = score[idxs]
        mask = mask[idxs, :, :]
        cate = cate[idxs]
        if len(score) > max_num:
            score = score[:max_num]
            mask = mask[:max_num, :, :]
            cate = cate[:max_num]
        rr.append((mask, cate, score))
    return rr


def save_mask(img, result, score_thr, out_dir):
    img_name = img.split('/')[-1]
    video_name = img.split('/')[-2]
    save_path = os.path.join(out_dir, video_name, img_name.replace('jpg', 'png'))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if result == []:
        return 0
    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score >= score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]

    if num_mask == 0:
        return 0

    color_masks = list(range(1, 256))
    _, h, w = seg_label.shape
    img_show = np.zeros((h, w)).astype(np.uint8)
    for idx in range(num_mask):
        cur_mask = seg_label[idx, :, :]
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = color_mask

    img_s = Image.fromarray(img_show)
    img_s.putpalette(PALETTE)
    img_s.save(save_path)


def process_data_root(data_root, img_root):
    print('Processing data root...')
    for i in ('ImageSets', 'Annotations'):
        if not os.path.exists(os.path.join(data_root, i)):
            os.makedirs(os.path.join(data_root, i))
    if os.path.exists(os.path.join(data_root, 'JPEGImages')):
        os.unlink(os.path.join(data_root, 'JPEGImages'))
    os.symlink(img_root, os.path.join(data_root, 'JPEGImages'))


def check_data_root(data_root):
    assert os.path.exists(os.path.join(data_root, 'JPEGImages'))
    assert os.path.exists(os.path.join(data_root, 'ImageSets'))
    assert os.path.exists(os.path.join(data_root, 'Annotations'))


def analyse_images(data_root):
    imgs = glob.glob(os.path.join(data_root, 'JPEGImages/*/*.jpg'))
    print('Total test images: {}'.format(len(imgs)))
    videos = []
    for file in os.listdir(os.path.join(data_root, 'JPEGImages')):
        if os.path.isdir(os.path.join(data_root, 'JPEGImages', file)):
            videos.append(file)
    print('Total test videos: {}'.format(len(videos)))

    v_frames = {}
    for video in videos:
        frames = os.listdir(os.path.join(data_root, 'JPEGImages', video))
        frames = [frame.split('.')[0] for frame in frames]
        frames.sort(key=int)
        v_frames.setdefault(video, frames)

    return v_frames


if __name__ == '__main__':
    DATA_ROOT = '/workspace/solo/code/user_data/data'
    IMG_ROOT = '/workspace/dataset/VOS/mini_fusai/JPEGImages/'
    MODEL_PATH = '/workspace/solo/code/user_data/model_data/model.pth'
    SAVE_PATH = '/solo/code/user_data/'
    TMP_PATH = '/workspace/solo/code/user_data/tmp_data'
    MERGE_PATH = '/workspace/solo/code/user_data/merge_data'
    MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')

    if IMG_ROOT is not None or IMG_ROOT != '':
        process_data_root(DATA_ROOT, IMG_ROOT)
    check_data_root(DATA_ROOT)

    CONFIG_FILE = r'/workspace/solo/code/cfg/aug_solov2_r101_imgaug.py'
    CKPT_FILE = r'/workspace/solo/code/user_data/model_data/solov2_r101_ssim.pth'

    TEMPLATE_MASK = r'/workspace/solo/code/user_data/template_data/00001.png'
    PALETTE = Image.open(TEMPLATE_MASK).getpalette()
    VIDEO_FRAMES = analyse_images(DATA_ROOT)

    MASK_THR = 0.2

    mask_inference(DATA_ROOT, CONFIG_FILE, CKPT_FILE, MASK_PATH)
