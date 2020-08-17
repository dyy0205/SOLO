from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
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
import random
import numpy as np

### My libs
from STM.models.model_fusai import STM
from STM.dataset import TIANCHI
from STM.tools.generate_videos import generate_videos
from STM.dataloader.fusai_dataset import TIANCHI_FUSAI

torch.set_grad_enabled(False)  # Volatile


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[finished {func_name} in {time:.2f}s]'.format(func_name=function.__name__, time=t1 - t0))
        return result

    return function_timer


def Run_video(model, Fs, seg_resuls, num_frames, Mem_every=None, Mem_number=None):
    # print('name:', name)
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    b, c, t, h, w = Fs.shape
    Es = torch.zeros((b, 1, t, h, w)).float().cuda()  # [1,1,50,480,864][b,c,t,h,w]
    Es[:, :, 0] = Ms[:, :, 0]

    for t in range(1, num_frames):
        # memorize
        pre_key, pre_value = model([Fs[:, :, t - 1], Es[:, :, t - 1]])
        pre_key = pre_key.unsqueeze(2)
        pre_value = pre_value.unsqueeze(2)

        if t - 1 == 0:  # the first frame
            this_keys_m, this_values_m = pre_key, pre_value
        else:  # other frame
            this_keys_m = torch.cat([keys, pre_key], dim=2)
            this_values_m = torch.cat([values, pre_value], dim=2)

        # segment
        logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m])  # B 2 h w
        em = F.softmax(logits, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

    pred = torch.round(Es.float())

    return pred, Es


@fn_timer
def vos_inference():
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on TIANCHI...')

    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testset = TIANCHI_FUSAI(DATA_ROOT, imset='test.txt')
    print('Total test videos: {}'.format(len(Testset)))
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(STM())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()  # turn-off BN

    print('Loading weights:', MODEL_PATH)
    model_ = torch.load(MODEL_PATH)
    if 'state_dict' in model_.keys():
        state_dict = model_['state_dict']
    else:
        state_dict = model_
    model.load_state_dict(state_dict)

    code_name = 'Tianchi fusai'
    # date = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d%H%M')
    print('Start Testing:', code_name)

    for seq, V in enumerate(Testloader):
        Fs, info = V
        seq_name = info['name'][0]
        ori_shape = info['ori_shape']
        num_frames = info['num_frames'][0].item()
        if '_' in seq_name:
            video_name = seq_name.split('_')[0]
        else:
            video_name = seq_name
        seg_results = mask_inference(video_name)

        frame_list = VIDEO_FRAMES[video_name]

        print('[{}]: num_frames: {}'.format(seq_name, num_frames))

        pred, Es = Run_video(model, Fs, seg_results, num_frames, Mem_every=5)

        # Save results for quantitative eval ######################
        test_path = os.path.join(TMP_PATH, seq_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        for f in range(num_frames):
            img_E = Image.fromarray(pred[0, 0, f].cpu().numpy().astype(np.uint8))
            img_E.putpalette(PALETTE)
            img_E = img_E.resize(ori_shape[::-1])
            img_E.save(os.path.join(test_path, '{}.png'.format(frame_list[f])))


# @fn_timer
# def vos_infer(data_root, model_path, palette):
#     # Model and version
#     MODEL = 'STM'
#     print(MODEL, ': Testing on TIANCHI...')
#
#     if torch.cuda.is_available():
#         print('using Cuda devices, num:', torch.cuda.device_count())
#
#     Testset = TIANCHI(data_root, imset='test.txt', single_object=True)
#     print('Total test videos: {}'.format(len(Testset)))
#     Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
#
#     model = nn.DataParallel(STM())
#     if torch.cuda.is_available():
#         model.cuda()
#     model.eval()  # turn-off BN
#
#     print('Loading weights:', model_path)
#     model_ = torch.load(model_path)
#     if 'state_dict' in model_.keys():
#         state_dict = model_['state_dict']
#     else:
#         state_dict = model_
#     model.load_state_dict(state_dict)
#
#     code_name = 'tianchi'
#     # date = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d%H%M')
#     print('Start Testing:', code_name)
#
#     for seq, V in enumerate(Testloader):
#         if len(V) == 3:
#             Fs, Ms, info = V
#             seq_name = info['name'][0]
#             ori_shape = info['ori_shape']
#             num_frames = info['num_frames'][0].item()
#             mode = info['mode']
#             if '_' in seq_name:
#                 video_name = seq_name.split('_')[0]
#             if mode == 0:
#                 frame_list = VIDEO_FRAMES[video_name]
#             else:
#                 frame_list = VIDEO_FRAMES[video_name][::-1]
#
#             print('[{}]: num_frames: {}'.format(seq_name, num_frames))
#
#             pred, Es = Run_video(model, Fs, Ms, num_frames, Mem_every=5, Mem_number=None, mode='test')
#
#             # Save results for quantitative eval ######################
#             test_path = os.path.join(TMP_PATH, seq_name)
#             if not os.path.exists(test_path):
#                 os.makedirs(test_path)
#             for f in range(num_frames):
#                 img_E = Image.fromarray(pred[0, 0, f].cpu().numpy().astype(np.uint8))
#                 img_E.putpalette(palette)
#                 img_E = img_E.resize(ori_shape[::-1])
#                 img_E.save(os.path.join(test_path, '{}.png'.format(frame_list[f])))
#
#         elif len(V) == 4:
#             print('Start at middle frame!')
#             Fs_p, Fs_r, Ms, info = V
#             seq_name = info['name'][0]
#             ori_shape = info['ori_shape']
#             num_frames = info['num_frames'][0].item()
#             start_index = info['start_index']
#             if '_' in seq_name:
#                 video_name = seq_name.split('_')[0]
#
#             _, _, prev_frame_num, _, _ = Fs_p.shape
#             _, _, rear_frame_num, _, _ = Fs_r.shape
#             prev_frame_list = VIDEO_FRAMES[video_name][:start_index + 1][::-1]
#             rear_frame_list = VIDEO_FRAMES[video_name][start_index:]
#
#             print('[{}]: num_frames: {}'.format(seq_name, num_frames))
#
#             pred, Es = Run_video(model, Fs_p, Ms, prev_frame_num, Mem_every=5, Mem_number=None, mode='test')
#             pred_r, Es_r = Run_video(model, Fs_r, Ms, rear_frame_num, Mem_every=5, Mem_number=None, mode='test')
#
#             # Save results for quantitative eval ######################
#             test_path = os.path.join(TMP_PATH, seq_name)
#             if not os.path.exists(test_path):
#                 os.makedirs(test_path)
#             for f in range(prev_frame_num):
#                 img_E = Image.fromarray(pred[0, 0, f].cpu().numpy().astype(np.uint8))
#                 img_E.putpalette(palette)
#                 img_E = img_E.resize(ori_shape[::-1])
#                 img_E.save(os.path.join(test_path, '{}.png'.format(prev_frame_list[f])))
#             for f in range(rear_frame_num):
#                 img_E = Image.fromarray(pred_r[0, 0, f].cpu().numpy().astype(np.uint8))
#                 img_E.putpalette(palette)
#                 img_E = img_E.resize(ori_shape[::-1])
#                 img_E.save(os.path.join(test_path, '{}.png'.format(rear_frame_list[f])))


@fn_timer
def mask_inference(video_name):
    # build the model from a config file and a checkpoint file
    print('Generating frame mask...')
    model = init_detector(CONFIG_FILE, CKPT_FILE, device='cuda:0')

    # test a single image
    frames = VIDEO_FRAMES.get(video_name)
    imgs = []
    fi = []
    i = 0
    interval = 5
    while True:
        fi.append(i)
        i += interval
        if i >= len(frames) - 5:
            fi.append(len(frames) - 1)
            break

    for f in fi:
        imgs.append(os.path.join(DATA_ROOT, 'JPEGImages/{}/{}.jpg'.format(video_name, frames[f])))
    results = []
    for img in imgs:
        result, cost_time = inference_detector(model, img)
        result = filter_result(result, max_num=MAX_NUM)
        results.append(result)
    return results


def generate_imagesets():
    print('Generating image set file...')
    videos = VIDEO_FRAMES.keys()
    videos = list(videos)
    videos = np.unique(videos)
    with open(os.path.join(DATA_ROOT, 'ImageSets/test.txt'), 'w') as f:
        for v in videos:
            f.write(v)
            f.write('\n')


def filter_result(result, index=0, max_num=3):
    assert isinstance(result, list)
    result = result[0]
    if result is None:
        return None
    mask, cate, score = result
    idxs = cate == index
    if not np.any(idxs.cpu().numpy()):
        return None
    score = score[idxs]
    mask = mask[idxs, :, :]
    cate = cate[idxs]
    if len(score) > max_num:
        score = score[:max_num]
        mask = mask[:max_num, :, :]
        cate = cate[:max_num]
    return (mask, cate, score)


# def save_mask(img, result, score_thr, out_dir):
#     img_name = img.split('/')[-1]
#     video_name = img.split('/')[-2]
#     save_path = os.path.join(out_dir, video_name, img_name.replace('jpg', 'png'))
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#
#     if result != []:
#         cur_result = result
#         seg_label = cur_result[0]
#         seg_label = seg_label.cpu().numpy().astype(np.uint8)
#         cate_label = cur_result[1]
#         cate_label = cate_label.cpu().numpy()
#         score = cur_result[2].cpu().numpy()
#
#         vis_inds = score >= score_thr
#         seg_label = seg_label[vis_inds]
#         num_mask = seg_label.shape[0]
#
#         if num_mask == 0:
#             print(img)
#             img_ = cv2.imread(img)
#             h, w, c = img_.shape
#             img_show = np.zeros((h, w)).astype(np.uint8)
#             img_s = Image.fromarray(img_show)
#             img_s.putpalette(PALETTE)
#             img_s.save(save_path)
#
#         # color_masks = list(range(1, 256))
#         color_mask = 1
#         _, h, w = seg_label.shape
#         img_show = np.zeros((h, w)).astype(np.uint8)
#         for idx in range(num_mask):
#             cur_mask = seg_label[idx, :, :]
#             # cur_mask = (cur_mask > 0.5).astype(np.uint8)
#             if cur_mask.sum() == 0:
#                 print(img)
#                 img_ = cv2.imread(img)
#                 h, w, c = img_.shape
#                 img_show = np.zeros((h, w)).astype(np.uint8)
#                 img_s = Image.fromarray(img_show)
#                 img_s.putpalette(PALETTE)
#                 img_s.save(save_path)
#             # color_mask = color_masks[idx]
#             cur_mask_bool = cur_mask.astype(np.bool) & (img_show == 0)
#             if not np.any(cur_mask_bool):
#                 continue
#             img_show[cur_mask_bool] = color_mask
#             color_mask += 1
#
#         img_s = Image.fromarray(img_show)
#         img_s.putpalette(PALETTE)
#         img_s.save(save_path)
#     else:
#         print(img)
#         img_ = cv2.imread(img)
#         h, w, c = img_.shape
#         img_show = np.zeros((h, w)).astype(np.uint8)
#         img_s = Image.fromarray(img_show)
#         img_s.putpalette(PALETTE)
#         img_s.save(save_path)


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


def blend_results(tmp_dir, merge_dir, data_dir):
    print('Blending results...')
    img_root = os.path.join(data_dir, 'JPEGImages')
    ann_root = os.path.join(data_dir, 'Annotations')
    with open(os.path.join(data_dir, 'ImageSets/test.txt')) as f:
        test = f.readlines()
    test = [img.strip() for img in test]
    print('test videos: ', len(test))

    ins_lst = os.listdir(tmp_dir)
    names = []
    for name in ins_lst:
        name = name.split('_')[0]
        if name not in names:
            names.append(name)
    print(len(names))

    for i, name in enumerate(test):
        num_frames = len(glob.glob(os.path.join(img_root, name, '*.jpg')))
        palette = PALETTE
        ins = [ins for ins in ins_lst if ins.startswith(name)]
        print(i, name, len(ins))

        video_dir = os.path.join(merge_dir, name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if len(ins) == 1:
            # only one instance, no need for blend
            for t in range(num_frames):
                path = os.path.join(tmp_dir, name + '_1', '{}.png'.format(VIDEO_FRAMES[name][t]))
                mask = Image.open(path).convert('P')
                mask.putpalette(palette)
                mask.save(os.path.join(video_dir, '{}.png'.format(VIDEO_FRAMES[name][t])))
        else:
            path = os.path.join(tmp_dir, name + '_{}'.format(1),
                                '{}.png'.format(VIDEO_FRAMES[name][0]))
            temp_ = np.array(Image.open(path).convert('P'), dtype=np.uint8)
            for t in range(num_frames):
                mask = np.zeros_like(temp_)
                for j in list(range(1, len(ins) + 1))[::-1]:
                    path = os.path.join(tmp_dir, name + '_{}'.format(j),
                                        '{}.png'.format(VIDEO_FRAMES[name][t]))
                    temp = np.array(Image.open(path).convert('P'), dtype=np.uint8)
                    mask[(temp == 1)] = j
                # print(len(ins), np.unique(mask))
                mask = Image.fromarray(mask)
                mask.putpalette(palette)
                mask.save(os.path.join(video_dir, '{}.png'.format(VIDEO_FRAMES[name][t])))


def zip_result(result_dir, save_path):
    print('Generating zip file...')
    f = zipfile.ZipFile(os.path.join(save_path, 'result.zip'), 'w', zipfile.ZIP_DEFLATED)
    for dir_path, dir_name, file_names in os.walk(result_dir):
        file_path = dir_path.replace(result_dir, '')
        file_path = file_path and file_path + os.sep or ''
        for file_name in file_names:
            f.write(os.path.join(dir_path, file_name), file_path + file_name)
    f.close()


if __name__ == '__main__':
    mode = 'online'
    if mode == 'online':
        DATA_ROOT = '/workspace/user_data/data'
        IMG_ROOT = '/tcdata'
        MODEL_PATH = '/workspace/user_data/model_data/dyy_i14_ckpt_29e.pth'
        SAVE_PATH = '/workspace'
        TMP_PATH = '/workspace/user_data/tmp_data'
        MERGE_PATH = '/workspace/user_data/merge_data'
        MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')
        CONFIG_FILE = r'/workspace/cfg/aug_solov2_r101.py'
        CKPT_FILE = r'/workspace/user_data/model_data/solov2_9cls.pth'
        TEMPLATE_MASK = r'/workspace/user_data/template_data/00001.png'
    else:
        DATA_ROOT = '/workspace/solo/code/user_data/data'
        IMG_ROOT = '/workspace/dataset/VOS/test_dataset/JPEGImages/'
        MODEL_PATH = '/workspace/solo/code/user_data/model_data/dyy_i14_ckpt_29e.pth'
        SAVE_PATH = '/workspace/solo/code/user_data/'
        TMP_PATH = '/workspace/solo/code/user_data/tmp_data'
        MERGE_PATH = '/workspace/solo/code/user_data/merge_data'
        MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')
        CONFIG_FILE = r'/workspace/solo/code/cfg/aug_solov2_r101.py'
        CKPT_FILE = r'/workspace/solo/code/user_data/model_data/solov2_9cls.pth'
        TEMPLATE_MASK = r'/workspace/solo/code/user_data/template_data/00001.png'
        VIDEO_PATH = '/workspace/solo/code/user_data/video_data'

    if IMG_ROOT is not None or IMG_ROOT != '':
        process_data_root(DATA_ROOT, IMG_ROOT)
    # check_data_root(DATA_ROOT)
    PALETTE = Image.open(TEMPLATE_MASK).getpalette()
    VIDEO_FRAMES = analyse_images(DATA_ROOT)

    MASK_THR = 0.3
    MAX_NUM = 8

    generate_imagesets()
    mask_inference(DATA_ROOT, VIDEO_FRAMES, CONFIG_FILE, CKPT_FILE, MASK_PATH)
    vos_infer(DATA_ROOT, MODEL_PATH, PALETTE)
    blend_results(TMP_PATH, MERGE_PATH, DATA_ROOT)
    zip_result(MERGE_PATH, SAVE_PATH)

    if mode != 'online':
        generate_videos(DATA_ROOT, MERGE_PATH, VIDEO_PATH)
