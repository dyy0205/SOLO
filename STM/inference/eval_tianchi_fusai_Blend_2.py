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
import tqdm
import itertools

### My libs
from STM.models.model_fusai import STM
from STM.dataset import TIANCHI
from STM.tools.generate_videos import generate_videos
from STM.dataloader.fusai_dataset import TIANCHI_FUSAI
from STM.tools.process_dir import process_tianchi_dir
from STM.tools.tianchi_eval_vos import calculate_videos_miou

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
    instance_idx = 1
    b, c, T, h, w = Fs.shape
    results = []

    if np.all([len(i[0]) == 0 for i in seg_resuls]):
        print('No segmentation result of solo!')
        pred = torch.zeros((b, 1, T, h, w)).float().cuda()
        return [(pred, 1)]

    while True:
        if np.all([len(i[0]) == 0 for i in seg_resuls]):
            print('Run video over!')
            break
        seg_result_idx = [i[3] for i in seg_resuls]
        start_frame_idx = np.argmax([max(i[2]) if i[2] != [] else 0 for i in seg_resuls])
        start_frame = seg_result_idx[start_frame_idx]
        num_ins_in_frame = len(seg_resuls[start_frame_idx][0])
        print('Running video on frame:{}, ins in this frame:{}'.format(start_frame, num_ins_in_frame))
        for p in range(num_ins_in_frame):
            if instance_idx > MAX_NUM:
                print('Max instance number!')
                break
            start_mask = seg_resuls[start_frame_idx][0][p].astype(np.uint8)
            # start_mask = cv2.resize(start_mask, (w, h))
            start_mask = torch.from_numpy(start_mask).cuda()

            Es = torch.zeros((b, 1, T, h, w)).float().cuda()
            Es[:, :, start_frame] = start_mask
            to_memorize = [int(i) for i in np.arange(start_frame, num_frames, step=Mem_every)]
            for t in range(start_frame + 1, num_frames):  # frames after
                # memorize
                pre_key, pre_value = model([Fs[:, :, t - 1], Es[:, :, t - 1]])
                pre_key = pre_key.unsqueeze(2)
                pre_value = pre_value.unsqueeze(2)

                if t - 1 == start_frame:  # the first frame
                    this_keys_m, this_values_m = pre_key, pre_value
                else:  # other frame
                    this_keys_m = torch.cat([keys, pre_key], dim=2)
                    this_values_m = torch.cat([values, pre_value], dim=2)

                # segment
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, Es[:, :, t - 1].detach()])  # B 2 h w
                em = F.softmax(logits, dim=1)[:, 1]  # B h w
                Es[:, 0, t] = em

                # check solo result
                pred = torch.round(em.float())
                if t in seg_result_idx:
                    idx = seg_result_idx.index(t)
                    this_frame_results = seg_resuls[idx]
                    masks = this_frame_results[0]
                    ious = []
                    for mask in masks:
                        mask = mask.astype(np.uint8)
                        mask = torch.from_numpy(mask)
                        iou = get_video_mIoU(pred, mask)
                        ious.append(iou)
                    if ious != []:
                        ious = np.array(ious)
                        reserve = list(range(len(ious)))
                        if sum(ious >= IOU1) >= 1:
                            same_idx = np.argmax(ious)
                            Es[:, 0, t] = torch.from_numpy(masks[same_idx]).cuda()
                            reserve.remove(same_idx)

                        for i, iou in enumerate(ious):
                            if iou >= IOU2 and iou < IOU1:
                                reserve.remove(i)

                        reserve_result = []
                        for n in range(3):
                            reserve_result.append([this_frame_results[n][i] for i in reserve])
                        reserve_result.append(this_frame_results[3])
                        seg_resuls[idx] = reserve_result

                # update key and value
                if t - 1 in to_memorize:
                    keys, values = this_keys_m, this_values_m

            to_memorize = [start_frame - int(i) for i in np.arange(0, start_frame + 1, step=Mem_every)]
            for t in list(range(0, start_frame))[::-1]:  # frames before
                # memorize
                pre_key, pre_value = model([Fs[:, :, t + 1], Es[:, :, t + 1]])
                pre_key = pre_key.unsqueeze(2)
                pre_value = pre_value.unsqueeze(2)

                if t + 1 == start_frame:  # the first frame
                    this_keys_m, this_values_m = pre_key, pre_value
                else:  # other frame
                    this_keys_m = torch.cat([keys, pre_key], dim=2)
                    this_values_m = torch.cat([values, pre_value], dim=2)

                # segment
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, Es[:, :, t + 1].detach()])  # B 2 h w
                em = F.softmax(logits, dim=1)[:, 1]  # B h w
                Es[:, 0, t] = em

                # check solo result
                pred = torch.round(em.float())
                if t in seg_result_idx:
                    idx = seg_result_idx.index(t)
                    this_frame_results = seg_resuls[idx]
                    masks = this_frame_results[0]
                    ious = []
                    for mask in masks:
                        mask = mask.astype(np.uint8)
                        mask = torch.from_numpy(mask)
                        iou = get_video_mIoU(pred, mask)
                        ious.append(iou)
                    if ious != []:
                        ious = np.array(ious)
                        reserve = list(range(len(ious)))
                        if sum(ious >= IOU1) >= 1:
                            same_idx = np.argmax(ious)
                            Es[:, 0, t] = torch.from_numpy(masks[same_idx]).cuda()
                            reserve.remove(same_idx)

                        for i, iou in enumerate(ious):
                            if iou >= IOU2 and iou < IOU1:
                                reserve.remove(i)

                        reserve_result = []
                        for n in range(3):
                            reserve_result.append([this_frame_results[n][i] for i in reserve])
                        reserve_result.append(this_frame_results[3])
                        seg_resuls[idx] = reserve_result

                # update key and value
                if t + 1 in to_memorize:
                    keys, values = this_keys_m, this_values_m

            # for j in range(3):
            #     seg_resuls[start_frame_idx][j].pop(0)

            pred = torch.round(Es.float())
            results.append((pred, instance_idx))

            instance_idx += 1

        seg_resuls.pop(start_frame_idx)

    return results


@fn_timer
def vos_inference():
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on TIANCHI...')

    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testset = TIANCHI_FUSAI(DATA_ROOT, imset='test.txt', target_size=TARGET_SHAPE)
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
    progressbar = tqdm.tqdm(Testloader)

    for V in progressbar:
        Fs, info = V
        seq_name = info['name'][0]
        ori_shape = info['ori_shape']
        target_shape = info['target_shape']
        target_shape = (target_shape[0].cpu().numpy()[0], target_shape[1].cpu().numpy()[0])
        num_frames = info['num_frames'][0].item()
        if '_' in seq_name:
            video_name = seq_name.split('_')[0]
        else:
            video_name = seq_name
        seg_results = mask_inference(video_name, target_shape)

        frame_list = VIDEO_FRAMES[video_name]

        print('[{}]: num_frames: {}'.format(seq_name, num_frames))

        results = Run_video(model, Fs, seg_results, num_frames, Mem_every=5)

        for result in results:
            pred, instance = result
            test_path = os.path.join(TMP_PATH, seq_name + '_{}'.format(instance))
            if not os.path.exists(test_path):
                os.makedirs(test_path)

            for f in range(num_frames):
                img_E = Image.fromarray(pred[0, 0, f].cpu().numpy().astype(np.uint8))
                img_E.putpalette(PALETTE)
                img_E = img_E.resize(ori_shape[::-1])
                img_E.save(os.path.join(test_path, '{}.png'.format(frame_list[f])))


def get_video_mIoU(predn, all_Mn):  # [c,t,h,w]
    pred = predn.squeeze().cpu().data.numpy()
    # np.save('blackswan.npy', pred)
    gt = all_Mn.squeeze().cpu().data.numpy()  # [t,h,w]
    agg = pred + gt
    i = float(np.sum(agg == 2))
    u = float(np.sum(agg > 0))
    return i / (u + 1e-6)


def get_img_miou(img1, img2):
    agg = img1 + img2
    i = float(np.sum(agg == 2))
    u = float(np.sum(agg > 0)) + 1e-6
    return i, u


@fn_timer
def mask_inference(video_name, mask_shape):
    # build the model from a config file and a checkpoint file
    print('Generating frame mask...')
    model = init_detector(CONFIG_FILE, CKPT_FILE, device='cuda:0')

    # test a single image
    frames = VIDEO_FRAMES.get(video_name)
    fi = []
    i = 0
    interval = SOLO_INTERVAL
    while True:
        fi.append(i)
        i += interval
        if i >= len(frames) - 1:
            fi.append(len(frames) - 1)
            break

    results = []
    for f in fi:
        img = os.path.join(DATA_ROOT, 'JPEGImages/{}/{}.jpg'.format(video_name, frames[f]))
        result, cost_time = inference_detector(model, img)
        result = filter_result(result, max_num=MAX_NUM)
        if result is None:
            continue
        result = process_solo_result(result, mask_shape)
        result = list(result) + [f]
        results.append(result)

    results = filter_score(results)

    # visualize solo mask
    for result in results:
        mask_result = result[:3]
        frame = result[3]
        img = os.path.join(DATA_ROOT, 'JPEGImages/{}/{}.jpg'.format(video_name, frames[frame]))
        save_mask(img, mask_result, 0, MASK_PATH)
    return results


def process_solo_result(result, mask_shape):
    num = len(result[1])
    result_ = []
    final = []
    for i in range(num):
        mask = result[0][i].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, mask_shape)
        result_.append(mask)
    final.append(np.array(result_))
    final.append(result[1].cpu().numpy())
    final.append(result[2].cpu().numpy())
    return final  # [array, array, array]


def generate_imagesets():
    print('Generating image set file...')
    videos = VIDEO_FRAMES.keys()
    videos = list(videos)
    videos = np.unique(videos)
    with open(os.path.join(DATA_ROOT, 'ImageSets/test.txt'), 'w') as f:
        for v in videos:
            f.write(v)
            f.write('\n')


def filter_score(results):  # list(array, array, array, int)
    filtered = []
    num = 0
    for result in results:
        idx = result[2] >= SCORE_THR
        num += np.sum(idx)
        if np.any(idx):
            filtered.append([list(result[i][idx]) for i in range(3)] + [result[3]])

    if num == 0:
        filtered = []
        # no mask score larger than threshold
        scores = np.array([result[2] for result in results])
        scores = np.concatenate(scores)
        score_thr = np.max(scores) * 0.8
        for result in results:
            idx = result[2] >= score_thr
            if np.any(idx):
                filtered.append([list(result[i][idx]) for i in range(3)] + [result[3]])

    return filtered


def filter_result(result, index=0, max_num=8):
    assert isinstance(result, list)
    result = result[0]
    if result is None:
        return None
    mask, cate, score = result
    idxs = (cate == index)
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


# TODO: another blend result method.
@fn_timer
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

        match_lst = get_tmp_match_lst(tmp_dir, ins, iou_thr=BLEND_IOU_THR)

        match_dict = {}
        if len(match_lst) != 0:
            for m in match_lst:
                ins1, ins2 = list(map(int, m.split('_')))
                match_dict.setdefault(ins2, ins1)
        flag = True
        while flag:
            flag = False
            for k, v in match_dict.items():
                if v in match_dict.keys():
                    match_dict[k] = match_dict[v]
                    flag = True
        print('Match tmp instance: {}'.format(match_dict))


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
                    if j in match_dict.keys():
                        j = match_dict[j]
                    mask[(temp == 1) & ((mask > j) | (mask == 0))] = j
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


def save_mask(img, result, score_thr, out_dir):
    img_name = img.split('/')[-1]
    video_name = img.split('/')[-2]
    save_path = os.path.join(out_dir, video_name, img_name.replace('jpg', 'png'))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if result != []:
        cur_result = result
        seg_label = np.array(cur_result[0])
        # seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = np.array(cur_result[1])
        # cate_label = cate_label.cpu().numpy()
        score = np.array(cur_result[2])

        vis_inds = score >= score_thr
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]

        if num_mask == 0:
            print(img)
            img_ = cv2.imread(img)
            h, w, c = img_.shape
            img_show = np.zeros((h, w)).astype(np.uint8)
            img_s = Image.fromarray(img_show)
            img_s.putpalette(PALETTE)
            img_s.save(save_path)

        # color_masks = list(range(1, 256))
        color_mask = 1
        _, h, w = seg_label.shape
        img_show = np.zeros((h, w)).astype(np.uint8)
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            # cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                print(img)
                img_ = cv2.imread(img)
                h, w, c = img_.shape
                img_show = np.zeros((h, w)).astype(np.uint8)
                img_s = Image.fromarray(img_show)
                img_s.putpalette(PALETTE)
                img_s.save(save_path)
            # color_mask = color_masks[idx]
            cur_mask_bool = cur_mask.astype(np.bool) & (img_show == 0)
            if not np.any(cur_mask_bool):
                continue
            img_show[cur_mask_bool] = color_mask
            color_mask += 1

        img_s = Image.fromarray(img_show)
        img_s.putpalette(PALETTE)
        img_s.save(save_path)
    else:
        print(img)
        img_ = cv2.imread(img)
        h, w, c = img_.shape
        img_show = np.zeros((h, w)).astype(np.uint8)
        img_s = Image.fromarray(img_show)
        img_s.putpalette(PALETTE)
        img_s.save(save_path)


def get_tmp_match_lst(tmp_dir, ins_lst, iou_thr=0.5):
    frame_lst = os.listdir(os.path.join(tmp_dir, ins_lst[0]))
    num_frames = len(frame_lst)
    num_ins = len(ins_lst)
    i_dict = {}
    u_dict = {}
    iou_dict = {}
    for f in range(num_frames):
        imgs = [np.array(Image.open(os.path.join(tmp_dir, ins_lst[i], frame_lst[f])).convert('P')) for i in
                range(num_ins)]
        non_empty_lst = []
        for i in range(num_ins):
            if np.any(imgs[i]):
                non_empty_lst.append(i)
        if len(non_empty_lst) >= 2:
            for sample in itertools.combinations(non_empty_lst, 2):
                id1, id2 = sample
                img1 = imgs[id1]
                img2 = imgs[id2]
                i, u = get_img_miou(img1, img2)
                key = '{}_{}'.format(min(id1 + 1, id2 + 1), max(id1 + 1, id2 + 1))
                i_dict.setdefault(key, 0)
                u_dict.setdefault(key, 0)
                iou_dict.setdefault(key, 0)
                i_dict[key] += i
                u_dict[key] += u
    for k in iou_dict.keys():
        iou_dict[k] = i_dict[k] / u_dict[k]

    match_lst = []
    for k, v in iou_dict.items():
        if v >= iou_thr:
            match_lst.append(k)

    return match_lst


if __name__ == '__main__':
    mode = 'offline'
    if mode == 'online':
        DATA_ROOT = '/workspace/user_data/data'
        IMG_ROOT = '/tcdata'
        MODEL_PATH = '/workspace/user_data/model_data/dyy_ckpt_124e.pth'
        SAVE_PATH = '/workspace'
        TMP_PATH = '/workspace/user_data/tmp_data'
        MERGE_PATH = '/workspace/user_data/merge_data'
        MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')
        CONFIG_FILE = r'/workspace/cfg/aug_solov2_r101.py'
        CKPT_FILE = r'/workspace/user_data/model_data/solov2_9cls.pth'
        TEMPLATE_MASK = r'/workspace/user_data/template_data/00001.png'
    else:
        DATA_ROOT = '/workspace/solo/code/user_data/data'
        IMG_ROOT = '/workspace/dataset/VOS/mini_fusai/JPEGImages/'
        MODEL_PATH = '/workspace/solo/code/user_data/model_data/dyy_ckpt_124e.pth'
        SAVE_PATH = '/workspace/solo/code/user_data/'
        TMP_PATH = '/workspace/solo/code/user_data/tmp_data'
        MERGE_PATH = '/workspace/solo/code/user_data/merge_data'
        MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')
        CONFIG_FILE = r'/workspace/solo/code/cfg/aug_solov2_r101.py'
        CKPT_FILE = r'/workspace/solo/code/user_data/model_data/solov2_9cls.pth'
        TEMPLATE_MASK = r'/workspace/solo/code/user_data/template_data/00001.png'
        VIDEO_PATH = '/workspace/solo/code/user_data/video_data'
        GT_PATH = r'/workspace/dataset/VOS/fusai_train/Annotations/'

        process_tianchi_dir(SAVE_PATH)

    if IMG_ROOT is not None or IMG_ROOT != '':
        process_data_root(DATA_ROOT, IMG_ROOT)
    # check_data_root(DATA_ROOT)
    PALETTE = Image.open(TEMPLATE_MASK).getpalette()
    VIDEO_FRAMES = analyse_images(DATA_ROOT)

    TARGET_SHAPE = (1008, 560)
    SCORE_THR = 0.5
    SOLO_INTERVAL = 2
    MAX_NUM = 8
    IOU1 = 0.5
    IOU2 = 0.1
    BLEND_IOU_THR = 0.5

    generate_imagesets()
    vos_inference()
    blend_results(TMP_PATH, MERGE_PATH, DATA_ROOT)
    # zip_result(MERGE_PATH, SAVE_PATH)

    if mode != 'online':
        generate_videos(DATA_ROOT, MERGE_PATH, VIDEO_PATH)
        miou, num = calculate_videos_miou(MERGE_PATH, GT_PATH)
        print('offline evaluation miou: {}, {} videos counted'.format(miou, num))