from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os
import glob
import time

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
import imgaug as ia
import imgaug.augmenters as iaa

### My libs
# from STM.models.model_fusai import STM
from STM.dataset import TIANCHI
from STM.tools.generate_videos import generate_videos
from STM.dataloader.fusai_dataset import TIANCHI_FUSAI
from STM.tools.process_dir import process_tianchi_dir
from STM.tools.tianchi_eval_vos import calculate_videos_miou
from STM.train.train_STM_fusai import *
from STM.tools.utils import _loss, get_video_mIoU


# torch.set_grad_enabled(False)  # Volatile


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[finished {func_name} in {time:.2f}s]'.format(func_name=function.__name__, time=t1 - t0))
        return result

    return function_timer


def Run_video(model, Fs, seg_results, num_frames, Mem_every=None, model_name='standard'):
    seg_result_idx = [i[3] for i in seg_results]

    instance_idx = 1
    b, c, T, h, w = Fs.shape
    results = []

    if np.all([len(i[0]) == 0 for i in seg_results]):
        print('No segmentation result of solo!')
        pred = torch.zeros((b, 1, T, h, w)).float().cuda()
        return [(pred, 1)]

    while True:
        if np.all([len(i[0]) == 0 for i in seg_results]):
            print('Run video over!')
            break
        if instance_idx > MAX_NUM:
            print('Max instance number!')
            break
        start_frame_idx = np.argmax([max(i[2]) if i[2] != [] else 0 for i in seg_results])
        start_frame = seg_result_idx[start_frame_idx]
        start_mask = seg_results[start_frame_idx][0][0].astype(np.uint8)
        # start_mask = cv2.resize(start_mask, (w, h))
        start_mask = torch.from_numpy(start_mask).cuda()

        if model_name in ('enhanced', 'enhanced_motion'):
            Os = torch.zeros((b, c, int(h / 4), int(w / 4)))
            first_frame = Fs[:, :, start_frame]
            first_mask = start_mask.cpu()
            if len(first_mask.shape) == 2:
                first_mask = first_mask.unsqueeze(0).unsqueeze(0)
            elif len(first_mask.shape) == 3:
                first_mask = first_mask.unsqueeze(0)
            first_frame = first_frame * first_mask.repeat(1, 3, 1, 1).type(torch.float)
            for i in range(b):
                mask_ = first_mask[i]
                mask_ = mask_.squeeze(0).cpu().numpy().astype(np.uint8)
                assert np.any(mask_)
                x, y, w_, h_ = cv2.boundingRect(mask_)
                patch = first_frame[i, :, y:(y + h_), x:(x + w_)].cpu().numpy()
                patch = patch.transpose(1, 2, 0)
                patch = cv2.resize(patch, (int(w / 4), int(h / 4)))
                patch = patch.transpose(2, 0, 1)
                patch = torch.from_numpy(patch)
                Os[i, :, :, :] = patch

        if model_name == 'varysize':
            os = []
            first_frame = Fs[:, :, start_frame]
            first_mask = start_mask.cpu()
            if len(first_mask.shape) == 2:
                first_mask = first_mask.unsqueeze(0).unsqueeze(0)
            elif len(first_mask.shape) == 3:
                first_mask = first_mask.unsqueeze(0)
            first_frame = first_frame * first_mask.repeat(1, 3, 1, 1).type(torch.float)
            for i in range(b):
                mask_ = first_mask[i]
                mask_ = mask_.squeeze(0).cpu().numpy().astype(np.uint8)
                assert np.any(mask_)
                x, y, w_, h_ = cv2.boundingRect(mask_)
                patch = first_frame[i, :, y:(y + h_), x:(x + w_)].cpu().numpy()
                Os = torch.zeros((1, c, h_, w_))
                patch = patch.transpose(1, 2, 0)
                patch = patch.transpose(2, 0, 1)
                patch = torch.from_numpy(patch)
                Os[0, :, :, :] = patch
                os.append(Os)

        Es = torch.zeros((b, 1, T, h, w)).float().cuda()
        Es[:, :, start_frame] = start_mask
        # to_memorize = [int(i) for i in np.arange(start_frame, num_frames, step=Mem_every)]
        to_memorize = [start_frame]
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
            if model_name == 'enhanced':
                logits, _, _ = model([Fs[:, :, t], Os, this_keys_m, this_values_m])
            elif model_name == 'motion':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, Es[:, :, t - 1]])
            elif model_name == 'aspp':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, torch.round(Es[:, :, t - 1])])
            elif model_name == 'sp':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, torch.round(Es[:, :, t - 1])])
            elif model_name == 'standard':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m])
            elif model_name == 'enhanced_motion':
                logits, _, _ = model([Fs[:, :, t], Os, this_keys_m, this_values_m, torch.round(Es[:, :, t - 1])])
            elif model_name == 'varysize':
                logits, _, _ = model([Fs[:, :, t], os, this_keys_m, this_values_m])
            else:
                raise NotImplementedError
            em = F.softmax(logits, dim=1)[:, 1]  # B h w
            Es[:, 0, t] = em

            # check solo result
            pred = torch.round(em.float())
            if t in seg_result_idx:
                idx = seg_result_idx.index(t)
                this_frame_results = seg_results[idx]
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
                        mask = torch.from_numpy(masks[same_idx]).cuda()
                        # if get_video_mIoU(mask, torch.round(Es[:, 0, t - 1])) \
                        #     > get_video_mIoU(pred, torch.round(Es[:, 0, t - 1])):
                        Es[:, 0, t] = mask
                        reserve.remove(same_idx)
                        # if abs(to_memorize[-1] - t) >= TO_MEMORY_MIN_INTERVAL:
                        to_memorize.append(t)

                    # for i, iou in enumerate(ious):
                    #     if iou >= IOU2:
                    #         if i in reserve:
                    #             reserve.remove(i)

                    reserve_result = []
                    for n in range(3):
                        reserve_result.append([this_frame_results[n][i] for i in reserve])
                    reserve_result.append(this_frame_results[3])
                    seg_results[idx] = reserve_result

            # update key and value
            if t - 1 in to_memorize:
                keys, values = this_keys_m, this_values_m

        # to_memorize = [start_frame - int(i) for i in np.arange(0, start_frame + 1, step=Mem_every)]
        to_memorize = [start_frame]
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
            if model_name == 'enhanced':
                logits, _, _ = model([Fs[:, :, t], Os, this_keys_m, this_values_m])
            elif model_name == 'motion':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, Es[:, :, t + 1]])
            elif model_name == 'aspp':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, torch.round(Es[:, :, t + 1])])
            elif model_name == 'sp':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m, torch.round(Es[:, :, t + 1])])
            elif model_name == 'standard':
                logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m])
            elif model_name == 'enhanced_motion':
                logits, _, _ = model([Fs[:, :, t], Os, this_keys_m, this_values_m, torch.round(Es[:, :, t + 1])])
            elif model_name == 'varysize':
                logits, _, _ = model([Fs[:, :, t], os, this_keys_m, this_values_m])
            else:
                raise NotImplementedError
            em = F.softmax(logits, dim=1)[:, 1]  # B h w
            Es[:, 0, t] = em

            # check solo result
            pred = torch.round(em.float())
            if t in seg_result_idx:
                idx = seg_result_idx.index(t)
                this_frame_results = seg_results[idx]
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
                        mask = torch.from_numpy(masks[same_idx]).cuda()
                        # if get_video_mIoU(mask, torch.round(Es[:, 0, t + 1])) \
                        #         > get_video_mIoU(pred, torch.round(Es[:, 0, t + 1])):
                        Es[:, 0, t] = mask
                        reserve.remove(same_idx)
                        # if abs(to_memorize[-1] - t) >= TO_MEMORY_MIN_INTERVAL:
                        to_memorize.append(t)

                    # for i, iou in enumerate(ious):
                    #     if iou >= IOU2:
                    #         if i in reserve:
                    #             reserve.remove(i)

                    reserve_result = []
                    for n in range(3):
                        reserve_result.append([this_frame_results[n][i] for i in reserve])
                    reserve_result.append(this_frame_results[3])
                    seg_results[idx] = reserve_result

            # update key and value
            if t + 1 in to_memorize:
                keys, values = this_keys_m, this_values_m

        for j in range(3):
            seg_results[start_frame_idx][j].pop(0)

        # pred = torch.round(Es.float())
        results.append((Es, instance_idx))

        instance_idx += 1

    return results


def ol_aug(image, mask):
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
        # iaa.PadToFixedSize(width=crop_size[0], height=crop_size[1]),
        # iaa.CropToFixedSize(width=crop_size[0], height=crop_size[1]),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            # scale images to 90-110% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            # translate by -10 to +10 percent (per axis)
            rotate=(-5, 5),  # rotate by -5 to +5 degrees
            shear=(-3, 3),  # shear by -3 to +3 degrees
        ),
        # iaa.Cutout(nb_iterations=(1, 5), size=0.2, cval=0, squared=False),
    ], random_order=False)  # apply augmenters in random order

    seq_f = iaa.Sequential([
        iaa.Sometimes(0.5,
                      iaa.OneOf([
                          iaa.GaussianBlur((0.0, 3.0)),
                          iaa.MotionBlur(k=(3, 20)),
                      ]),
                      ),
        iaa.Sometimes(0.5,
                      iaa.OneOf([
                          iaa.Multiply((0.8, 1.2), per_channel=0.2),
                          iaa.MultiplyBrightness((0.5, 1.5)),
                          iaa.LinearContrast((0.5, 2.0), per_channel=0.2),
                          iaa.BlendAlpha((0., 1.), iaa.HistogramEqualization()),
                          iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=0.2),
                      ]),
                      ),
    ], random_order=False)

    combo_aug = np.array(seq_all.augment_images(images=combo))
    # print('combo_au: ', combo_aug.shape)
    images_aug = combo_aug[:, :, :, :3]
    masks_aug = combo_aug[:, :, :, 3:]
    images_aug = seq_f.augment_images(images=images_aug)

    return images_aug, masks_aug


@fn_timer
def online_learning():
    print('online learning...')
    Testset = TIANCHI_FUSAI(DATA_ROOT, imset='test.txt', target_size=OL_TARGET_SHAPE)
    print('Total test videos: {}'.format(len(Testset)))
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(MODEL)

    optimizer = torch.optim.Adam(model.parameters(), OL_LR, betas=(0.9, 0.99))

    print('Loading weights:', MODEL_PATH)
    model_ = torch.load(MODEL_PATH)
    if 'state_dict' in model_.keys():
        state_dict = model_['state_dict']
    else:
        state_dict = model_
    model.load_state_dict(state_dict, strict=True)

    # if 'optimizer' in model_.keys():
    #     try:
    #         optimizer.load_state_dict(model_['optimizer'])
    #     except Exception as e:
    #         print(e)

    if torch.cuda.is_available():
        model.cuda()
    # model.eval()  # turn-off BN
    model.train()
    # freeze bn
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    print('Start online learning...')
    progressbar = tqdm.tqdm(Testloader)

    for V in progressbar:
        F, info = V
        seq_name = info['name'][0]
        ori_shape = info['ori_shape']
        target_shape = info['target_shape']
        target_shape = (target_shape[0].cpu().numpy()[0], target_shape[1].cpu().numpy()[0])
        num_frames = info['num_frames'][0].item()
        if '_' in seq_name:
            video_name = seq_name.split('_')[0]
        else:
            video_name = seq_name
        seg_results = mask_inference(video_name, OL_TARGET_SHAPE, OL_SOLO_INTERVAL, OL_SCORE_THR)
        ol_clip_frames = OL_CLIPS

        # online learning by aug
        for _ in range(OL_ITER_PER_VIDEO):
            count = 1
            while True:
                frame = random.choice(seg_results)
                if len(frame[1]) > 0 or count >= 10:
                    break
                else:
                    count += 1
            if len(frame[1]) == 0:
                continue

            select_idx = random.choice(list(range(len(frame[0]))))
            start_mask = frame[0][select_idx][np.newaxis, np.newaxis, :, :].transpose(0, 2, 3, 1)
            score = frame[2][select_idx]
            frame_idx = frame[3]
            start_frame = F[:, :, frame_idx].cpu().numpy().transpose(0, 2, 3, 1)

            frames = []
            masks = []
            frames.append(start_frame)
            masks.append(start_mask)

            for _ in range(ol_clip_frames - 1):
                img_aug, mask_aug = ol_aug((start_frame * 255).astype(np.uint8), start_mask)
                frames.append(img_aug / 255)
                masks.append(mask_aug)
            prevs = masks[1:]
            Fs = torch.from_numpy(
                np.concatenate([f[np.newaxis, ...] for f in frames], axis=0).transpose(1, 4, 0, 2, 3)).float()
            Ms = torch.from_numpy(
                np.concatenate([m[np.newaxis, ...] for m in masks], axis=0).transpose(1, 4, 0, 2, 3)).long()
            Ps = torch.from_numpy(
                np.concatenate([p[np.newaxis, ...] for p in prevs], axis=0).transpose(1, 4, 0, 2, 3)).float()

            optimizer.zero_grad()
            loss_video, video_mIou = Run_video_sp(model, {'Fs': Fs, 'Ms': Ms, 'Ps': Ps, 'info': info},
                                                  Mem_every=1,
                                                  mode='train')
            print('aug finetune loss: {:.3f}, miou: {:.2f}'.format(loss_video, video_mIou))
            # backward
            loss_video.backward()
            optimizer.step()

        # online learning by sequence
        for _ in range(OL_ITER_PER_VIDEO):
            count = 1
            while True:
                idx, frame = random.choice(list(enumerate(seg_results)))
                if len(frame[1]) > 0 or count >= 10:
                    break
                else:
                    count += 1
            if len(frame[1]) == 0:
                continue

            seg_result_idx = [i[3] for i in seg_results]
            # start_frame_idx = np.argmax([max(i[2]) if i[2] != [] else 0 for i in seg_results])
            start_frame_idx = idx
            start_frame = frame[3]
            num_mask = len(frame[0])
            start_mask = frame[0][random.choice(range(num_mask))].astype(np.uint8)

            Ms = torch.empty((1, 1, ol_clip_frames) + OL_TARGET_SHAPE[::-1]).cuda().long()
            Ps = torch.empty((1, 1, ol_clip_frames - 1) + OL_TARGET_SHAPE[::-1]).cuda()
            complete_flag = True
            masks = []
            masks.append(start_mask)
            if start_frame_idx + ol_clip_frames <= len(seg_results):
                # train after
                frames = [seg_result_idx[start_frame_idx + i] for i in range(ol_clip_frames)]
                result_idxs = [start_frame_idx + i for i in range(ol_clip_frames)]
                for i in range(1, ol_clip_frames):
                    seg_result = seg_results[result_idxs[i]]
                    if seg_result[0] == []:
                        complete_flag = False
                        break
                    else:
                        ious = []
                        for mask in seg_result[0]:
                            iou = get_video_mIoU(masks[-1], mask)
                            ious.append(iou)
                        if np.max(ious) >= 0.5:
                            mi = np.argmax(ious)
                            masks.append(seg_result[0][mi])
                        else:
                            complete_flag = False
                            break
                if complete_flag and len(masks) == ol_clip_frames:
                    for i, mask in enumerate(masks):
                        Ms[:, :, i] = torch.from_numpy(mask).cuda()
                        if i != 0:
                            Ps[:, :, i - 1] = torch.from_numpy(mask).cuda()
                    Fs = F[:, :, frames].cuda()
                    optimizer.zero_grad()
                    loss_video, video_mIou = Run_video_sp(model, {'Fs': Fs, 'Ms': Ms, 'Ps': Ps, 'info': info},
                                                          Mem_every=1,
                                                          mode='train')
                    print('sequence finetune loss: {:.3f}, miou: {:.2f}'.format(loss_video, video_mIou))
                    # backward
                    loss_video.backward()
                    optimizer.step()
                else:
                    continue

    return model


@fn_timer
def vos_inference():
    # Model and version
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testset = TIANCHI_FUSAI(DATA_ROOT, imset='test.txt', target_size=TARGET_SHAPE, with_flip=WITH_FLIP,
                            test_scale=TEST_SCALE)
    print('Total test videos: {}'.format(len(Testset)))
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if not OL:
        model = nn.DataParallel(MODEL)
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
    else:
        model = online_learning()

    model.eval()
    torch.set_grad_enabled(False)

    code_name = 'Tianchi fusai'
    # date = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d%H%M')
    print('Start Testing:', code_name)
    progressbar = tqdm.tqdm(Testloader)

    for V in progressbar:
        Fs, info = V
        if isinstance(Fs, list):
            b, c, t, h, w = Fs[0].shape
        else:
            b, c, t, h, w = Fs.shape
        seq_name = info['name'][0]
        ori_shape = info['ori_shape']
        target_shape = info['target_shape']
        target_shape = (target_shape[0].cpu().numpy()[0], target_shape[1].cpu().numpy()[0])
        num_frames = info['num_frames'][0].item()
        if '_' in seq_name:
            video_name = seq_name.split('_')[0]
        else:
            video_name = seq_name

        frame_list = VIDEO_FRAMES[video_name]

        print('[{}]: num_frames: {}'.format(seq_name, num_frames))

        if isinstance(Fs, list):
            result = []
            for idx, f in enumerate(Fs):
                if idx == 1:
                    seg_results = mask_inference(video_name, target_shape, SOLO_INTERVAL, SCORE_THR, hflip=True)
                elif idx == 2:
                    seg_results = mask_inference(video_name, TEST_SCALE, SOLO_INTERVAL, SCORE_THR, hflip=False)
                else:
                    seg_results = mask_inference(video_name, target_shape, SOLO_INTERVAL, SCORE_THR, hflip=False)
                results = Run_video(model, f, seg_results, num_frames, Mem_every=5, model_name=MODEL_NAME)
                if idx == 1:
                    for i, (es, ins) in enumerate(results):
                        es = es.cpu().detach().numpy()
                        es = es[:, :, :, :, ::-1]
                        es = np.ascontiguousarray(es)
                        es = torch.from_numpy(es).cuda()
                        results[i] = (es, ins)
                if idx == 2:
                    for i, (es, ins) in enumerate(results):
                        e = torch.empty(b, 1, t, h, w)
                        for f in range(t):
                            e[:, :, f, :, :] = F.interpolate(es[:, :, f, :, :], (h, w))
                        e = e.cuda()
                        results[i] = (e, ins)
                result.append(results)
            results = merge_result(result)
        else:
            seg_results = mask_inference(video_name, target_shape, SOLO_INTERVAL, SCORE_THR)
            results = Run_video(model, Fs, seg_results, num_frames, Mem_every=5, model_name=MODEL_NAME)

        results = [(torch.round(a), b) for a, b in results]

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


def merge_result(result):
    base_result = result[0]
    final_result = {}

    for i in range(1, len(result)):
        ious = {}
        r = result[i]
        for es, ins in r:
            for bs, bi in base_result:
                es = torch.round(es)
                bs = torch.round(bs)
                iou = get_video_mIoU(es, bs)
                ious.setdefault(bi, {}).setdefault(ins, iou)

        for bs, bi in base_result:
            final_result.setdefault(bi, []).append(bs)
            if bi in ious.keys():
                best_match_iou = max(ious.get(bi).values())
                if best_match_iou >= 0.9:
                    for k, v in ious.get(bi).items():
                        if v == best_match_iou:
                            matched = k
                    for es, ins in r:
                        if ins == matched:
                            final_result[bi].append(es)
    fr = []
    for k, v in final_result.items():
        ins = k
        es = torch.mean(torch.cat([a.unsqueeze(0) for a in v]), dim=0)
        fr.append((es, ins))
    return fr


def get_video_mIoU(predn, all_Mn):  # [c,t,h,w]
    if isinstance(predn, torch.Tensor):
        pred = predn.squeeze().cpu().data.numpy()
    else:
        pred = predn
    # np.save('blackswan.npy', pred)
    if isinstance(all_Mn, torch.Tensor):
        gt = all_Mn.squeeze().cpu().data.numpy()  # [t,h,w]
    else:
        gt = all_Mn
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
def mask_inference(video_name, mask_shape, interval, score_thr, hflip=False):
    # build the model from a config file and a checkpoint file
    print('Generating frame mask...')
    model = SOLO_MODEL

    # test a single image
    frames = VIDEO_FRAMES.get(video_name)
    fi = []
    i = 0
    while True:
        fi.append(i)
        i += interval
        if i >= len(frames) - 1:
            fi.append(len(frames) - 1)
            break

    results = []
    for f in fi:
        img = os.path.join(DATA_ROOT, 'JPEGImages/{}/{}.jpg'.format(video_name, frames[f]))
        # img = Fs[0, :, f, :, :].cpu().numpy().transpose(1, 2, 0) * 255
        # img = img.astype(np.uint8)
        img = cv2.imread(img)
        if hflip:
            img = cv2.flip(img, 1)
        result, cost_time = inference_detector(model, img)
        result = filter_result(result, max_num=MAX_NUM)
        if result is None:
            continue
        result = process_solo_result(result, mask_shape)
        result = list(result) + [f]
        results.append(result)

    results = filter_score(results, score_thr=score_thr)

    # if MODE == 'offline':
    #     # visualize solo mask
    #     for result in results:
    #         mask_result = result[:3]
    #         frame = result[3]
    #         img = os.path.join(DATA_ROOT, 'JPEGImages/{}/{}.jpg'.format(video_name, frames[frame]))
    #         save_mask(img, mask_result, 0, MASK_PATH)
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


def filter_score(results, score_thr):  # list(array, array, array, int)
    filtered = []
    num = 0
    for result in results:
        idx = result[2] >= score_thr
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


@fn_timer
def blend_results(tmp_dir, merge_dir, data_dir):
    print('Blending results...')
    img_root = os.path.join(data_dir, 'JPEGImages')
    ann_root = os.path.join(data_dir, 'Annotations')
    # with open(os.path.join(data_dir, 'ImageSets/test.txt')) as f:
    #     test = f.readlines()
    # test = [img.strip() for img in test]
    test = os.listdir(tmp_dir)
    test = [name.split('_')[0] if '_' in name else name for name in test]
    test = np.unique(test)

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


@fn_timer
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


def _model(model_name):
    if model_name == 'motion':
        from STM.models.model_fusai import STM
        model = STM()
    elif model_name == 'aspp':
        from STM.models.model_fusai_aspp import STM
        model = STM()
    elif model_name == 'enhanced':
        from STM.models.model_enhanced import STM
        model = STM()
    elif model_name == 'enhanced_motion':
        from STM.models.model_enhanced_motion import STM
        model = STM()
    elif model_name == 'standard':
        from STM.models.model import STM
        model = STM()
    elif model_name == 'varysize':
        from STM.models.model_enhanced_varysize import STM
        model = STM()
    elif model_name == 'sp':
        from STM.models.model_fusai_spatial_prior import STM
        model = STM()
    else:
        raise ValueError

    return model


if __name__ == '__main__':
    MODE = 'online'
    if MODE == 'online':
        DATA_ROOT = '/versa/dyy/solo-vos/user_data/data'
        IMG_ROOT = '/versa/dataset/TIANCHI2021/preTest'
        MODEL_PATH = '/versa/dyy/solo-vos/user_data/model_data/sp_interval4.pth'
        # MODEL_PATH = '/workspace/user_data/model_data/enhanced_motion_ckpt_1e_0827.pth'
        SAVE_PATH = '/versa/dyy/solo-vos'
        TMP_PATH = '/versa/dyy/solo-vos/user_data/tmp_data'
        MERGE_PATH = '/versa/dyy/solo-vos/user_data/merge_data'
        MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')
        CONFIG_FILE = r'/versa/dyy/solo-vos/cfg/aug_solov2_r101.py'
        CKPT_FILE = r'/versa/dyy/solo-vos/user_data/model_data/solov2_9cls.pth'
        TEMPLATE_MASK = r'/versa/dyy/solo-vos/user_data/template_data/00001.png'

        MODEL_NAME = 'sp'
    else:
        DATA_ROOT = '/workspace/solo/code/user_data/data'
        IMG_ROOT = '/workspace/dataset/VOS/mini_fusai/JPEGImages/'
        # IMG_ROOT = '/workspace/dataset/VOS/test_dataset2/JPEGImages/'
        # IMG_ROOT = '/workspace/dataset/VOS/tianchi_val/JPEGImages/'
        MODEL_PATH = '/workspace/solo/backup_models/STM/sp_interval4.pth'
        # MODEL_PATH = '/workspace/solo/backup_models/motion_crop_ckpt_44e.pth' # aspp + motion
        # MODEL_PATH = '/workspace/solo/backup_models/enhanced2_interval7.pth'
        # MODEL_PATH = r'/workspace/solo/backup_models/enhanced2_interval22.pth'
        SAVE_PATH = '/workspace/solo/code/user_data/'
        TMP_PATH = '/workspace/solo/code/user_data/tmp_data'
        MERGE_PATH = '/workspace/solo/code/user_data/merge_data'
        MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')
        CONFIG_FILE = r'/workspace/solo/code/cfg/aug_solov2_r101.py'
        CKPT_FILE = r'/workspace/solo/backup_models/solo/solov2_9cls.pth'
        TEMPLATE_MASK = r'/workspace/solo/code/user_data/template_data/00001.png'
        VIDEO_PATH = '/workspace/solo/code/user_data/video_data'
        GT_PATH = r'/workspace/dataset/VOS/tianchiyusai/Annotations/'

        MODEL_NAME = 'sp'

        process_tianchi_dir(SAVE_PATH)

    MODEL = _model(MODEL_NAME)

    if IMG_ROOT is not None or IMG_ROOT != '':
        process_data_root(DATA_ROOT, IMG_ROOT)

    PALETTE = Image.open(TEMPLATE_MASK).getpalette()
    VIDEO_FRAMES = analyse_images(DATA_ROOT)

    OL = False
    OL_LR = 1e-7
    OL_TARGET_SHAPE = (864, 480)
    OL_CLIPS = 3
    OL_SOLO_INTERVAL = 2
    OL_ITER_PER_VIDEO = 50
    OL_SCORE_THR = 0.7

    WITH_FLIP = True
    # WITH_FLIP = True
    # TEST_SCALE = (1120, 608)
    TEST_SCALE = None
    TARGET_SHAPE = (1008, 560)
    # TARGET_SHAPE = (864, 480)
    SCORE_THR = 0.3
    SOLO_INTERVAL = 2
    MAX_NUM = 8
    IOU1 = 0.5
    # IOU2 = 0.1
    # TO_MEMORY_MIN_INTERVAL = 5
    BLEND_IOU_THR = 1

    SOLO_MODEL = init_detector(CONFIG_FILE, CKPT_FILE, device='cuda:0')

    generate_imagesets()
    vos_inference()
    blend_results(TMP_PATH, MERGE_PATH, DATA_ROOT)
    if MODE == 'online':
        zip_result(MERGE_PATH, SAVE_PATH)
    else:
        generate_videos(DATA_ROOT, MERGE_PATH, VIDEO_PATH)
        miou, num, miou2 = calculate_videos_miou(MERGE_PATH, GT_PATH)
        print('offline evaluation miou: {:.3f}, instances miou: {:.3f}, {} videos counted'.format(miou, miou2, num))
        with open(os.path.join(VIDEO_PATH, 'result.txt'), 'w') as f:
            f.write(
                'offline evaluation miou: {:.3f}, instances miou: {:.3f}, {} videos counted'.format(miou, miou2, num))
