from __future__ import division
import torch
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
import tqdm
import os
import argparse
import matplotlib.pyplot as plt
import csv
import warnings
import cv2

warnings.filterwarnings('ignore')

import os.path
import time
import datetime

### My libs
from STM.tools.log import Logger
from STM.dataloader.dataset_rgmp_v1 import DAVIS
from STM.dataloader.tianchi_dataset import TIANCHI
# from STM.models.model_fusai import STM
from STM.tools.utils import _loss, get_video_mIoU


def parse_args():
    parser = argparse.ArgumentParser(description='Train a tracker')

    parser.add_argument('--work_dir', type=str, default='./exp/stm_reg800_v4.3',
                        help='the dir to save models.pth and logs and masks')
    parser.add_argument("--mode", type=str, default='train', help="train or val")
    parser.add_argument("--model", type=str, default='motion', help="model type")
    parser.add_argument('--load_from', type=str,
                        default='/workspace/STM_test/user_data/model_data/model.pth')
    # train
    parser.add_argument('--train_with_val', action='store_true', help='whether to val before train')
    parser.add_argument('--val_at_start', action='store_true', help='whether to val at start of training')
    parser.add_argument('--resume_from', default='', help='the checkpoint file to resume from')
    parser.add_argument('--train_data', type=str, default='davis')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip_size', type=int, default=3)
    parser.add_argument('--interval', type=str, default='1,25', help='interval range')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--epoch_per_interval', type=int, default=30, help='epochs per interval')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--validate_interval', type=int, default=1)
    parser.add_argument('--info_interval', type=int, default=10)
    parser.add_argument('--train_aug', action='store_true', help='whether to train with data aug')
    parser.add_argument("--davis", type=str, default='/workspace/dataset/VOS/tianchiyusai')
    parser.add_argument("--gpu", type=str, default='0', help="0; 0,1; 0,3; etc")
    # val
    parser.add_argument("--save_masks", type=bool, default=True, help='whether save predicting mask when mode is val')
    parser.add_argument('--vis_val', type=bool, default=False, help='visualize the result of val')

    return parser.parse_args()


def Run_video_motion(model, Fs, Ms, info, Mem_every=None, Mem_number=None, mode='train'):
    num_frames = info['num_frames'][0].item()
    intervals = info['intervals']
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    B, _, f, H, W = Fs.shape
    Es = torch.zeros((B, 1, f, H, W)).float().cuda()  # [1,1,50,480,864][b,c,t,h,w]
    Es[:, :, 0] = Ms[:, :, 0]

    loss_video = torch.tensor(0.0).cuda()
    loss_total = torch.tensor(0.0).cuda()

    for t in range(1, num_frames):
        interval = intervals[t][0].item()
        if mode == 'train':
            if interval != 1:
                model.module.Memory.eval()
            else:
                model.module.Memory.train()
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
        prev_mask = torch.round(Es[:, :, t - 1].detach()).float()
        logits, p_m2, p_m3 = model([Fs[:, :, t], this_keys_m, this_values_m, prev_mask])
        em = F.softmax(logits, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        #  calculate loss on cuda
        if mode == 'train' or mode == 'val':
            Ms_cuda = Ms[:, 0, t].cuda()
            loss_video += _loss(logits, Ms_cuda) + 0.5 * _loss(p_m2, Ms_cuda) + 0.25 * _loss(p_m3, Ms_cuda)
            loss_total = loss_video

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

    #  calculate mIOU on cuda
    pred = torch.round(Es.float().cuda())
    if mode == 'train' or mode == 'val':
        video_mIoU = 0
        for n in range(len(Ms)):  # Nth batch
            video_mIoU = video_mIoU + get_video_mIoU(pred[n], Ms[n].cuda())  # mIOU of video(t frames) for each batch
        video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch

        return loss_total / num_frames, video_mIoU

    elif mode == 'test':
        return pred, Es


def Run_video_standard(model, Fs, Ms, info, Mem_every=None, Mem_number=None, mode='train'):
    num_frames = info['num_frames'][0].item()
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    B, _, f, H, W = Fs.shape
    Es = torch.zeros((B, 1, f, H, W)).float().cuda()  # [1,1,50,480,864][b,c,t,h,w]
    Es[:, :, 0] = Ms[:, :, 0]

    loss_video = torch.tensor(0.0).cuda()
    loss_total = torch.tensor(0.0).cuda()

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
        logits, p_m2, p_m3 = model([Fs[:, :, t], this_keys_m, this_values_m])
        em = F.softmax(logits, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        #  calculate loss on cuda
        if mode == 'train' or mode == 'val':
            Ms_cuda = Ms[:, 0, t].cuda()
            loss_video += _loss(logits, Ms_cuda) + 0.5 * _loss(p_m2, Ms_cuda) + 0.25 * _loss(p_m3, Ms_cuda)
            loss_total = loss_video

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

    #  calculate mIOU on cuda
    pred = torch.round(Es.float().cuda())
    if mode == 'train' or mode == 'val':
        video_mIoU = 0
        for n in range(len(Ms)):  # Nth batch
            video_mIoU = video_mIoU + get_video_mIoU(pred[n], Ms[n].cuda())  # mIOU of video(t frames) for each batch
        video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch

        return loss_total / num_frames, video_mIoU

    elif mode == 'test':
        return pred, Es


def Run_video_enhanced(model, Fs, Ms, info, Mem_every=None, Mem_number=None, mode='train'):
    num_frames = info['num_frames'][0].item()
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    b, c, f, h, w = Fs.shape
    Es = torch.zeros((b, 1, f, h, w)).float().cuda()  # [1,1,50,480,864][b,c,t,h,w]
    Es[:, :, 0] = Ms[:, :, 0]

    # template_size = 7 * 16
    # Os = torch.zeros((b, c, template_size, template_size))
    Os = torch.zeros((b, c, int(h / 4), int(w / 4)))
    first_frame = Fs[:, :, 0].detach()
    first_mask = Ms[:, :, 0].detach()
    first_frame = first_frame * first_mask.repeat(1, 3, 1, 1).type(torch.float)
    for i in range(b):
        mask_ = first_mask[i]
        mask_ = mask_.squeeze(0).cpu().numpy().astype(np.uint8)
        assert np.any(mask_)
        x, y, w_, h_ = cv2.boundingRect(mask_)
        patch = first_frame[i, :, y:(y + h_), x:(x + w_)].cpu().numpy()
        patch = patch.transpose(1, 2, 0)
        # patch = cv2.resize(patch, (template_size, template_size))
        # patch = patch.transpose(2, 1, 0)
        patch = cv2.resize(patch, (int(w / 4), int(h / 4)))
        patch = patch.transpose(2, 0, 1)
        patch = torch.from_numpy(patch)
        Os[i, :, :, :] = patch

    loss_video = torch.tensor(0.0).cuda()

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
        logits, p_m2, p_m3 = model([Fs[:, :, t], Os, this_keys_m, this_values_m])  # B 2 h w
        em = F.softmax(logits, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

        #  calculate loss on cuda
        if mode == 'train' or mode == 'val':
            Ms_cuda = Ms[:, 0, t].cuda()
            loss_video += (_loss(logits, Ms_cuda) + 0.5 * _loss(p_m2, Ms_cuda) + 0.25 * _loss(p_m3, Ms_cuda))

    #  calculate mIOU on cuda
    pred = torch.round(Es.float().cuda())
    if mode == 'train' or mode == 'val':
        video_mIoU = 0
        for n in range(len(Ms)):  # Nth batch
            video_mIoU = video_mIoU + get_video_mIoU(pred[n],
                                                     Ms[n].float().cuda())  # mIOU of video(t frames) for each batch
        video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch

        return loss_video / num_frames, video_mIoU

    elif mode == 'test':
        return pred, Es


def Run_video_enhanced_varysize(model, Fs, Ms, info, Mem_every=None, Mem_number=None, mode='train'):
    num_frames = info['num_frames'][0].item()
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    b, c, f, h, w = Fs.shape
    Es = torch.zeros((b, 1, f, h, w)).float().cuda()  # [1,1,50,480,864][b,c,t,h,w]
    Es[:, :, 0] = Ms[:, :, 0]

    os = []
    first_frame = Fs[:, :, 0].detach()
    first_mask = Ms[:, :, 0].detach()
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

    loss_video = torch.tensor(0.0).cuda()

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
        logits, p_m2, p_m3 = model([Fs[:, :, t], os, this_keys_m, this_values_m])  # B 2 h w
        em = F.softmax(logits, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

        #  calculate loss on cuda
        if mode == 'train' or mode == 'val':
            Ms_cuda = Ms[:, 0, t].cuda()
            loss_video += (_loss(logits, Ms_cuda) + 0.5 * _loss(p_m2, Ms_cuda) + 0.25 * _loss(p_m3, Ms_cuda))

    #  calculate mIOU on cuda
    pred = torch.round(Es.float().cuda())
    if mode == 'train' or mode == 'val':
        video_mIoU = 0
        for n in range(len(Ms)):  # Nth batch
            video_mIoU = video_mIoU + get_video_mIoU(pred[n],
                                                     Ms[n].float().cuda())  # mIOU of video(t frames) for each batch
        video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch

        return loss_video / num_frames, video_mIoU

    elif mode == 'test':
        return pred, Es


def Run_video_enhanced_motion(model, Fs, Ms, info, Mem_every=None, Mem_number=None, mode='train'):
    num_frames = info['num_frames'][0].item()
    intervals = info['intervals']
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    b, c, f, h, w = Fs.shape
    Es = torch.zeros((b, 1, f, h, w)).float().cuda()  # [1,1,50,480,864][b,c,t,h,w]
    Es[:, :, 0] = Ms[:, :, 0]

    loss_video = torch.tensor(0.0).cuda()
    loss_total = torch.tensor(0.0).cuda()

    Os = torch.zeros((b, c, int(h / 4), int(w / 4)))
    first_frame = Fs[:, :, 0].detach()
    first_mask = Ms[:, :, 0].detach()
    first_frame = first_frame * first_mask.repeat(1, 3, 1, 1).type(torch.float)
    for i in range(b):
        mask_ = first_mask[i]
        mask_ = mask_.squeeze(0).cpu().numpy().astype(np.uint8)
        assert np.any(mask_)
        x, y, w_, h_ = cv2.boundingRect(mask_)
        # c_x = x + w_ / 2
        # c_y = y + h_ / 2
        # c_x = np.clip(c_x, h / 8, 7 * h / 8)
        # c_y = np.clip(c_y, w / 8, 7 * w / 8)
        patch = first_frame[i, :, y: (y + h_), x: (x + w_)].cpu().numpy()
        patch = patch.transpose(1, 2, 0)
        # patch = cv2.resize(patch, (template_size, template_size))
        # patch = patch.transpose(2, 1, 0)
        patch = cv2.resize(patch, (int(w / 4), int(h / 4)))
        patch = patch.transpose(2, 0, 1)
        patch = torch.from_numpy(patch)
        Os[i, :, :, :] = patch

    for t in range(1, num_frames):
        interval = intervals[t][0].item()
        if mode == 'train':
            if interval != 1:
                model.module.Memory.eval()
            else:
                model.module.Memory.train()
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
        prev_mask = torch.round(Es[:, :, t - 1].detach()).float()
        logits, p_m2, p_m3 = model([Fs[:, :, t], Os, this_keys_m, this_values_m, prev_mask])
        em = F.softmax(logits, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        #  calculate loss on cuda
        if mode == 'train' or mode == 'val':
            Ms_cuda = Ms[:, 0, t].cuda()
            loss_video += _loss(logits, Ms_cuda) + 0.5 * _loss(p_m2, Ms_cuda) + 0.25 * _loss(p_m3, Ms_cuda)
            loss_total = loss_video

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

    #  calculate mIOU on cuda
    pred = torch.round(Es.float().cuda())
    if mode == 'train' or mode == 'val':
        video_mIoU = 0
        for n in range(len(Ms)):  # Nth batch
            video_mIoU = video_mIoU + get_video_mIoU(pred[n], Ms[n].cuda())  # mIOU of video(t frames) for each batch
        video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch

        return loss_total / num_frames, video_mIoU

    elif mode == 'test':
        return pred, Es


def validate(args, val_loader, model):
    print('validating...')
    model.eval()  # turn-off BN
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS')

    loss_all_videos = 0.0
    miou_all_videos = 0.0
    videos_name = []
    videos_miou = []
    videos_loss = []
    progressbar = tqdm.tqdm(val_loader)
    for seq, batch in enumerate(progressbar):
        Fs, Ms, info = batch['Fs'], batch['Ms'], batch['info']
        num_frames = info['num_frames'][0].item()
        # error_nums = 0
        with torch.no_grad():
            name = info['name']
            loss_video, video_mIou = run_fun(model, Fs, Ms, info, Mem_every=5, Mem_number=None,
                                             mode='val')
            loss_all_videos += loss_video
            miou_all_videos += video_mIou
            progressbar.set_description(
                'val_complete:{}, name:{}, loss:{}, miou:{}'.format(seq / len(val_loader), name, loss_video,
                                                                    video_mIou))

            if args.vis_val and args.mode == 'val':
                videos_name.append(name[0])
                videos_miou.append(video_mIou)
                videos_loss.append(loss_video.cpu().numpy())

    loss_all_videos /= len(val_loader)
    miou_all_videos /= len(val_loader)

    # if args.vis_val and args.mode == 'val':
    #     plt.bar(videos_name, videos_loss)
    #     plt.xticks(videos_name, videos_name, rotation=90)
    #     plt.axhline(y=loss_all_videos, color="red")
    #     plt.savefig(args.work_dir + '/' + 'loss.png')
    #     plt.close()
    #
    #     plt.bar(videos_name, videos_miou)
    #     plt.xticks(videos_name, videos_name, rotation=90)
    #     plt.axhline(y=miou_all_videos, color="red")
    #     plt.savefig(args.work_dir + '/' + 'miou.png')
    #     plt.close()
    #
    #     ## writer the result into csv
    #     csv_file = args.work_dir + '/' + 'result.csv'
    #     with open(csv_file, 'w') as f:
    #         csv_write = csv.writer(f)
    #         csv_head = ['video_name', 'miou']
    #         csv_write.writerow(csv_head)
    #     with open(csv_file, 'a+') as f:
    #         csv_write = csv.writer(f)
    #         for i in range(len(videos_name)):
    #             data_row = [videos_name[i], str(videos_miou[i])]
    #             csv_write.writerow(data_row)
    #         csv_write.writerow(['all_videos', str(np.mean(videos_miou))])

    return loss_all_videos, miou_all_videos


def train(args, optimizer, train_loader, model, epochs, epoch_start=0, lr=1e-5):
    print('training...')
    MODEL = 'STM'
    print(MODEL, 'Training on ', args.train_data)

    code_name = '{}_DAVIS_{}'.format(MODEL, args.train_data)
    print('Start Training:', code_name)

    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epoch_start, epochs):
        model.train()
        # turn-off BN
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        video_parts = len(train_loader)
        loss_record = 0
        miou_record = 0
        loss_total = 0
        miou_total = 0
        progressbar = tqdm.tqdm(train_loader)
        for seq, batch in enumerate(progressbar):
            Fs, Ms, info = batch['Fs'], batch['Ms'], batch['info']
            num_frames = info['num_frames'][0].item()
            optimizer.zero_grad()

            loss_video, video_mIou = run_fun(model, Fs, Ms, info, Mem_every=1, Mem_number=None,
                                             mode='train')

            # backward
            loss_video.backward()
            optimizer.step()

            # record loss
            loss_record += loss_video.cpu().detach().numpy()
            loss_total += loss_video.cpu().detach().numpy()
            miou_record += video_mIou
            miou_total += video_mIou
            if (seq + 1) % INFO_INTERVAL == 0:
                log.logger.info(
                    'epoch:{}, loss_video:{:.3f}({:.3f}), video_mIou:{:.3f}({:.3f}), complete:{:.2f}, lr:{}'.format(
                        epoch,
                        loss_record / INFO_INTERVAL,
                        loss_total / (seq + 1),
                        miou_record / INFO_INTERVAL,
                        miou_total / (seq + 1),
                        seq / video_parts,
                        lr))
                loss_record = 0
                miou_record = 0

        # save checkpoints
        if (epoch + 1) % args.save_interval == 0:
            print('saving checkpoints...')
            ckpt_dir = os.path.join(args.work_dir, "ckpt", DATETIME)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'lr': args.lr,
                'optimizer': optimizer.state_dict()
            }, os.path.join(ckpt_dir, 'ckpt_{}e.pth'.format(epoch)))

        if args.train_with_val and (epoch + 1) % args.validate_interval == 0:
            # validate
            loss_val, miou_val = validate(args, val_loader, model)
            log.logger.info('val loss:{:.3f}, val miou:{:.3f}'.format(loss_val, miou_val))


def _model(model_name):
    if model_name == 'motion':
        from STM.models.model_fusai import STM
        model = STM()
    elif model_name == 'aspp':
        from STM.models.model_fusai_aspp import STM
        model = STM()
        # model.eval()
        # model.Decoder.train()
    elif model_name == 'enhanced':
        from STM.models.model_enhanced import STM
        model = STM()
        model.eval()
        model.KV_Q.train()
    elif model_name == 'standard':
        from STM.models.model import STM
        model = STM()
    elif model_name == 'enhanced_motion':
        from STM.models.model_enhanced_motion import STM
        model = STM()
    elif model_name == 'varysize':
        from STM.models.model_enhanced_varysize import STM
        model = STM()
    elif model_name == 'sp':
        from STM.models.model_fusai_spatial_prior import STM
        model = STM()

    return model


def _run(model_name):
    if model_name == 'motion':
        return Run_video_motion
    elif model_name == 'aspp':
        return Run_video_motion
    elif model_name == 'enhanced':
        return Run_video_enhanced
    elif model_name == 'standard':
        return Run_video_standard
    elif model_name == 'enhanced_motion':
        return Run_video_enhanced_motion
    elif model_name == 'varysize':
        return Run_video_enhanced_varysize
    elif model_name == 'sp':
        return Run_video_motion


if __name__ == '__main__':
    args = parse_args()
    TARGET_SHAPE = (992, 544)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    GPU = args.gpu
    INFO_INTERVAL = args.info_interval
    intervals = list(range(int(args.interval.split(',')[0]), int(args.interval.split(',')[1]) + 1))
    epoch_per_interval = args.epoch_per_interval

    model_name = args.model
    model = _model(model_name)

    run_fun = _run(model_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    DATETIME = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m%d-%H%M%S')

    log_path = os.path.join(args.work_dir, 'ckpt', DATETIME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = Logger(os.path.join(log_path, DATETIME + '.log'))
    log.logger.info(args)

    # prepare val data
    DAVIS_ROOT = args.davis
    palette = Image.open(DAVIS_ROOT + '/Annotations/606332/00000.png').getpalette()

    val_dataset = TIANCHI(DAVIS_ROOT, phase='val', imset='tianchi_val.txt', separate_instance=True,
                          target_size=TARGET_SHAPE, same_frames=False)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    # load weights.pth
    if args.load_from:
        print('load pretrained from:', args.load_from)
        model.load_state_dict(torch.load(args.load_from), strict=False)
    # resume_from
    if args.resume_from and args.mode == 'val':
        print('resume from:', args.resume_from)
        ckpt = torch.load(args.resume_from)
        model.load_state_dict(ckpt['state_dict'], strict=False)

    if args.mode == "val":
        loss_val, miou_val = validate(args, val_loader, model)
        log.logger.info('val loss:{}, val miou:{}'.format(loss_val, miou_val))

    elif args.mode == "train":
        # set training para
        clip_size = args.clip_size
        BATCH_SIZE = args.batch_size
        interval = args.interval

        epoch_start = 0
        lr = args.lr
        optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.99))

        if args.resume_from:
            print('resume from:', args.resume_from)
            ckpt = torch.load(args.resume_from)
            model.load_state_dict(ckpt['state_dict'], strict=False)
            # epoch_start = ckpt['epoch'] + 1
            # if 'lr' in ckpt.keys():
            #     lr = ckpt['lr']
            if 'optimizer' in ckpt.keys():
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except Exception as e:
                    print(e)

        if args.val_at_start:
            # validate
            loss_val, miou_val = validate(args, val_loader, model)
            log.logger.info('val loss:{:.3f}, val miou:{:.3f}'.format(loss_val, miou_val))
        # run train
        end_epochs = epoch_per_interval
        for i in intervals:
            log.logger.info('Training interval:{}'.format(i))
            # prepare training data
            train_dataset = TIANCHI(DAVIS_ROOT, phase='train', imset='tianchi_train.txt', separate_instance=True,
                                    only_single=False, target_size=TARGET_SHAPE, clip_size=clip_size, mode='sequence',
                                    interval=i, train_aug=args.train_aug, keep_one_prev=True)
            train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                           pin_memory=True)

            train(args, optimizer, train_loader, model, end_epochs, epoch_start, lr)
            epoch_start += epoch_per_interval
            end_epochs += epoch_per_interval
