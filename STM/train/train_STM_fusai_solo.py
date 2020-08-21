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

warnings.filterwarnings('ignore')

import logging  # 引入logging模块
from logging import handlers
import os.path
import time
import datetime
import cv2
import glob
from mmdet.apis import init_detector, inference_detector

### My libs
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
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip_size', type=int, default=3)
    parser.add_argument('--interval', type=str, default='1,25', help='interval range')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--epoch_per_interval', type=int, default=30, help='epochs per interval')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--validate_interval', type=int, default=1)
    parser.add_argument("--davis", type=str, default='/workspace/dataset/VOS/tianchiyusai')
    parser.add_argument("--gpu", type=str, default='0', help="0; 0,1; 0,3; etc")
    # val
    parser.add_argument("--save_masks", type=bool, default=True, help='whether save predicting mask when mode is val')
    parser.add_argument('--vis_val', type=bool, default=True, help='visualize the result of val')

    return parser.parse_args()


# def Run_video(model, Fs, Ms, num_frames, Mem_every=None, Mem_number=None, mode='train'):
#     if Mem_every:
#         to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
#     elif Mem_number:
#         to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
#     else:
#         raise NotImplementedError
#
#     B, _, f, H, W = Fs.shape
#     Es = torch.zeros((B, 1, f, H, W)).float().cuda()  # [1,1,50,480,864][b,c,t,h,w]
#     Es[:, :, 0] = Ms[:, :, 0]
#
#     loss_video = torch.tensor(0.0).cuda()
#     loss_total = torch.tensor(0.0).cuda()
#
#     for t in range(1, num_frames):
#         # memorize
#         pre_key, pre_value = model([Fs[:, :, t - 1], Es[:, :, t - 1]])
#         pre_key = pre_key.unsqueeze(2)
#         pre_value = pre_value.unsqueeze(2)
#
#         if t - 1 == 0:  # the first frame
#             this_keys_m, this_values_m = pre_key, pre_value
#         else:  # other frame
#             this_keys_m = torch.cat([keys, pre_key], dim=2)
#             this_values_m = torch.cat([values, pre_value], dim=2)
#
#         # segment
#         logits, p_m2, p_m3 = model([Fs[:, :, t], this_keys_m, this_values_m, Es[:, :, t - 1].detach()])  # B 2 h w
#         em = F.softmax(logits, dim=1)[:, 1]  # B h w
#         Es[:, 0, t] = em
#
#         #  calculate loss on cuda
#         if mode == 'train' or mode == 'val':
#             Ms_cuda = Ms[:, 0, t].cuda()
#             # loss_video += (_loss(logits, Ms_cuda) + 0.5 * _loss(p_m2, Ms_cuda) + 0.25 * _loss(p_m3, Ms_cuda))
#             loss_video += _loss(logits, Ms_cuda) + 0.5 * _loss(p_m2, Ms_cuda) + 0.25 * _loss(p_m3, Ms_cuda)
#             # loss_video += _loss(logits, Ms_cuda)
#             loss_total = loss_video
#
#         # update key and value
#         if t - 1 in to_memorize:
#             keys, values = this_keys_m, this_values_m
#             # keys, values = this_keys_m.detach(), this_values_m.detach()
#
#     # if args.save_masks and args.mode == 'val':
#     #     # save mask
#     #     save_img_path = os.path.join(args.work_dir, 'masks', name[0])
#     #     if not os.path.exists(save_img_path):
#     #         os.makedirs(save_img_path)
#     #     for i in range(len(Es[0, 0])):
#     #         img_np = Es[0, 0, i].detach().cpu().numpy()
#     #         img_np = (np.round(img_np * 255)).astype(np.uint8)
#     #         img = Image.fromarray(img_np).convert('L')
#     #         img.save(save_img_path + '/' + '{:05d}.png'.format(i))
#
#     #  calculate mIOU on cuda
#     pred = torch.round(Es.float().cuda())
#     if mode == 'train' or mode == 'val':
#         video_mIoU = 0
#         for n in range(len(Ms)):  # Nth batch
#             video_mIoU = video_mIoU + get_video_mIoU(pred[n], Ms[n].cuda())  # mIOU of video(t frames) for each batch
#         video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch
#
#         return loss_total / num_frames, video_mIoU
#
#     elif mode == 'test':
#         return pred, Es


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
        name = info['name']
        batch_size = len(name)
        solo_results = []
        for i in range(batch_size):
            seq_name = name[i]
            if '_' in seq_name:
                video_name = seq_name.split('_')[0]
            else:
                video_name = seq_name
            target_shape = tuple([t[i].item() for t in info['target_shape']])
            frames = [f[i].item() for f in info['frames']]
            seg_results = mask_inference(video_name, target_shape, frames)
            solo_results.append(seg_results)
        # error_nums = 0
        with torch.no_grad():
            # loss_video, video_mIou = Run_video(model, Fs, Ms, num_frames, Mem_every=1, Mem_number=None, args=args)
            loss_video, video_mIou = Run_video(model, Fs, Ms, num_frames, solo_results=solo_results, Mem_every=5,
                                               Mem_number=None, mode='val')

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

    if args.vis_val and args.mode == 'val':
        plt.bar(videos_name, videos_loss)
        plt.xticks(videos_name, videos_name, rotation=90)
        plt.axhline(y=loss_all_videos, color="red")
        plt.savefig(args.work_dir + '/' + str(args.year) + '_loss.png')
        plt.close()

        plt.bar(videos_name, videos_miou)
        plt.xticks(videos_name, videos_name, rotation=90)
        plt.axhline(y=miou_all_videos, color="red")
        plt.savefig(args.work_dir + '/' + str(args.year) + '_miou.png')
        plt.close()

        ## writer the result into csv
        csv_file = args.work_dir + '/' + str(args.year) + '_result.csv'
        with open(csv_file, 'w') as f:
            csv_write = csv.writer(f)
            csv_head = ['video_name', 'miou']
            csv_write.writerow(csv_head)
        with open(csv_file, 'a+') as f:
            csv_write = csv.writer(f)
            for i in range(len(videos_name)):
                data_row = [videos_name[i], str(videos_miou[i])]
                csv_write.writerow(data_row)
            csv_write.writerow(['all_videos', str(np.mean(videos_miou))])

    return loss_all_videos, miou_all_videos


def train(args, optimizer, train_loader, model, epochs, epoch_start=0, lr=1e-5):
    print('training...')
    MODEL = 'STM'

    code_name = 'Tianchi'
    print('Start Training', code_name)

    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epoch_start, epochs):
        if args.val_at_start:
            # validate
            loss_val, miou_val = validate(args, val_loader, model)
            log.logger.info('val loss:{:.3f}, val miou:{:.3f}'.format(loss_val, miou_val))

        model.train()
        # turn-off BN
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        video_parts = len(train_loader)
        loss_record = 0
        miou_record = 0
        progressbar = tqdm.tqdm(train_loader)
        for seq, batch in enumerate(progressbar):
            Fs, Ms, info = batch['Fs'], batch['Ms'], batch['info']
            num_frames = info['num_frames'][0].item()
            name = info['name']
            batch_size = len(name)
            solo_results = []
            for i in range(batch_size):
                seq_name = name[i]
                if '_' in seq_name:
                    video_name = seq_name.split('_')[0]
                else:
                    video_name = seq_name
                target_shape = tuple([t[i].item() for t in info['target_shape']])
                frames = [f[i].item() for f in info['frames']]
                seg_results = mask_inference(video_name, target_shape, frames)
                solo_results.append(seg_results)

            optimizer.zero_grad()

            loss_video, video_mIou = Run_video(model, Fs, Ms, num_frames, solo_results=solo_results, Mem_every=1,
                                               Mem_number=None)

            # backward
            loss_video.backward()
            optimizer.step()

            # record loss
            loss_record += loss_video.cpu().detach().numpy()
            miou_record += video_mIou
            if (seq + 1) % 5 == 0:
                log.logger.info(
                    'epoch:{}, loss_video:{:.3f}, video_mIou:{:.3f}, complete:{:.2f}, lr:{}'.format(
                        epoch,
                        loss_record / (seq + 1),
                        miou_record / (seq + 1),
                        seq / video_parts,
                        lr))

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


def mask_inference(video_name, mask_shape, frames):
    # build the model from a config file and a checkpoint file
    # print('Generating frame mask...')
    model = init_detector(CONFIG_FILE, CKPT_FILE, device='cuda:0')

    # test a single imag
    fs = os.listdir(os.path.join(DATA_ROOT, 'JPEGImages/{}'.format(video_name)))

    results = []
    for f in frames:
        img = os.path.join(DATA_ROOT, 'JPEGImages/{}/{}'.format(video_name, fs[f]))
        result, cost_time = inference_detector(model, img)
        result = filter_result(result, max_num=MAX_NUM, score_thr=SCORE_THR)
        if result is None:
            result = [None, None, None]
        else:
            result = process_solo_result(result, mask_shape)
        result = list(result) + [f]
        results.append(result)

    # results = filter_score(results)

    # visualize solo mask
    # for result in results:
    #     mask_result = result[:3]
    #     frame = result[3]
    #     img = os.path.join(DATA_ROOT, 'JPEGImages/{}/{}.jpg'.format(video_name, frames[frame]))
    #     save_mask(img, mask_result, 0, MASK_PATH)
    return results


# def filter_score(results):  # list(array, array, array, int)
#     filtered = []
#     num = 0
#     for result in results:
#         idx = result[2] >= SCORE_THR
#         num += np.sum(idx)
#         if np.any(idx):
#             filtered.append([list(result[i][idx]) for i in range(3)] + [result[3]])
#
#     if num == 0:
#         filtered = []
#         # no mask score larger than threshold
#         scores = np.array([result[2] for result in results])
#         scores = np.concatenate(scores)
#         score_thr = np.max(scores) * 0.8
#         for result in results:
#             idx = result[2] >= score_thr
#             if np.any(idx):
#                 filtered.append([list(result[i][idx]) for i in range(3)] + [result[3]])
#
#     return filtered


def filter_result(result, index=0, max_num=8, score_thr=0.0):
    assert isinstance(result, list)
    result = result[0]
    if result is None:
        return None
    mask, cate, score = result
    idxs = (cate == index) & (score >= score_thr)
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


# def analyse_images(data_root):
#     imgs = glob.glob(os.path.join(data_root, 'JPEGImages/*/*.jpg'))
#     print('Total images: {}'.format(len(imgs)))
#     videos = []
#     for file in os.listdir(os.path.join(data_root, 'JPEGImages')):
#         if os.path.isdir(os.path.join(data_root, 'JPEGImages', file)):
#             videos.append(file)
#     print('Total videos: {}'.format(len(videos)))
#
#     v_frames = {}
#     for video in videos:
#         frames = os.listdir(os.path.join(data_root, 'JPEGImages', video))
#         frames = [frame.split('.')[0] for frame in frames]
#         frames.sort(key=int)
#         v_frames.setdefault(video, frames)
#
#     return v_frames


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = logging.FileHandler(filename, mode='w')
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


if __name__ == '__main__':
    args = parse_args()
    DATA_ROOT = args.davis
    MODEL_PATH = '/workspace/solo/code/user_data/model_data/dyy_ckpt_124e.pth'

    CONFIG_FILE = r'/workspace/solo/code/cfg/aug_solov2_r101.py'
    CKPT_FILE = r'/workspace/solo/code/user_data/model_data/solov2_9cls.pth'

    TARGET_SHAPE = (1008, 560)
    SCORE_THR = 0.8
    MAX_NUM = 8


    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    GPU = args.gpu
    intervals = list(range(int(args.interval.split(',')[0]), int(args.interval.split(',')[1]) + 1))
    epoch_per_interval = args.epoch_per_interval

    model_name = args.model
    if model_name == 'motion':
        from STM.models.model_fusai import STM
    elif model_name == 'aspp':
        from STM.models.model_fusai_aspp import STM
    elif model_name == 'solo':
        from STM.models.model_fusai_solo import STM, Run_video

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

    # val_dataset = DAVIS(DAVIS_ROOT, phase='val', imset='tianchi_val_cf.txt', resolution='480p',
    #                     separate_instance=True, only_single=False, target_size=(864, 480))
    val_dataset = TIANCHI(DAVIS_ROOT, phase='val', imset='tianchi_val_cf.txt', separate_instance=True,
                          target_size=(864, 480), same_frames=False)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = nn.DataParallel(STM())

    if torch.cuda.is_available():
        model.cuda()

    # load weights.pth
    if args.load_from and not args.resume_from:
        print('load pretrained from:', args.load_from)
        model.load_state_dict(torch.load(args.load_from), strict=False)

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
        # resume_from
        if args.resume_from:
            print('resume from:', args.resume_from)
            ckpt = torch.load(args.resume_from)
            model.load_state_dict(ckpt['state_dict'], strict=False)
            # epoch_start = ckpt['epoch'] + 1
            # if 'lr' in ckpt.keys():
            #     lr = ckpt['lr']
            if 'optimizer' in ckpt.keys():
                optimizer.load_state_dict(ckpt['optimizer'])

        # run train
        end_epochs = epoch_per_interval
        for i in intervals:
            log.logger.info('Training interval:{}'.format(i))
            # prepare training data
            train_dataset = TIANCHI(DAVIS_ROOT, phase='train', imset='tianchi_train_cf.txt', separate_instance=True,
                                    only_single=False, target_size=(864, 480), clip_size=clip_size, mode='sequence',
                                    interval=i)
            train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                           pin_memory=True)

            train(args, optimizer, train_loader, model, end_epochs, epoch_start, lr)
            epoch_start += epoch_per_interval
            end_epochs += epoch_per_interval
