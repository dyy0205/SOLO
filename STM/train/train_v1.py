from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys
from tensorboardX import SummaryWriter

### My libs
from dataloader.dataset_rgmp_v1 import DAVIS
from dataloader.train_stm_stage1 import YOUTUBE
from models.model import STM
from models.loss.smooth_cross_entropy_loss import SmoothCrossEntropyLoss
from models.loss.dice_loss import DiceLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a tracker')

    parser.add_argument('--work_dir', type=str, default='./exp/test', help='the dir to save models.pth and logs and masks')
    parser.add_argument("--mode", type=str, default='train', help="train or val")
    parser.add_argument('--load_from', type=str, default='')
    # train
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--train_data', type=str, default='youtube')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument("--davis", type=str, default='/cfs/mazhongke/databases/DAVIS2017/')
    parser.add_argument("--youtube", type=str, default='/cfs/dataset/youtube_complete/')
    parser.add_argument("--gpu", type=str, default='0', help="0; 0,1; 0,3; etc")
    parser.add_argument("--year", type=int, default=2016, help="year")
    # val
    parser.add_argument("--save_masks", type=bool, default=True, help='whether save predicting mask when mode is val')

    return parser.parse_args()

_ce_loss = SmoothCrossEntropyLoss(eps=1e-3)
_dice_loss = DiceLoss(ignore_index=0)

def _loss(x, y):
    return _ce_loss(x, y) + _dice_loss(x, y)

def get_video_mIoU(predn,all_Mn):#[c,t,h,w]
    pred = predn.squeeze().cpu().data.numpy()
    # np.save('blackswan.npy', pred)
    gt = all_Mn.squeeze().cpu().data.numpy()#[t,h,w]
    agg = pred + gt
    i = float(np.sum(agg == 2))
    u = float(np.sum(agg > 0))
    return i / (u+1e-6)


def Run_video(args, Fs, Ms, num_frames, name, Mem_every=None, Mem_number=None):
    print('name:', name)
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms).float().cuda()#[1,1,50,480,864][b,c,t,h,w]
    Es[:, :, 0] = Ms[:, :, 0]

    loss_video = torch.tensor(0.0).cuda()

    # initialize the size of pre_value and pre_key
    key, value = model([Fs[:, :, 0], Ms[:, :, 0].float()])
    [b,c,h,w] = key.shape
    pre_key = torch.zeros([b,c,1,h,w])
    pre_value = torch.zeros([b,c*4,1,h,w])

    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        preFs = (Fs[:, :, t - 1]).cuda()
        preEs = (Es[:, :, t - 1]).float().cuda()
        pre_key[:,:,0], pre_value[:,:,0]= model([preFs, preEs])

        if t - 1 == 0:  # the first frame
            this_keys_m, this_values_m = pre_key, pre_value
        else:  # other frame
            this_keys_m = torch.cat([keys, pre_key], dim=2)
            this_values_m = torch.cat([values, pre_value], dim=2)


        # segment
        logits = model([Fs[:, :, t], this_keys_m, this_values_m]) # B 2 h w
        em = F.softmax(logits, dim=1)[:, 1] # B h w
        Es[:,0,t] = em

        #  calculate loss on cuda
        Ms_cuda = Ms[:,0,t].cuda()
        loss_video += _loss(logits, Ms_cuda)

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

    if args.save_masks and args.mode=='val':
        #save mask
        save_img_path = os.path.join(args.work_dir, 'masks', name[0])
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        for i in range(len(Es[0,0])):
            img_np = Es[0,0,i].detach().cpu().numpy()
            img_np = (np.round(img_np*255)).astype(np.uint8)
            img = Image.fromarray(img_np).convert('L')
            img.save(save_img_path+'/'+'{:05d}.png'.format(i))

    #  calculate mIOU on cuda
    pred = torch.round(Es.float().cuda())
    video_mIoU = 0
    for n in range(len(Ms)):  # Nth batch
        video_mIoU = video_mIoU + get_video_mIoU(pred[n], Ms[n].cuda())  # mIOU of video(t frames) for each batch
    video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch


    return loss_video/num_frames, video_mIoU

def validate(args, val_loader, model):
    model.eval()  # turn-off BN
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS', args.year)

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    loss_all_videos = 0.0
    miou_all_videos = 0.0
    for seq, batch in enumerate(val_loader):
        Fs, Ms, info = batch['Fs'], batch['Ms'], batch['info']
        num_frames = info['num_frames'][0].item()
        # error_nums = 0
        with torch.no_grad():
            name = info['name']
            loss_video, video_mIou = Run_video(args, Fs, Ms, num_frames, name, Mem_every=5, Mem_number=None)
            loss_all_videos += loss_video
            miou_all_videos += video_mIou
            print('loss_video:', loss_video, 'video_mIou:', video_mIou)

    loss_all_videos /= len(val_loader)
    miou_all_videos /= len(val_loader)

    return loss_all_videos, miou_all_videos



def train(args, train_loader, model, writer):
    MODEL = 'STM'
    print(MODEL, 'Training on ', args.train_data)

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    code_name = '{}_DAVIS_{}{}'.format(MODEL, args.train_data, YEAR)
    print('Start Training:', code_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

    epochs = args.epoch
    se = 0
    lr_cos = lambda n: 0.5 * (1 + np.cos((n - se) / (epochs - se) * np.pi)) * args.lr

    for epoch in range(epochs):
        lr = lr_cos(epoch)

        # validate
        loss_val, miou_val = validate(args, val_loader, model)

        # write loss and mIOU into tensorboard
        writer.add_scalar('val/Loss', loss_val.detach().cpu().numpy(), epoch)
        writer.add_scalar('val/miou', miou_val, epoch)
        print('***lr', lr)
        # save checkpoints
        ckpt_dir = os.path.join(args.work_dir, "ckpt", args.train_data)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
            'lr': lr,
            'cur_val_loss': loss_val,
            'cur_val_miou': miou_val
        }, os.path.join(ckpt_dir, 'ckpt_%04d.ckpt' % epoch, ))


        model.train()  # turn-on BN

        for seq, batch in enumerate(train_loader):
            Fs, Ms, info = batch['Fs'], batch['Ms'], batch['info']
            num_frames = info['num_frames'][0].item()
            name = info['name']
            loss_video, video_mIou = Run_video(args, Fs, Ms, num_frames, name, Mem_every=5, Mem_number=None)
            print('loss_video:', loss_video)

            # backward
            optimizer.zero_grad()
            loss_video.backward()
            optimizer.step()

            print('loss_video:', loss_video, 'video_mIou:', video_mIou)

            # write into tensorboardX
            y1 = loss_video.detach().cpu().numpy()
            writer.add_scalar('train/loss', y1, epoch*len(train_loader)+seq)
            writer.add_scalar('train/miou', video_mIou, epoch*len(train_loader)+seq)


if __name__ == '__main__':
    args = parse_args()
    writer = SummaryWriter(args.work_dir+'/runs')

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    GPU = args.gpu
    YEAR = args.year

    #prepare val data
    DAVIS_ROOT = args.davis
    palette = Image.open(DAVIS_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()

    val_dataset = DAVIS(DAVIS_ROOT, phase='val', imset=str(args.year)+'/val.txt', resolution='480p', separate_instance=False, only_single=False, target_size=(864, 480))
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # build model
    model = nn.DataParallel(STM())
    if torch.cuda.is_available():
        model.cuda()

    # load weights.pth
    if args.load_from:
        print('load pretrained from:', args.load_from)
        try:
            model.load_state_dict(torch.load(args.load_from), strict=True)
        except:
            model.load_state_dict(torch.load(args.load_from)['state_dict'], strict=True)

    if args.mode == "val":
        # run val
        with torch.no_grad():
            loss_val, miou_val = validate(args, val_loader, model)
            print('loss_val_all:', loss_val)
            print('miou_val_all:', miou_val)

    elif args.mode == "train":
        # set training para
        clip_size = args.clip_size
        BATCH_SIZE = args.batch_size
        base_lr = args.lr  # 1e-4

        # prepare training data
        if args.train_data == 'youtube':
            YOUTUBE_ROOT = args.youtube
            train_dataset = YOUTUBE(YOUTUBE_ROOT, phase='train', imset='train.txt', resolution='480p',
                                  separate_instance=True,
                                  only_single=False, target_size=(864, 480), clip_size=clip_size)
        elif args.train_data == 'davis':
            train_dataset = DAVIS(DAVIS_ROOT, phase='train', imset=str(args.year) + '/val.txt', resolution='480p',
                                  separate_instance=True,
                                  only_single=False, target_size=(864, 480), clip_size=clip_size)
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

        # run train
        train(args, train_loader, model, writer)

