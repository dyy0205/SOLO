from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
# import visdom
import sys
from tensorboardX import SummaryWriter

### My libs
from dataloader.dataset_rgmp_v1 import DAVIS
from models.model import STM
from models.loss.smooth_cross_entropy_loss import SmoothCrossEntropyLoss
from models.loss.dice_loss import DiceLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a tracker')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument("-g", type=str, default='0', help="0; 0,1; 0,3; etc")
    parser.add_argument("-s", type=str, default='val', help="set")
    parser.add_argument("-y", type=int, default=16, help="year")
    parser.add_argument("-D", type=str, help="path to data", default='/cfs/mazhongke/databases/DAVIS2017/')
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


def Run_video(Fs, Ms, num_frames, name, Mem_every=None, Mem_number=None):
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
        # print('logits',logits.shape)
        # print(logits.shape)
        em = F.softmax(logits, dim=1)[:, 1] # B h w
        Es[:,0,t] = em

        #  calculate loss on cuda
        Ms_cuda = Ms[:,0,t].cuda()
        # try:
        loss_video += _loss(logits, Ms_cuda)
        # except:
        #     print('error:', name )

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m
    # np.save('Es.npy', Es.detach().cpu().numpy())
    #save mask
    # save_img_path = os.path.join('./masks', name[0])
    # if not os.path.exists(save_img_path):
    #     os.makedirs(save_img_path)
    # for i in range(len(Es[0,0])):
    #     img_np = Es[0,0,i].detach().cpu().numpy()
    #     img_np = (np.round(img_np*255)).astype(np.uint8)
    #     img = Image.fromarray(img_np).convert('L')
    #     img.save(save_img_path+'/'+'{:05d}.png'.format(i))

    #  calculate mIOU on cuda
    pred = torch.round(Es.float().cuda())
    video_mIoU = 0
    for n in range(len(Ms)):  # Nth batch
        video_mIoU = video_mIoU + get_video_mIoU(pred[n], Ms[n].cuda())  # mIOU of video(t frames) for each batch
    video_mIoU = video_mIoU / len(Ms)  # mean IoU among batch


    return loss_video/num_frames, video_mIoU

def validate(val_loader, model):
    model.eval()  # turn-off BN
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    code_name = '{}_DAVIS_{}{}'.format(MODEL, YEAR, SET)
    print('Start Testing:', code_name)

    loss_all_videos = 0.0
    miou_all_videos = 0.0
    for seq, batch in enumerate(val_loader):
        Fs, Ms, info = batch['Fs'], batch['Ms'], batch['info']
        num_frames = info['num_frames'][0].item()
        # print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, 1))
        # error_nums = 0
        with torch.no_grad():
            name = info['name']
            # try:
            loss_video, video_mIou = Run_video(Fs, Ms, num_frames, name, Mem_every=5, Mem_number=None)
            loss_all_videos += loss_video
            miou_all_videos += video_mIou
            print('loss_video:', loss_video, 'video_mIou:', video_mIou)
            # except:
            #     error_nums += 1
            #     print('!!!!!!!!!!!error:', name)


    loss_all_videos /= len(val_loader)
    miou_all_videos /= len(val_loader)

    print('loss_all_video:', loss_all_videos)
    print('miou_all_videos:', miou_all_videos)

    return loss_all_videos, miou_all_videos



def train(train_loader, model, writer):
    # visdom
    # vis = visdom.Visdom(env='stm davis16 v4.1')

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Training on DAVIS')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    code_name = '{}_DAVIS_{}{}'.format(MODEL, YEAR, SET)
    print('Start Training:', code_name)

    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.99))
    optimizer = torch.optim.SGD(model.parameters(), base_lr, momentum=0.9, weight_decay=1e-4)

    epochs = 30
    se = 0
    lr_cos = lambda n: 0.5 * (1 + np.cos((n - se) / (epochs - se) * np.pi)) * base_lr

    best_loss = 10.0
    for epoch in range(epochs):
        lr = lr_cos(epoch)

        # validate
        # try:
        loss_val, miou_val = validate(val_loader, model)

        # draw loss and mIOU in visdom
        # vis.line(X=[epoch], Y=[loss_val.detach().cpu().numpy()], win='loss_val', update='append')
        # vis.line(X=[epoch], Y=[miou_val], win='mIOU_val', update='append')

        writer.add_scalar('val/Loss', loss_val.detach().cpu().numpy(), epoch)
        writer.add_scalar('val/miou', miou_val, epoch)

        # save checkpoints
        # state_dict = model.state_dict()
        # torch.save({
        #     'epoch': epoch,
        #     'state_dict': state_dict,
        #     'lr': lr,
        #     'cur_val_loss': loss_val,
        #     'cur_val_miou': miou_val
        # }, os.path.join("/cfs/yuanlei/stm_train/ckpt/youtube", 'ckpt_%04d.ckpt' % epoch, ))


        # except:
        #     print('error in val at epoch:', epoch)


        model.train()  # turn-on BN

        for seq, batch in enumerate(train_loader):
            Fs, Ms, info = batch['Fs'], batch['Ms'], batch['info']
            num_frames = info['num_frames'][0].item()
            # print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, 1))
            name = info['name']
            loss_video, video_mIou = Run_video(Fs, Ms, num_frames, name, Mem_every=5, Mem_number=None)
            print('loss_video:', loss_video)

            # backward
            optimizer.zero_grad()
            loss_video.backward()
            optimizer.step()

            print('loss_video:', loss_video, 'video_mIou:', video_mIou)

            # draw loss and mIOU in visdom
            y1 = loss_video.detach().cpu().numpy()
            # vis.line(X=[epoch*len(train_loader)+seq], Y=[y1], win='loss', update='append')
            # vis.line(X=[epoch*len(train_loader)+seq], Y=[video_mIou], win='mIOU', update='append')
            writer.add_scalar('train/loss', y1, epoch*len(train_loader)+seq)
            writer.add_scalar('train/miou', video_mIou, epoch*len(train_loader)+seq)


if __name__ == '__main__':
    args = parse_args()
    writer = SummaryWriter('run_test')

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D

    #prepare data
    clip_size = 8
    iou_ignore_bg = True
    BATCH_SIZE = 1
    base_lr = 1e-4 # 1e-4

    DAVIS_ROOT = '/cfs/mazhongke/databases/DAVIS2017/'
    YOUTUBE_ROOT = '/cfs/dataset/youtube_complete/'
    palette = Image.open(DAVIS_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()

    val_dataset = DAVIS(DAVIS_ROOT, phase='val', imset='2016/val.txt', resolution='480p', separate_instance=False, only_single=False, target_size=(864, 480))
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    train_dataset = DAVIS(YOUTUBE_ROOT, phase='train', imset='train.txt', resolution='480p', separate_instance=True,
                          only_single=False, target_size=(864, 480), clip_size=clip_size)
    # train_dataset = DAVIS(DAVIS_ROOT, phase='train', imset='2016/val.txt', resolution='480p', separate_instance=True,
    #                       only_single=False, target_size=(864, 480), clip_size=clip_size)
    train_loader = data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)

    # build model
    model = nn.DataParallel(STM())
    if torch.cuda.is_available():
        model.cuda()

    # load weights.pth
    print('load pretrained')
    model.load_state_dict(torch.load("/cfs/yuanlei/STM/STM_weights.pth"), strict=True)

    if args.s == "val":
        # run val
        with torch.no_grad():
            loss_val, miou_val = validate(val_loader, model)
            print('loss_val_all:', loss_val)
            print('miou_val_all:', miou_val)

    elif args.s == "train":
        # run train
        train(train_loader, model, writer)

