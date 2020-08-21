from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import datetime

### My libs
from dataset import TIANCHI
from models.model3 import STM

torch.set_grad_enabled(False)  # Volatile


def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data", default='/local/DATA')
    parser.add_argument("-M", type=str, help="path to model", default='/local/model')
    return parser.parse_args()


args = get_arguments()

GPU = args.g
SET = args.s
VIZ = args.viz
DATA_ROOT = args.D
MODEL_PATH = args.M

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on TIANCHI')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')

palette = Image.open(DATA_ROOT + '/Annotations/606332/00000.png').getpalette()

def Run_video(Fs, Ms, num_frames, Mem_every=None, Mem_number=None):
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
        logits, p3, p4, r_1, r_2, r_3 = model([Fs[:, :, t], Es[:, :, t - 1], this_keys_m, this_values_m])  # B 2 h w
        em = F.softmax(r_1, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

    pred = torch.round(Es.float())

    return pred, Es

# def Run_video(Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
#     # initialize storage tensors
#     if Mem_every:
#         to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
#     elif Mem_number:
#         to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]]
#     else:
#         raise NotImplementedError
#
#     Es = torch.zeros_like(Ms)
#     Es[:, :, 0] = Ms[:, :, 0]
#
#     for t in tqdm.tqdm(range(1, num_frames)):
#         # memorize
#         with torch.no_grad():
#             prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))
#
#         if t - 1 == 0:  #
#             this_keys, this_values = prev_key, prev_value  # only prev memory
#         else:
#             this_keys = torch.cat([keys, prev_key], dim=3)
#             this_values = torch.cat([values, prev_value], dim=3)
#
#         # segment
#         with torch.no_grad():
#             logit = model(Fs[:, :, t], this_keys, this_values, torch.tensor([num_objects]))
#         Es[:, :, t] = F.softmax(logit, dim=1)
#
#         # update
#         if t - 1 in to_memorize:
#             keys, values = this_keys, this_values
#
#     pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
#     return pred, Es


Testset = TIANCHI(DATA_ROOT, imset='test.txt', single_object=True)
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

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

code_name = 'tianchi'
date = datetime.datetime.strftime(datetime.datetime.now(), '%y%m%d%H%M')
print('Start Testing:', code_name)

for seq, V in enumerate(Testloader):
    Fs, Ms, info = V
    seq_name = info['name'][0]
    ori_shape = info['ori_shape']
    num_frames = info['num_frames'][0].item()
    print('[{}]: num_frames: {}'.format(seq_name, num_frames))

    pred, Es = Run_video(Fs, Ms, num_frames, Mem_every=5, Mem_number=None)

    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', date, code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[0, 0, f].cpu().numpy().astype(np.uint8))
        img_E.putpalette(palette)
        img_E = img_E.resize(ori_shape[::-1])
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

    if VIZ:
        from helpers import overlay_davis

        # visualize results #######################
        viz_path = os.path.join('./viz/', code_name, seq_name)
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        for f in range(num_frames):
            pF = (Fs[0, :, f].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(viz_path, 'f{}.jpg'.format(f)))

        vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
        frame_path = os.path.join('./viz/', code_name, seq_name, 'f%d.jpg')
        os.system(
            'ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(
                frame_path, vid_path))
