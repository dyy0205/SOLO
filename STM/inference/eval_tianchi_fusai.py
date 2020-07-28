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

### My libs
from STM.models.model import STM
from STM.dataset import TIANCHI

torch.set_grad_enabled(False)  # Volatile


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
        logits, _, _ = model([Fs[:, :, t], this_keys_m, this_values_m])  # B 2 h w
        em = F.softmax(logits, dim=1)[:, 1]  # B h w
        Es[:, 0, t] = em

        # update key and value
        if t - 1 in to_memorize:
            keys, values = this_keys_m, this_values_m

    pred = torch.round(Es.float())

    return pred, Es


def blend_results(tmp_dir, merge_dir, data_dir):
    img_root = os.path.join(data_dir, 'JPEGImages')
    ann_root = os.path.join(data_dir, 'Annotations')
    with open(os.path.join(data_dir, 'ImageSets/test.txt'), 'r') as f:
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
        ann_path = os.path.join(ann_root, name, '00000.png')
        mask_f = Image.open(ann_path)
        w, h = mask_f.size
        palette = mask_f.getpalette()
        ins = [ins for ins in ins_lst if ins.startswith(name)]
        print(i, name, len(ins))

        video_dir = os.path.join(merge_dir, name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if len(ins) == 1:
            for t in range(num_frames):
                path = os.path.join(tmp_dir, name + '_1', '{:05d}.png'.format(t))
                mask = Image.open(path).convert('P').resize((w, h))
                mask.putpalette(palette)
                mask.save(os.path.join(video_dir, '{:05d}.png'.format(t)))
        else:
            for t in range(num_frames):
                mask = np.zeros((h, w), dtype=np.uint8)
                for j in range(1, len(ins) + 1):
                    path = os.path.join(tmp_dir, name + '_{}'.format(j), '{:05d}.png'.format(t))
                    temp = np.array(Image.open(path).convert('P').resize((w, h)), dtype=np.uint8)
                    temp[temp == 1] = j
                    mask += temp
                    mask[mask > j] = j
                # print(len(ins), np.unique(mask))
                mask = Image.fromarray(mask)
                mask.putpalette(palette)
                mask.save(os.path.join(video_dir, '{:05d}.png'.format(t)))


def zip_result(result_dir, save_path):
    f = zipfile.ZipFile(os.path.join(save_path, 'submit.zip'), 'w', zipfile.ZIP_DEFLATED)
    for dir_path, dir_name, file_names in os.walk(result_dir):
        file_path = dir_path.replace(result_dir, '')
        file_path = file_path and file_path + os.sep or ''
        for file_name in file_names:
            f.write(os.path.join(dir_path, file_name), file_path + file_name)
    f.close()


def main():
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on TIANCHI')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    template_mask = glob.glob(os.path.join(DATA_ROOT, 'Annotations/*/00001.png'))[0]
    palette = Image.open(template_mask).getpalette()

    Testset = TIANCHI(DATA_ROOT, imset='test.txt', single_object=True)
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
        test_path = os.path.join(TMP_PATH, seq_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        for f in range(num_frames):
            img_E = Image.fromarray(pred[0, 0, f].cpu().numpy().astype(np.uint8))
            img_E.putpalette(palette)
            img_E = img_E.resize(ori_shape[::-1])
            img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))


def mask_inference(data_dir, config, ckpt, out_dir):
    # build the model from a config file and a checkpoint file
    model = init_detector(config, ckpt, device='cuda:0')

    # test a single image
    imgs = glob.glob(os.path.join(data_dir, 'JPEGImages/*/00001.jpg'))
    generate_imagesets(data_dir, imgs)
    for img in imgs:
        # img = '/workspace/solo/test/00001.jpg'
        result, cost_time = inference_detector(model, img)
        result = filter_result(result)
        save_mask(img, result, 0.1, out_dir)


def generate_imagesets(data_dir, imgs):
    videos = [img.split('/')[-2] for img in imgs]
    videos = np.unique(videos)
    if not os.path.exists(os.path.join(data_dir, 'ImageSets')):
        os.makedirs(os.path.join(data_dir, 'ImageSets'))
    with open(os.path.join(data_dir, 'ImageSets/test.txt'), 'w') as f:
        for v in videos:
            f.write(v)
            f.write('\n')


def filter_result(result, index=0, max_num=8):
    assert isinstance(result, list)
    rr = []
    for r in result:
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

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    _, h, w = seg_label.shape
    img_show = np.zeros((h, w, 3)).astype(np.uint8)
    for idx in range(num_mask):
        idx = -(idx + 1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = color_mask

    img_s = Image.fromarray(img_show).convert('P')
    # mmcv.imwrite(save_path, out_dir)
    img_s.save(save_path)


if __name__ == '__main__':
    GPU = '0'
    DATA_ROOT = '/workspace/solo/test/0728'
    MODEL_PATH = '/workspace/STM_test/user_data/model_data/model.pth'
    SAVE_PATH = '/workspace/STM_test/prediction_result'
    TMP_PATH = '/workspace/STM_test/user_data/tmp_data'
    MERGE_PATH = '/workspace/STM_test/user_data/merge_data'
    MASK_PATH = os.path.join(DATA_ROOT, 'Annotations')

    CONFIG_FILE = r'../../cfg/aug_solov2_r101_imgaug.py'
    CKPT_FILE = r'/workspace/solo/workdir/solov2_r101_ssim.pth'

    mask_inference(DATA_ROOT, CONFIG_FILE, CKPT_FILE, MASK_PATH)
    main()
    blend_results(TMP_PATH, MERGE_PATH, DATA_ROOT)
    zip_result(MERGE_PATH, SAVE_PATH)
