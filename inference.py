from mmdet.apis import init_detector, inference_detector, show_result_ins, save_sem_mask
import mmcv
import numpy as np
import os, glob, time
from PIL import Image

config_file = 'ade_cfg/solov2_r101_dcn_22.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './indoor_dcn.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:2')
# CLASSES = ('person', 'cat', 'dog', 'cartoon', 'horse', 'sheep', 'cow', 'car', 'airplane')

# imgs = ['./plane_estimation/rgb_00535.jpg']
imgs = sorted(glob.glob('./nyu_images/*.jpg'))
# imgs = sorted(glob.glob('/versa/dyy/dataset/ADE/panoptic/val/*.jpg'))
save_dir = './nyu_images_out'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

total = 0
for i, img in enumerate(imgs):
    name = img.split('/')[-1]
    result, cost_time = inference_detector(model, img)
    print(i, name, cost_time)
    total += cost_time
    try:
        show_result_ins(img, result, model.CLASSES, score_thr=0.3,
                    out_file=os.path.join(save_dir, name))
        # save_sem_mask(img, result, out_file=os.path.join(save_dir, name.replace('.jpg', '.png')))
    except:
        continue
print('average cost time: ', total / len(imgs))