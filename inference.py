from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import pycocotools.mask as maskUtils
import numpy as np
import mmcv, cv2
import os, glob, time

config_file = './cfg/aug_solov2_r101.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'solov2_r101_tuned.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = './versa-car-2209.jpg'
result, cost_time = inference_detector(model, img)
show_result_ins(img, result, model.CLASSES, score_thr=0.3,
                    out_file='./2209_out.png')

# imgs = glob.glob('../coco/new_test/*.*')
# save_dir = '../coco/new_test_out'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# total = 0
# for i, img in enumerate(imgs):
#     name = img.split('/')[-1]
#     if not name.endswith('HEIC'):
#         result, cost_time = inference_detector(model, img)
#         print(i, name, cost_time)
#         total += cost_time
#         try:
#             show_result_ins(img, result, model.CLASSES, score_thr=0.3,
#                     out_file=os.path.join(save_dir, name))
#         except:
#             continue
# print('average cost time: ', total / len(imgs))


# tianchi_root = '/versa/dataset/TIANCHI/tianchiyusai/'
# with open(os.path.join(tianchi_root, 'ImageSets/test.txt'), 'r') as f:
#     test = f.readlines()
#
# for video_id in test:
#     video_id = video_id.strip()
#     save_dir = os.path.join('tianchi_out', video_id)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     print('Start evaluating {}...'.format(video_id))
#     imgs = glob.glob(os.path.join(tianchi_root, 'JPEGImages', video_id, '*.jpg'))
#     for i, img in enumerate(sorted(imgs)):
#         name = img.split('/')[-1]
#         result, cost_time = inference_detector(model, img)
#         try:
#             show_result_ins(img, result, model.CLASSES, score_thr=0.3,
#                             out_file=os.path.join(save_dir, name))
#         except:
#             continue
