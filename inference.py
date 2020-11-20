from mmdet.apis import init_detector, inference_detector, show_result_ins, save_sem_mask
import mmcv
import numpy as np
import os, glob, time
from PIL import Image

config_file = 'ade_cfg/solov2_r101_dcn_22.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './indoor_dcn.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:1')
# CLASSES = ('person', 'cat', 'dog', 'cartoon', 'horse', 'sheep', 'cow', 'car', 'airplane')

# # test a single image
# img = './WechatIMG21.png'
# result, cost_time = inference_detector(model, img)
# # show_result_ins(img, result, model.CLASSES, score_thr=0.2,
# #                     out_file='./WechatIMG21_out.png')
# save_sem_mask(img, result, out_file='./WechatIMG21_sem.png')

imgs = ['./plane_estimation/rgb_00535.jpg']
# imgs = sorted(glob.glob('./val_imgs/*.jpg'))
# imgs = sorted(glob.glob('/versa/dyy/dataset/ADE/panoptic/val/*.jpg'))
save_dir = './val_imgs_out'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# total = 0
# for i, img in enumerate(imgs):
#     name = img.split('/')[-1]
#     result, cost_time = inference_detector(model, img)
#     print(i, name, cost_time)
#     total += cost_time
#     try:
#         show_result_ins(img, result, model.CLASSES, score_thr=0.3,
#                     out_file=os.path.join(save_dir, name))
#         # save_sem_mask(img, result, out_file=os.path.join(save_dir, name.replace('.jpg', '.png')))
#     except:
#         continue
# print('average cost time: ', total / len(imgs))

for i, img in enumerate(imgs):
    name = img.split('/')[-1]
    result, _ = inference_detector(model, img)
    print(i, name)

    # save_path = os.path.join(save_dir, name)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # cur_result = result[0]
    # if cur_result is not None:
    #     masks = cur_result[0].cpu().numpy().astype(np.uint8)
    #     classes = cur_result[1].cpu().numpy()
    #     scores = cur_result[2].cpu().numpy()
    #     h, w = masks[0].shape
    #
    #     vis_inds = (scores > 0.3)
    #     masks = masks[vis_inds]
    #
    #     areas = [mask.sum() for mask in masks]
    #     sorted_inds = np.argsort(areas)[::-1]
    #     keep_inds = []
    #     for i in sorted_inds:
    #         overlap = False
    #         # 根据面积重合度, 小面积的被舍弃，不考虑类别
    #         if i != 0:
    #             for j in range(i):
    #                 if np.sum((masks[i, :, :] > 0) * (masks[j, :, :] > 0)) / np.sum(masks[j, :, :] > 0) > 0.85:
    #                     overlap = True
    #                     break
    #         keep_inds.append(i)
    #     masks = masks[keep_inds]
    #     classes = classes[keep_inds]
    #     scores = scores[keep_inds]

        # for i in range(len(masks)):
        #     Image.fromarray(masks[i] * 255).save(
        #         os.path.join(save_path, name.replace('.png', f'_{classes[i]}_{round(scores[i], 3)}.png')))

    show_result_ins(img, result, model.CLASSES, score_thr=0.2,
                    out_file=os.path.join(save_dir, name))