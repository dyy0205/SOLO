from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os, glob, time

config_file = 'cfg/solo_lite3_bifpn.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './work_dirs/solo_lite3_bifpn/epoch_5.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = './WechatIMG14.jpeg'
result, cost_time = inference_detector(model, img)
show_result_ins(img, result, model.CLASSES, score_thr=0.25,
                    out_file='./WechatIMG14_out.jpeg')

# imgs = glob.glob('../coco/test/*.*')
# save_dir = '../coco/contour'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# total = 0
# for i, img in enumerate(imgs):
#     name = img.split('/')[-1]
#     result, cost_time = inference_detector(model, img)
#     print(i, name, cost_time)
#     total += cost_time
#     try:
#         show_result_ins(img, result, model.CLASSES, score_thr=0.25,
#                     out_file=os.path.join(save_dir, name))
#     except:
#         continue
# print('average cost time: ', total / len(imgs))