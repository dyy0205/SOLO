from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os, glob, time

config_file = './cfg/aug_solov2_r101_imgaug.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/workspace/solo/workdir/solov2_r101_ssim.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '/workspace/solo/test/00001.jpg'
result, cost_time = inference_detector(model, img)
show_result_ins(img, result, model.CLASSES, score_thr=0.14,
                    out_file='/workspace/solo/result/00001_out.jpg')

# imgs = glob.glob('./test_imgs/*.*')
# # imgs = glob.glob('/home/versa/dataset/MSCOCO/aug_seg/val_imgs/*.*')
# save_dir = './aug_solov2_r101_imgaug'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# total = 0
# for i, img in enumerate(imgs):
#     name = img.split('/')[-1]
#     if not name.endswith('HEIC'):
#         result, cost_time = inference_detector(model, img)
#         print(i, name, cost_time)
#         total += cost_time
#         try:
#             show_result_ins(img, result, model.CLASSES, score_thr=0.15,
#                         out_file=os.path.join(save_dir, name))
#         except:
#             continue
# print('average cost time: ', total / len(imgs))