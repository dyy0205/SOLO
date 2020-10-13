from mmdet.apis import init_detector, inference_detector, show_result_ins, save_sem_mask
import mmcv
import os, glob, time

config_file = 'ade_cfg/solov2_r101_22.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './work_dirs/ade_22/epoch_36.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# CLASSES = ('person', 'cat', 'dog', 'cartoon', 'horse', 'sheep', 'cow', 'car', 'airplane')

# # test a single image
# img = './WechatIMG21.png'
# result, cost_time = inference_detector(model, img)
# # show_result_ins(img, result, model.CLASSES, score_thr=0.2,
# #                     out_file='./WechatIMG21_out.png')
# save_sem_mask(img, result, out_file='./WechatIMG21_sem.png')

# imgs = glob.glob('./nyu_images/*.jpg')
imgs = sorted(glob.glob('/versa/dyy/dataset/ADE/panoptic/val/*.jpg'))
save_dir = './ade_22'
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