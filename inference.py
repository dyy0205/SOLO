from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os, glob, time

config_file = 'cfg/aug_solov2_r101.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './work_dirs/add_solov2_r101_from_coco_car2/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:3')
# CLASSES = ('person', 'cat', 'dog', 'cartoon', 'horse', 'sheep', 'cow', 'car', 'airplane')

# # test a single image
# img = './test1.png'
# result, cost_time = inference_detector(model, img)
# show_result_ins(img, result, model.CLASSES, score_thr=0.2,
#                     out_file='./test1_out.jpeg')

imgs = glob.glob('./test_add_imgs/*.*')
# imgs = glob.glob('/home/versa/dataset/MSCOCO/aug_seg/val_imgs/*.*')
save_dir = './add_out'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

total = 0
for i, img in enumerate(imgs):
    name = img.split('/')[-1]
    if not name.endswith('HEIC'):
        result, cost_time = inference_detector(model, img)
        print(i, name, cost_time)
        total += cost_time
        try:
            show_result_ins(img, result, model.CLASSES, score_thr=0.25,
                        out_file=os.path.join(save_dir, name))
        except:
            continue
print('average cost time: ', total / len(imgs))