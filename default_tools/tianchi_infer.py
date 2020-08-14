from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import numpy as np
from PIL import Image
import os, glob, time
from skimage.morphology import remove_small_objects


# config_file = '../cfg/aug_solov2_r101_imgaug.py'
config_file = '../cfg/aug_solov2_r101.py'
# checkpoint_file = '../work_dirs/aug_solov2_r101_tuned_ssim/epoch_12.pth'
checkpoint_file = '../work_dirs/add_solov2_r101_from_coco_car2/epoch_12.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

root = '/versa/dyy/dataset/TIANCHI'
imgs = sorted(glob.glob(os.path.join(root, 'val_total/JPEGImages/*/*.jpg')))
palette = Image.open(os.path.join(root, 'val_total/Annotations/606396/00001.png')).getpalette()

for i, img in enumerate(imgs):
    name = img.split('/')[-1]
    video_id = img.split('/')[-2]
    result, _ = inference_detector(model, img)
    print(i, video_id + '_' + name)
    cur_result = result[0]
    if cur_result is not None:
        masks = cur_result[0].cpu().numpy().astype(np.uint8)
        classes = cur_result[1].cpu().numpy()
        scores = cur_result[2].cpu().numpy()
        h, w = masks[0].shape

        vis_inds = (scores > 0.5) & (classes == 0)
        masks = masks[vis_inds]

        areas = [mask.sum() for mask in masks]
        sorted_inds = np.argsort(areas)[::-1]
        keep_inds = []
        for i in sorted_inds:
            overlap = False
            # 根据面积重合度, 小面积的被舍弃，不考虑类别
            if i != 0:
                for j in range(i):
                    if np.sum((masks[i, :, :] > 0) * (masks[j, :, :] > 0)) / np.sum(masks[j, :, :] > 0) > 0.85:
                        overlap = True
                        break
            keep_inds.append(i)
        masks = masks[keep_inds]

        blend = np.zeros((h, w), dtype=np.uint8)
        if len(masks) > 0:
            for i, mask in enumerate(masks):
                mask = remove_small_objects(mask > 0,
                                            min_size=mask[mask > 0].size // 20,
                                            connectivity=1).astype(np.uint8)
                mask[mask == 1] = i+1
                blend += mask
                blend[blend > (i+1)] = i+1
            blend = Image.fromarray(blend)
            blend.putpalette(palette)
        else:
            blend = Image.fromarray(blend)
        save_dir = os.path.join(root, 'v3_out', video_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        blend.save(os.path.join(save_dir, name.replace('.jpg', '.png')))
