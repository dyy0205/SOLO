from mmdet.apis import init_detector, inference_detector
import numpy as np
import os, glob
from PIL import Image


def solo_infer(model, img, conf):
    image_np = np.array(Image.open(img))
    result, _ = inference_detector(model, img)
    cur_result = result[0]
    if cur_result is not None:
        masks = cur_result[0].cpu().numpy().astype(np.uint8)
        classes = cur_result[1].cpu().numpy()
        scores = cur_result[2].cpu().numpy()
        h, w = masks[0].shape

        vis_inds = (scores > conf)
        masks = masks[vis_inds]
        classes = classes[vis_inds]

        areas = [mask.sum() for mask in masks]
        sorted_inds = np.argsort(areas)[::-1]
        keep_inds = []
        for i in sorted_inds:
            if i != 0:
                for j in range(i):
                    if np.sum((masks[i, :, :] > 0) * (masks[j, :, :] > 0)) / np.sum(masks[j, :, :] > 0) > 0.85:
                        break
            keep_inds.append(i)
        masks = masks[keep_inds]
        classes = classes[keep_inds]

        instance_map = np.zeros((h, w), dtype=np.uint8)
        semantic_map = np.zeros((h, w), dtype=np.uint8)
        if masks is not None:
            for i, (mask, cls) in enumerate(zip(masks, classes)):
                instance_map[mask > 0] = i + 1
                semantic_map[mask > 0] = cls + 1
                if cls in [0, 1, 7]:
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask_bool = mask.astype(np.bool)
                    image_np[mask_bool] = image_np[mask_bool] * 0.5 + color_mask * 0.5

        final_mask = np.stack([instance_map, semantic_map], axis=-1)
        return masks, classes, final_mask, image_np


if __name__ == '__main__':
    config_file = 'ade_cfg/solov2_r101_dcn_22.py'
    checkpoint_file = './indoor_dcn.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:2')

    root = '/versa/dyy/dataset/nyu_depth_v2/'
    imgs = sorted(glob.glob(os.path.join(root, 'sync/*/*.jpg')))
    mask_dir = os.path.join(root, '2channels')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    blend_dir = os.path.join(root, 'blend')
    if not os.path.exists(blend_dir):
        os.makedirs(blend_dir)

    total = 0
    for i, img in enumerate(imgs):
        name = img.split('/')[-2] + '_' + img.split('/')[-1]
        print(i, name)
        masks, classes, final_mask, img_blend = solo_infer(model, img, conf=0.2)
        Image.fromarray(final_mask).save(os.path.join(mask_dir, name.replace('.jpg', '.png')))
        Image.fromarray(img_blend).save(os.path.join(blend_dir, name))