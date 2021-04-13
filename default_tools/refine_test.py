from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import numpy as np
from PIL import Image
import os, glob, time
import cv2
import torch
from skimage.morphology import remove_small_objects


bbox_w_scale = 1.2
bbox_h_scale = 1.2
size = (384, 384)

from net.lwef4_softlabel_b3_tc import LWef as human_network
human_ckpt_path = '/home/dingyangyang/SOLO/refine_044.ckpt'
human_checkpoint = torch.load(human_ckpt_path)
human_net = human_network(2, arch='tf_efficientnet_b3', pretrained=False)
human_net.load_state_dict(human_checkpoint['state_dict'], strict=True)
human_net.cuda().eval()


# config_file = '../cfg/solov2_r101_dcn.py'
config_file = '../cfg/aug_solov2_r101.py'
# checkpoint_file = '../solov2_dcn.pth'
checkpoint_file = '../solov2_9cls.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

palette = Image.open('/versa/dyy/dataset/TIANCHI/PreRoundData/Annotations/606396/00001.png').getpalette()
imgs = sorted(glob.glob('/versa/dyy/dataset/TIANCHI/PreRoundData/preTest/000263/*.jpg'))[30:31]


for idx, img_path in enumerate(imgs):
    img = cv2.imread(img_path)
    name = img_path.split('/')[-1]
    video_id = img_path.split('/')[-2]
    result, _ = inference_detector(model, img_path)
    print(idx, video_id + '_' + name)
    cur_result = result[0]
    if cur_result is not None:
        masks = cur_result[0].cpu().numpy().astype(np.uint8)
        classes = cur_result[1].cpu().numpy()
        scores = cur_result[2].cpu().numpy()
        h0, w0 = masks[0].shape

        vis_inds = (scores > 0.2) & (classes == 0)
        masks = masks[vis_inds]

        areas = [mask.sum() for mask in masks]
        sorted_inds = np.argsort(areas)[::-1]
        keep_inds = []
        for i in sorted_inds:
            overlap = False
            # 根据面积重合度, 小面积的被舍弃，不考虑类别
            if i != 0:
                for j in range(i):
                    if np.sum((masks[i, :, :] > 0) * (masks[j, :, :] > 0)) / np.sum(masks[i, :, :] > 0) > 0.8:
                        overlap = True
                        break
            if overlap:
                continue
            keep_inds.append(i)
        # if len(keep_inds) != len(sorted_inds):
        #     print(len(keep_inds), len(sorted_inds))
        masks = masks[keep_inds]

        masks_refine = []
        for i, mask in enumerate(masks):
            x, y, w, h = cv2.boundingRect(mask)
            ori_x = x - w * (bbox_w_scale - 1) / 2
            ori_y = y - h * (bbox_h_scale - 1) / 2
            leftOffset = int(min(ori_x, 0))
            topOffset = int(min(ori_y, 0))
            x = int(ori_x - leftOffset)
            y = int(ori_y - topOffset)
            w = int(min(w0 - x, bbox_w_scale * w))
            h = int(min(h0 - y, bbox_h_scale * h))

            img_crop = img[y:y+h, x:x+w, :]
            mask_crop = mask[y:y+h, x:x+w]
            img_x = cv2.resize(img_crop, size, cv2.INTER_LINEAR)
            mask_x = cv2.resize(mask_crop, size, cv2.INTER_NEAREST)
            input_x = np.concatenate([img_x, mask_x[:, :, np.newaxis]], axis=2)
            input_x = torch.from_numpy(input_x / 255.).permute(2, 0, 1).unsqueeze(0).float()
            pred = human_net(input_x.cuda())
            pred = pred.squeeze().detach().cpu().numpy()
            pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
            cv2.imwrite(f'/versa/dyy/dataset/TIANCHI/PreRoundData/refine_result/000263/{i}.jpg', img_crop)
            alpha_mask = np.tile(mask_crop[:, :, np.newaxis], (1, 1, 3))
            cv2.imwrite(f'/versa/dyy/dataset/TIANCHI/PreRoundData/refine_result/000263/{i}_before.png',
                        alpha_mask*img_crop + (1-alpha_mask)*np.zeros((h,w,3)))
            alpha_mask = np.tile(pred[:, :, np.newaxis], (1, 1, 3))
            cv2.imwrite(f'/versa/dyy/dataset/TIANCHI/PreRoundData/refine_result/000263/{i}_after.png',
                        alpha_mask*img_crop + (1-alpha_mask)*np.zeros((h,w,3)))

            mask_ = np.zeros_like(mask)
            mask_[y:y+h, x:x+w] = pred
            masks_refine.append(mask_)

        masks = np.array(masks_refine)
        # areas = [mask.sum() for mask in masks]
        # sorted_inds = np.argsort(areas)[::-1]
        # keep_inds = []
        # for i in sorted_inds:
        #     overlap = False
        #     # 根据面积重合度, 小面积的被舍弃，不考虑类别
        #     if i != 0:
        #         for j in range(i):
        #             if np.sum((masks[i, :, :] > 0) * (masks[j, :, :] > 0)) / np.sum(masks[j, :, :] > 0) > 0.8:
        #                 overlap = True
        #                 break
        #     if overlap:
        #         continue
        #     keep_inds.append(i)
        # # if len(keep_inds) != len(sorted_inds):
        # #     print(len(keep_inds), len(sorted_inds))
        # masks = masks[keep_inds]

        blend = np.zeros((h0, w0), dtype=np.uint8)
        if len(masks) > 0:
            for i, mask in enumerate(masks):
                # mask = remove_small_objects(mask > 0,
                #                             min_size=mask[mask > 0].size // 50,
                #                             connectivity=1).astype(np.uint8)
                mask[mask > 0] = i+1
                blend += mask
                blend[blend > (i+1)] = i+1
            blend = Image.fromarray(blend)
            blend.putpalette(palette)
        else:
            blend = Image.fromarray(blend)
        save_dir = os.path.join('/versa/dyy/dataset/TIANCHI/PreRoundData/refine_result', video_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        blend.save(os.path.join(save_dir, name.replace('.jpg', '.png')))
