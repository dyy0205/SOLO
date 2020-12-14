import os, glob
import numpy as np
from PIL import Image
import torch
import cv2
import argparse
import torchvision.transforms as T

from plane_estimation.misc import get_nyu_coordinate_map
from plane_estimation.planar import NYUPlanar
from plane_estimation.BTS import BtsModel
from plane_estimation.loss import nyu_loss
from mmdet.apis import init_detector, inference_detector

SOLO_CLASSES = ('floor', 'wall', 'door', 'window', 'curtain', 'painting', 'wall_o',
                'ceiling', 'fan', 'bed', 'desk', 'cabinet', 'chair', 'sofa',
                'lamp', 'furniture', 'electronics', 'person', 'cat', 'dog', 'plant', 'other')


def convert_3dpoints(depth_map, h, w):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240
    points = []
    for v in range(h):
        for u in range(w):
            Z = depth_map[v][u]
            X = (u - offset_x) * Z / focal_length
            Y = (v - offset_y) * Z / focal_length
            points.append([X, Y, Z])
    points = np.array(points).reshape((h, w, 3)) / 1000.
    return points.transpose(2, 0, 1)


def solo_infer(model, img, conf, plane_cls, h, w):
    result, _ = inference_detector(model, img)
    cur_result = result[0]
    if cur_result is not None:
        masks = cur_result[0].cpu().numpy().astype(np.uint8)
        classes = cur_result[1].cpu().numpy()
        scores = cur_result[2].cpu().numpy()

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

        plane_ins, plane_cate = [], []
        for mask, cls in zip(masks, classes):
            if cls in plane_cls:
                mask = cv2.resize(mask, (w, h))
                plane_ins.append(mask)
                plane_cate.append(cls)

        instance_map = np.zeros((h, w), dtype=np.uint8)
        valid_region = np.zeros((h, w), dtype=np.uint8)
        if masks is not None:
            for i, mask in enumerate(plane_ins):
                valid_region[mask > 0] = 1
                instance_map[mask > 0] = i + 1

        return masks, classes, plane_ins, plane_cate, instance_map, valid_region


def predict(args, bts_model, solo_model, plane_model):
    h, w = 480, 640
    k_inv_dot_xy1 = get_nyu_coordinate_map('cuda')

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if os.path.isdir(args.image_path):
        images = glob.glob(os.path.join(args.image_path, '*.jpg'))
    else:
        images = [args.image_path]

    with torch.no_grad():
        for i, image_path in enumerate(images):
            print(i, image_path)
            image_pil = Image.open(image_path)
            raw_w, raw_h = image_pil.size
            image = transforms(image_pil.resize((w, h)))
            image = image.unsqueeze(0).cuda()

            # SOLO
            # print('Start SOLO predicting...')
            solo_masks, solo_cates, plane_ins, plane_cate, instance_map, valid_region = \
                solo_infer(solo_model, np.array(image_pil), args.solo_conf, args.plane_cls, h, w)
            instance_map = torch.from_numpy(instance_map).long().cuda()
            valid_region = torch.from_numpy(valid_region).cuda()

            # BTS
            # print('Start BTS predicting...')
            _, _, _, _, last_feat, bts_depth = bts_model(image)
            bts_depth = bts_depth.squeeze().cpu().numpy()
            if args.from_bts_feat:
                plane_input = last_feat
            else:
                plane_input = convert_3dpoints(bts_depth * 1000, h, w)
                plane_input = torch.from_numpy(plane_input).unsqueeze(0).float().cuda()

            # Plane
            # print('Generating planes...')
            plane_params = plane_model(plane_input)

            plane_depth, instance_param = nyu_loss(
                instance_map, plane_params.squeeze(0), k_inv_dot_xy1, valid_region,
                torch.ones((3, h, w)), torch.ones((h, w)), return_loss=False)

            plane_depth = plane_depth.cpu().numpy()
            instance_param = instance_param.cpu().numpy()

            Image.fromarray(bts_depth * 1000).convert('I').resize((raw_w, raw_h)).save(
                image_path.replace('.jpg', '_bts.png'))
            Image.fromarray(plane_depth * 1000).convert('I').resize((raw_w, raw_h)).save(
                image_path.replace('.jpg', '_depth.png'))

            image_np = np.array(image_pil.resize((w, h)))
            for mask in plane_ins:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask_bool = mask.astype(np.bool)
                image_np[mask_bool] = image_np[mask_bool] * 0.5 + color_mask * 0.5
            Image.fromarray(image_np).save(
                image_path.replace('.jpg', '_blend.png'))
            print('Plane params: \n', instance_param)
            print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # BTS settings
    parser.add_argument('--bts_ckpt', type=str, default='./depth_model')
    parser.add_argument('--bts_size', type=int, default=512, help='initial num_filters in bts')
    parser.add_argument('--max_depth', type=float, default=10, help='maximum depth in estimation')
    parser.add_argument('--dataset', type=str, default='nyu', help='dataset to train on, kitti or nyu')
    parser.add_argument('--encoder', type=str, default='densenet161_bts',
                        help='type of encoder, desenet121_bts, densenet161_bts, '
                             'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts')
    # SOLO settings
    parser.add_argument('--solo_cfg', type=str, default='../ade_cfg/solov2_r101_dcn_22.py')
    parser.add_argument('--solo_ckpt', type=str, default='../indoor_dcn.pth')
    parser.add_argument('--plane_cls', type=int, nargs='+', help='plane categories: floor,wall,ceiling,etc')
    parser.add_argument('--solo_conf', type=float, default=0.2)
    # Predict
    parser.add_argument('--plane_ckpt', type=str, default='./work_dir/debug/epoch_99.pt')
    parser.add_argument('--image_path', type=str, default='/versa/dyy/dataset/scannet/val')
    parser.add_argument('--from_bts_feat', action='store_true', help='use bts last layer features or final depth')

    args = parser.parse_args()

    # BTS
    bts_model = BtsModel(params=args).cuda()
    bts_ckpt = torch.load(args.bts_ckpt)
    bts_ckpt = {k.replace('module.', ''): v for k, v in bts_ckpt['model'].items()}
    bts_model.load_state_dict(bts_ckpt)
    bts_model.cuda()
    bts_model.eval()

    # SOLO
    solo_model = init_detector(args.solo_cfg, args.solo_ckpt, device='cuda:0')

    # Plane
    plane_model = NYUPlanar(args.from_bts_feat)
    plane_ckpt = torch.load(args.plane_ckpt)
    plane_ckpt = {k.replace('module.', ''): v for k, v in plane_ckpt.items()}
    plane_model.load_state_dict(plane_ckpt)
    plane_model.cuda()
    plane_model.eval()

    predict(args, bts_model, solo_model, plane_model)
