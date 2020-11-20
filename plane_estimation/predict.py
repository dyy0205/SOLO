import os, glob
import cv2
import numpy as np
from PIL import Image
import torch
import argparse
import torchvision.transforms as T
from mmdet.apis import show_result_ins

from plane_estimation.misc import get_coordinate_map
from plane_estimation.planar import Planar
from plane_estimation.loss import depth_loss, instance_aware_loss

SOLO_CLASSES = ('floor', 'wall', 'door', 'window', 'curtain', 'painting', 'wall_o',
                'ceiling', 'fan', 'bed', 'desk', 'cabinet', 'chair', 'sofa',
                'lamp', 'furniture', 'electronics', 'person', 'cat', 'dog', 'plant', 'other')


def predict(args, model, device):
    h, w = 192, 256
    k_inv_dot_xy1 = get_coordinate_map(device)

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if os.path.isdir(args.image_path):
        images = glob.glob(os.path.join(args.image_path, '*.jpg'))
    else:
        images = [args.image_path]

    with torch.no_grad():
        for image_path in images:
            image_pil = Image.open(image_path)
            raw_w, raw_h = image_pil.size
            image = transforms(image_pil)
            image = image.to(device).unsqueeze(0)
            raw_image = [torch.from_numpy(np.array(image_pil))]

            # forward pass
            bts_depth, param, mask_lst, instance_map, valid_region, solo_results = \
                model(image, raw_image, args.bts_input, args.plane_cls, args.solo_conf)

            # # infer per pixel depth using per pixel plane parameter,
            # # currently depth_loss need a dummy gt_depth as input
            # _, _, per_pixel_depth = depth_loss(param, k_inv_dot_xy1, torch.ones((1, 1, h, w)))

            # infer instance depth
            _, plane_depth, _, instance_param = instance_aware_loss(
                mask_lst[0], instance_map[0], param[0], k_inv_dot_xy1,
                torch.ones((1, 1, h, w)), torch.ones((1, 1, h, w)), return_loss=False)

            plane_depth = plane_depth.cpu().numpy()[0, 0].reshape(h, w)
            bts_depth = bts_depth.cpu().numpy()[0, 0].reshape(640, 480)
            bts_depth = cv2.resize(bts_depth, (w, h))
            valid_region = valid_region[0].cpu().numpy()

            # use per pixel depth for non planar region
            depth_clean = plane_depth * valid_region
            depth = plane_depth * valid_region + bts_depth * (1 - valid_region)
            Image.fromarray(depth * 1000).convert('I').resize((raw_w, raw_h)).save(
                image_path.replace('.jpg', '_depth.png'))
            Image.fromarray(depth_clean * 1000).convert('I').resize((raw_w, raw_h)).save(
                image_path.replace('.jpg', '_depth_clean.png'))

            # # visualize depth map
            # depth = 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
            # depth = cv2.cvtColor(cv2.resize(depth, (raw_w, raw_h)), cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(image_path.replace('.jpg', '.png'), depth)

            # visualize solo results
            show_result_ins(np.array(image_pil), solo_results[0], SOLO_CLASSES,
                            score_thr=0.2, out_file=image_path.replace('.jpg', '_blend.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # BTS settings
    parser.add_argument('--bts_ckpt', type=str, default='./depth_model')
    parser.add_argument('--bts_input', type=tuple, default=(640, 480), help='bts input image shape')
    parser.add_argument('--bts_size', type=int, default=512, help='initial num_filters in bts')
    parser.add_argument('--max_depth', type=float, default=10, help='maximum depth in estimation')
    parser.add_argument('--dataset', type=str, default='nyu', help='dataset to train on, kitti or nyu')
    parser.add_argument('--encoder', type=str, default='densenet161_bts',
                        help='type of encoder, desenet121_bts, densenet161_bts, '
                             'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts')
    # SOLO settings
    parser.add_argument('--solo_cfg', type=str, default='../ade_cfg/solov2_r101_dcn_22.py')
    parser.add_argument('--solo_ckpt', type=str, default='../indoor_dcn.pth')
    parser.add_argument('--plane_cls', type=str, default='0,1,7', help='plane categories: floor,wall,ceiling,etc')
    parser.add_argument('--solo_conf', type=float, default=0.2)
    # Predict
    parser.add_argument('--final_ckpt', type=str, default='./work_dir/epoch_95.pt')
    parser.add_argument('--image_path', type=str, default='/versa/dyy/dataset/scannet/val')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Planar(params=args, device=device)

    checkpoint = torch.load(args.final_ckpt)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    predict(args, model, device)
