import os
import tqdm
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as T
import datetime

from plane_estimation.log import Logger
from plane_estimation.misc import AverageMeter, get_nyu_coordinate_map
from plane_estimation.dataset import NYUPlaneDataset
from plane_estimation.planar import NYUPlanar
from plane_estimation.loss import nyu_loss


def load_dataset(args, mode):
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = args.batch_size if mode == 'train' else 1
    is_shuffle = mode == 'train'
    loaders = data.DataLoader(
        NYUPlaneDataset(subset=mode, transform=transforms, root_dir=args.dataset_dir),
        batch_size=batch_size, shuffle=is_shuffle, num_workers=args.num_workers, pin_memory=True
    )

    return loaders


def train(args, model):
    print('Start Training...')
    model.train()

    dataloader = load_dataset(args, mode='train')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)

    k_inv_dot_xy1 = get_nyu_coordinate_map('cuda')

    for epoch in range(args.num_epochs):
        losses = AverageMeter()
        losses_param = AverageMeter()
        losses_depth = AverageMeter()
        losses_instance = AverageMeter()

        for iter, sample in enumerate(dataloader):
            # points_3d = sample['points_3d'].cuda()
            bts_feat = sample['last_layer_feat'].cuda(non_blocking=True)
            raw_image = sample['raw_image'].cuda(non_blocking=True)
            gt_seg = sample['gt_seg'].cuda(non_blocking=True)
            gt_depth = sample['depth'].cuda(non_blocking=True)
            gt_plane_param = sample['plane_parameters'].cuda(non_blocking=True)
            valid_region = sample['valid_region'].cuda(non_blocking=True)

            # forward pass
            input = bts_feat    # if args.from_bts_feat else points_3d
            param = model(input)

            loss, loss_param, loss_depth = 0., 0., 0.
            batch_size = input.size(0)
            for i in range(batch_size):
                _loss_param, _loss_depth, instance_depth, instance_param = \
                    nyu_loss(gt_seg[i], param[i], k_inv_dot_xy1, valid_region[i], gt_plane_param[i], gt_depth[i])

                if i == 0:
                    Image.fromarray(raw_image[i].cpu().numpy()).save(
                        f'val_imgs/image_{i}.jpg')
                    Image.fromarray(gt_depth[i].cpu().numpy()*1000).convert('I').save(
                        f'val_imgs/depth_{i}.png')
                    Image.fromarray(instance_depth.detach().cpu().numpy()*1000).convert('I').save(
                        f'val_imgs/depth_ins_{i}.png')

                _loss = _loss_param * 10.  # + _loss_depth

                loss += _loss
                # loss_param += _loss_param
                # loss_depth += _loss_depth

            loss /= batch_size
            # loss_param /= batch_size
            # loss_depth /= batch_size

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss
            losses.update(loss.item())
            # losses_param.update(loss_param.item())
            # losses_depth.update(loss_depth.item())

            if iter % args.print_interval == 0:
                log.logger.info(f"[{epoch:2d}][{iter:3d}/{len(dataloader):3d}] "
                                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                                # f"Param: {losses_param.val:.4f} ({losses_param.avg:.4f}) "
                                # f"Depth: {losses_depth.val:.4f} ({losses_depth.avg:.4f}) "
                                )

        log.logger.info(f"* epoch: {epoch:2d} "
                        f"Loss: {losses.avg:.6f} "
                        # f"Param: {losses_param.avg:.6f} "
                        # f"Depth: {losses_depth.avg:.6f} "
                        )

        # save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.work_dir, f"epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # BTS settings
    parser.add_argument('--bts_ckpt', type=str, default='./depth_model')
    parser.add_argument('--bts_input', type=tuple, default=(480, 640), help='bts input image shape')
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
    # training
    parser.add_argument('--mode', type=str, default='train', help='train or val')
    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--dataset_dir', type=str, default='/versa/dyy/dataset/nyu_depth_v2')
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--train_with_val', action='store_true')
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--resume_from', type=str, default='None')
    parser.add_argument('--from_bts_feat', action='store_true', help='use bts last layer features or final depth')

    args = parser.parse_args()

    DATETIME = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m%d-%H%M%S')
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    log = Logger(os.path.join(args.work_dir, DATETIME + '.log'))
    log.logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = NYUPlanar(args.from_bts_feat)
    model = nn.DataParallel(model)
    model.cuda()

    if not (args.resume_from == 'None'):
        print(f'\nResume from {args.resume_from}...')
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint)

    if args.mode == 'train':
        train(args, model)
    else:
        raise NotImplementedError
