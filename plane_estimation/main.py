import os
import tqdm
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as T
import datetime

from plane_estimation.log import Logger
from plane_estimation.misc import AverageMeter, get_coordinate_map
from plane_estimation.dataset import PlaneDataset
from plane_estimation.planar import Planar
from plane_estimation.loss import param_loss, depth_loss, instance_aware_loss, total_loss


def load_dataset(args, mode):
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = args.batch_size if mode == 'train2' else 1
    is_shuffle = mode == 'train2'
    loaders = data.DataLoader(
        PlaneDataset(subset=mode, transform=transforms, root_dir=args.dataset_dir),
        batch_size=batch_size, shuffle=is_shuffle, num_workers=args.num_workers
    )

    return loaders


def train(args, model, device):
    print('Start Training...')

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # if torch.cuda.is_available():
    #     print(f'using CUDA devices {args.gpu}')
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    #     # model = model.module
    model.to(device)
    model.eval()
    model.get_plane.train()

    dataloader = load_dataset(args, mode='train2')

    optimizer = torch.optim.Adam(model.get_plane.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.get_plane.parameters(), lr=args.init_lr, momentum=0.9)

    k_inv_dot_xy1 = get_coordinate_map(device)

    for epoch in range(args.num_epochs):
        losses = AverageMeter()
        losses_param = AverageMeter()
        losses_depth = AverageMeter()
        losses_instance = AverageMeter()

        for iter, sample in enumerate(dataloader):
            image = sample['image'].to(device)
            raw_image = sample['raw_image'].to(device)
            instance = sample['instance'].to(device)
            semantic = sample['semantic'].to(device)
            gt_seg = sample['gt_seg'].to(device)
            gt_depth = sample['depth'].to(device)
            gt_plane_param = sample['plane_parameters'].to(device)
            valid_region = sample['valid_region'].to(device)

            # forward pass
            bts_depth, param, mask_lst, instance_map, solo_valid, solo_results = \
                model(image, raw_image, args.bts_input, args.plane_cls, args.solo_conf)
            # bts_depth = F.interpolate(bts_depth, size=(192, 256))

            loss, loss_param, loss_depth, loss_instance = 0., 0., 0., 0.
            batch_size = image.size(0)
            for i in range(batch_size):
                if mask_lst[i] is None:
                    continue
                # calculate loss

                valid = (valid_region[i:i+1] + solo_valid[i].unsqueeze(0).unsqueeze(0)) == 2

                # _loss_param = param_loss(
                #     param[i:i+1], gt_plane_param[i:i+1], valid)
                #
                # _loss_depth, rmse, inferred_depth = depth_loss(
                #     param[i:i+1], k_inv_dot_xy1, gt_depth[i:i+1])
                #
                # _loss_instance, instance_depth, abs_distance, instance_param = instance_aware_loss(
                #     mask_lst[i], instance_map[i], param[i], k_inv_dot_xy1, valid, gt_depth[i:i+1])

                _loss_param, _loss_instance, _loss_depth, inferred_depth, instance_depth, instance_param = \
                    total_loss(mask_lst[i], param[i], k_inv_dot_xy1, valid, gt_depth[i:i+1], gt_plane_param[i])

                if i == 1:
                    Image.fromarray(raw_image[i].detach().cpu().numpy()).save(
                        f'val_imgs/image_{i}.jpg')
                    Image.fromarray(gt_depth[i][0].detach().cpu().numpy()*1000).convert('I').save(
                        f'val_imgs/depth_{i}.png')
                    Image.fromarray(inferred_depth[0][0].detach().cpu().numpy()*1000).convert('I').save(
                        f'val_imgs/depth_infer_{i}.png')
                    Image.fromarray(instance_depth[0][0].detach().cpu().numpy() * 1000).convert('I').save(
                        f'val_imgs/depth_ins_{i}.png')

                _loss = _loss_param  #  + _loss_depth + _loss_instance

                loss += _loss
                # loss_param += _loss_param
                # loss_depth += _loss_depth
                # loss_instance += _loss_instance

            loss /= batch_size
            # loss_param /= batch_size
            # loss_depth /= batch_size
            # loss_instance /= batch_size

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss
            losses.update(loss.item())
            # losses_param.update(loss_param.item())
            # losses_depth.update(loss_depth.item())
            # losses_instance.update(loss_instance.item())

            if iter % args.print_interval == 0:
                log.logger.info(f"[{epoch:2d}][{iter:3d}/{len(dataloader):3d}] "
                                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                                # f"Param: {losses_param.val:.4f} ({losses_param.avg:.4f}) "
                                # f"Depth: {losses_depth.val:.4f} ({losses_depth.avg:.4f}) "
                                # f"Instance: {losses_instance.val:.4f} ({losses_instance.avg:.4f}) "
                                )

        log.logger.info(f"* epoch: {epoch:2d} "
                        f"Loss: {losses.avg:.6f} "
                        # f"Param: {losses_param.avg:.6f} "
                        # f"Depth: {losses_depth.avg:.6f} "
                        # f"Instance: {losses_instance.avg:.6f}\t"
                        )

        # save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.work_dir, f"epoch_{epoch}.pt"))

        # validate
        if args.train_with_val and epoch % args.val_interval == 0:
                val(args, model, device)


def val(args, model, device):

    model.to(device)
    model.eval()

    dataloader = load_dataset(args, mode='val')
    k_inv_dot_xy1 = get_coordinate_map(device)

    losses = AverageMeter()
    losses_param = AverageMeter()
    losses_depth = AverageMeter()
    losses_instance = AverageMeter()

    for iter, sample in enumerate(dataloader):
        image = sample['image'].to(device)
        raw_image = sample['raw_image'].to(device)
        instance = sample['instance'].to(device)
        semantic = sample['semantic'].to(device)
        gt_seg = sample['gt_seg'].to(device)
        gt_depth = sample['depth'].to(device)
        gt_plane_param = sample['plane_parameters'].to(device)
        valid_region = sample['valid_region'].to(device)

        # forward pass
        bts_depth, param, mask_lst, instance_map, _, solo_results = \
            model(image, raw_image, args.bts_input, args.plane_cls, args.solo_conf)

        loss, loss_param, loss_depth, loss_instance = 0., 0., 0., 0.
        batch_size = image.size(0)
        for i in range(batch_size):
            # calculate loss
            _loss_param = param_loss(param[i:i+1], gt_plane_param[i:i+1], valid_region[i:i+1])
            _loss_depth, rmse, infered_depth = depth_loss(param[i:i+1], k_inv_dot_xy1, gt_depth[i:i+1])

            if instance_map[i] is None:
                continue
            _loss_instance, inferred_depth, abs_distance, instance_param = instance_aware_loss(
                mask_lst[i], instance_map[i], param[i], k_inv_dot_xy1, valid_region[i:i+1], gt_depth[i:i+1])

            _loss = _loss_param + _loss_depth + _loss_instance

            loss += _loss
            loss_param += _loss_param
            loss_depth += _loss_depth
            loss_instance += _loss_instance

        loss /= batch_size
        loss_param /= batch_size
        loss_depth /= batch_size
        loss_instance /= batch_size

        # update loss
        losses.update(loss.item())
        losses_param.update(loss_param.item())
        losses_depth.update(loss_depth.item())
        losses_instance.update(loss_instance.item())

    log.logger.info(f"*******Val: "
                    f"Loss: {losses.avg:.6f} "
                    f"Param: {losses_param.avg:.6f} "
                    f"Depth: {losses_depth.avg:.6f} "
                    f"Instance: {losses_instance.avg:.6f}\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # BTS settings
    parser.add_argument('--bts_ckpt', type=str, default='./depth_model')
    parser.add_argument('--bts_input', type=tuple, default=(192, 256), help='bts input image shape (h, w)')
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
    parser.add_argument('--dataset_dir', type=str, default='/versa/dyy/dataset/scannet')
    parser.add_argument('--gpu', type=str, default='0,1,2')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--train_with_val', action='store_true')
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    args = parser.parse_args()

    DATETIME = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m%d-%H%M%S')
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    log = Logger(os.path.join(args.work_dir, DATETIME + '.log'))
    log.logger.info(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Planar(params=args, device=device)

    if args.mode == 'train':
        train(args, model, device)
    elif args.mode == 'val':
        val(args, model, device)
    else:
        raise NotImplementedError
