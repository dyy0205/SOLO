import torch
import torch.nn.functional as F
import numpy as np


def param_loss(pred_param, gt_param, valid_region):
    b, c, h, w = pred_param.size()
    if valid_region is None:
        valid_pred = torch.transpose(pred_param.view(c, -1), 0, 1)
        valid_gt = torch.transpose(gt_param.view(c, -1), 0, 1)
    else:
        valid_pred = torch.transpose(torch.masked_select(pred_param, valid_region).view(c, -1), 0, 1)
        valid_gt = torch.transpose(torch.masked_select(gt_param, valid_region).view(c, -1), 0, 1)

    return torch.mean(torch.sum(torch.abs(valid_pred - valid_gt), dim=1))


def depth_loss(param, gt_param, k_inv_dot_xy1, gt_depth):
    '''
    infer per pixel depth using perpixel plane parameter and
    return depth loss, mean abs distance to gt depth, perpixel depth map
    :param param: plane parameters defined as n/d , tensor with size (1, 3, h, w)
    :param k_inv_dot_xy1: tensor with size (3, h*w)
    :param gt_depth: tensor with size(1, 1, h, w)
    :return: error and abs distance
    '''

    b, c, h, w = param.size()
    assert (b == 1 and c == 3)

    gt_depth = gt_depth.view(1, h*w)
    param = param.view(c, h*w)

    # infer depth for every pixel
    infered_depth = 1. / torch.sum(param * k_inv_dot_xy1, dim=0, keepdim=True)  # (1, h*w)
    infered_depth = infered_depth.view(1, h * w)

    gt_param = gt_param.view(c, h*w)
    gt = 1. / torch.sum(gt_param * k_inv_dot_xy1, dim=0, keepdim=True)  # (1, h*w)
    gt = gt.view(h, w)

    # ignore insufficient depth
    infered_depth = torch.clamp(infered_depth, 1e-4, 10.0)

    # select valid depth
    mask = gt_depth != 0.0
    valid_gt_depth = torch.masked_select(gt_depth, mask)
    valid_depth = torch.masked_select(infered_depth, mask)
    valid_param = torch.masked_select(param, mask).view(3, -1)
    valid_ray = torch.masked_select(k_inv_dot_xy1, mask).view(3, -1)

    diff = torch.abs(valid_depth - valid_gt_depth)
    abs_distance = torch.mean(diff)

    Q = valid_ray * valid_gt_depth   # (3, n): 3D points inferred from gt depth
    q_diff = torch.abs(torch.sum(valid_param * Q, dim=0, keepdim=True) - 1.)
    loss = torch.mean(q_diff)
    return loss, abs_distance, infered_depth.view(1, 1, h, w), gt


def instance_aware_loss(solo_masks, solo_classmap, pred_param, k_inv_dot_xy1, valid_region, gt_depth, return_loss=True):
    """
    calculate loss of parameters
    first we combine sample segmentation with sample params to get K plane parameters
    then we used this parameter to infer plane based Q loss as done in PlaneRecover
    the loss enforce parameter is consistent with ground truth depth

    :param solo_masks: tensor with size (K, h, w)
    :param solo_classmap: tensor with size (h, w)
    :param pred_param: tensor with size (3, h, w)
    :param valid_region: tensor with size (1, 1, h, w), indicate planar region
    :param gt_depth: tensor with size (1, 1, h, w)
    :param return_loss: bool
    :return: loss
             inferred depth with size (1, 1, h, w) corresponded to instance parameters
    """

    _, _, h, w = gt_depth.size()

    # # combine sample segmentation and sample params to get instance parameters
    # instance_param = []
    # for mask in solo_masks:
    #     param = torch.cat([param[mask > 0] for param in pred_param], dim=0)
    #     param = param.view(3, -1).mean(dim=1)
    #     instance_param.append(param.detach().cpu().numpy())
    # instance_param = torch.tensor(instance_param, device=pred_param.device)  # (K, 3)
    #
    # # infer depth for every pixels and select the one with highest probability
    # depth_maps = 1. / torch.matmul(instance_param, k_inv_dot_xy1)  # (K, h*w)
    # solo_classmap = solo_classmap.view(-1)
    # inferred_depth = depth_maps.t()[range(h * w), solo_classmap].view(1, 1, h, w)

    # infer depth for every pixels
    param = pred_param.clone()
    instance_param = []
    for mask in solo_masks:
        mask = (mask > 0)
        ins_param = pred_param[:, mask].mean(dim=1)
        param[:, mask] = ins_param.repeat(mask.sum(), 1).transpose(0, 1)
        instance_param.append(ins_param)
    instance_param = torch.cat(instance_param, dim=0).view(-1, 3)   # (K, 3)
    param = param.view(-1, h*w)

    inferred_depth = 1. / torch.sum(param * k_inv_dot_xy1, dim=0, keepdim=True)  # (1, h*w)
    inferred_depth = inferred_depth.view(1, 1, h, w)

    if not return_loss:
        return _, inferred_depth, _, instance_param

    # select valid region
    valid_region = ((valid_region + (gt_depth != 0.0)) == 2).view(-1)
    ray = k_inv_dot_xy1[:, valid_region]                             # (3, N)
    valid_depth = gt_depth.view(1, -1)[:, valid_region]              # (1, N)
    valid_inferred_depth = inferred_depth.view(1, -1)[:, valid_region]

    # Q_loss for every instance
    Q = valid_depth * ray                                          # (3, N)
    Q_loss = torch.abs(torch.matmul(instance_param, Q) - 1.)       # (K, N)

    # weight Q_loss with probability
    solo_masks = solo_masks.view(-1, h*w)[:, valid_region]         # (K, N)
    weighted_Q_loss = Q_loss * solo_masks                          # (K, N)

    loss = torch.sum(torch.mean(weighted_Q_loss, dim=1))

    # abs distance for valid infered depth
    abs_distance = torch.mean(torch.abs(valid_inferred_depth - valid_depth))

    return loss, inferred_depth, abs_distance, instance_param
