import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plane_estimation.BTS import BtsModel
from mmdet.apis import init_detector, inference_detector


def init_bts(params, device):
    bts = BtsModel(params=params).to(device)
    ckpt = torch.load(params.bts_ckpt)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    bts.load_state_dict(ckpt)
    bts.eval()
    return bts


class Planar(nn.Module):
    def __init__(self, params, device='cuda:0'):
        super(Planar, self).__init__()
        self.params = params
        self.device = device

        # instance segmentation model
        self.solo = init_detector(self.params.solo_cfg, self.params.solo_ckpt, device=self.device)

        # depth estimation model
        self.bts = init_bts(self.params, device=self.device)

        # plane param branch
        self.get_plane = nn.Conv2d(self.params.bts_size // 16, 3, 3, padding=1, bias=False)

    def forward(self, x, x_raw, bts_input, plane_cls, solo_conf):
        # BTS
        x_bts = F.interpolate(x, size=bts_input)   # bts input: (480, 640)
        _, _, _, _, feat, depth = self.bts(x_bts, focal=512, device=self.device)
        # Planes
        feat = F.interpolate(feat, size=(192, 256))
        plane_params = self.get_plane(feat)

        # SOLO
        results, mask_lst, instance_map, valid_region = [], [], [], []
        for img in x_raw:
            img = img.cpu().numpy()
            result, _ = inference_detector(self.solo, img)
            cur_result = result[0]
            if cur_result is not None:
                masks = cur_result[0].cpu().numpy().astype(np.uint8)
                classes = cur_result[1].cpu().numpy()
                scores = cur_result[2].cpu().numpy()

                inds = []
                for cls in plane_cls.split(','):
                    idx = (classes == int(cls)).nonzero()
                    inds.extend(list(idx))
                inds = np.concatenate(inds, axis=0)

                vis_inds = []
                for i in inds:
                    if scores[i] > solo_conf:
                        vis_inds.append(i)
                vis_inds = np.array(vis_inds)

                if len(vis_inds) != 0:
                    masks = masks[vis_inds]
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
                else:
                    masks, classes = None, None
            else:
                masks, classes = None, None

            valid = np.zeros((192, 256), dtype=np.uint8)
            ins_map = np.zeros((192, 256), dtype=np.uint8)
            if masks is not None:
                resized_mask = []
                for i, mask in enumerate(masks):
                    mask = cv2.resize(mask, (256, 192))
                    resized_mask.append(mask)
                    valid[mask > 0] = 1
                    ins_map[mask > 0] = i + 1

                masks = torch.tensor(resized_mask).float().to(self.device)
                ins_map = torch.from_numpy(ins_map).long().to(self.device)
                valid = torch.from_numpy(valid).to(self.device)
            else:
                masks, ins_map, valid = None, None, None

            results.append(result)
            mask_lst.append(masks)
            instance_map.append(ins_map)
            valid_region.append(valid)

        return depth, plane_params, mask_lst, instance_map, valid_region, results


class NYUPlanar(nn.Module):
    def __init__(self, from_bts_feat=False):
        super(NYUPlanar, self).__init__()
        self.from_bts_feat = from_bts_feat

        self.conv0 = nn.Conv2d(3, 32, 3, padding=1)
        self.daspp_3 = atrous_conv(32, 32, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(32 * 2, 32, 6)
        self.daspp_12 = atrous_conv(32 * 3, 32, 12)
        self.daspp_18 = atrous_conv(32 * 4, 32, 18)
        self.daspp_24 = atrous_conv(32 * 5, 32, 24)
        self.daspp_conv = nn.Sequential(nn.Conv2d(32 * 6, 32, 1, bias=False),
                                        nn.ELU())
        self.get_plane = nn.Conv2d(32, 3, 1, bias=False)

    def forward(self, x):
        if not self.from_bts_feat:
            x = self.conv0(x)

        daspp_3 = self.daspp_3(x)
        concat4_2 = torch.cat([x, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([x, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
        plane_params = self.get_plane(daspp_feat)

        return plane_params


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))
    def forward(self, x):
        return self.atrous_conv.forward(x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bts_size', default=512)
    parser.add_argument('--max_depth', default=10)
    parser.add_argument('--encoder', default='densenet161_bts')
    parser.add_argument('--dataset', default='nyu')
    parser.add_argument('--bts_ckpt', default='./depth_model')
    parser.add_argument('--solo_cfg', default='../ade_cfg/solov2_r101_dcn_22.py')
    parser.add_argument('--solo_ckpt', default='../indoor_dcn.pth')

    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(1, 3, 480, 640, device=device)

    model = Planar(params=args, device=device)
    print(model)