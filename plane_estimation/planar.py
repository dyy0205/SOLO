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
        x_bts = F.interpolate(x, size=bts_input)   # bts input: (640, 480)
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
            ins_map = np.zeros((192, 256), dtype=np.uint8) * 255
            if masks is not None:
                resized_mask = []
                for i, mask in enumerate(masks):
                    mask = cv2.resize(mask, (256, 192))
                    resized_mask.append(mask)
                    valid[mask > 0] = 1
                    ins_map[mask > 0] = i

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