import torch
from collections import OrderedDict, defaultdict

if __name__ == '__main__':

    net = torch.load('/home/dingyangyang/pretrained_models/SOLOv2_R101_DCN_3x.pth')
    state_dict = OrderedDict()
    for k, v in net['state_dict'].items():
        # print(k, v.shape)
        if k.startswith('mask_feat_head.convs_all_levels'):
            k = k.replace('mask_feat_head.convs_all_levels', 'bbox_head.feature_convs')
        elif k.startswith('mask_feat_head.conv_pred.0'):
            k = k.replace('mask_feat_head.conv_pred.0', 'bbox_head.solo_mask')
        else:
            k = k
        state_dict[k] = v

    for k, v in state_dict.items():
        print(k)
    torch.save(state_dict, '/home/dingyangyang/pretrained_models/SOLOv2_R101_DCN_3x_modify.pth')
