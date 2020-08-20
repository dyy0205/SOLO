import torch
from collections import OrderedDict, defaultdict

if __name__ == '__main__':

    # net0 = torch.load('/versa/dyy/SOLO/solo2-best.pth')
    # net = torch.load('/versa/dyy/SOLO/work_dirs/solo_attention_td/epoch_1.pth')
    # print(net0['state_dict']['neck.bifpn.0.p2_down_channel.0.conv.weight'] == net['state_dict']['neck.bifpn.0.p2_down_channel.0.conv.weight'])
    # state_dict = OrderedDict()
    # for k, v in net['state_dict'].items():
    #     print(k, v.shape)
    #     if k.startswith('backbone') or k.startswith('neck') \
    #             or 'bbox_head.feature_convs' in k:
    #         state_dict[k] = v
    #
    # for k, v in state_dict.items():
    #     print(k)
    # torch.save(state_dict, '/versa/dyy/SOLO/solo2-lite3_bifpn.pth')

    net = torch.load('/versa/dyy/pretrained_models/RegNetX-1.6GF_dds_8gpu.pth')
    for k, v in net['state_dict'].items():
        print(k, v.shape)