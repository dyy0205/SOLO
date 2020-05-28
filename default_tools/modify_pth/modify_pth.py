import torch
from collections import OrderedDict, defaultdict

if __name__ == '__main__':

    net = torch.load('/home/dingyangyang/pretrained_models/solo2-best.pth')
    state_dict = OrderedDict()
    for k, v in net['state_dict'].items():
        # print(k, v.shape)
        if k.startswith('backbone') or k.startswith('neck'):
            state_dict[k] = v

    for k, v in state_dict.items():
        print(k)
    torch.save(state_dict, '/home/dingyangyang/pretrained_models/solo2-lite3_bifpn.pth')
