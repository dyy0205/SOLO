import torch
from collections import OrderedDict, defaultdict

if __name__ == '__main__':

    net = torch.load('/home/dingyangyang/SOLO/work_dirs/ade_indoor_resnest/epoch_24_tuned.pth')
    state_dict = OrderedDict()
    for k, v in net['state_dict'].items():
        # print(k, v.shape)
        if k.startswith('backbone'):
            state_dict[k] = v

    for k, v in state_dict.items():
        print(k)
    torch.save(state_dict, '/home/dingyangyang/SOLO/work_dirs/ade_indoor_resnest/epoch_24_backbone.pth')
