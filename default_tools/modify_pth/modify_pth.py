import torch
from collections import OrderedDict, defaultdict

if __name__ == '__main__':

    net = torch.load('/versa/dyy/pretrained_models/vovnet39_torchvision.pth')
    state_dict = OrderedDict()
    for i, (k, v) in enumerate(net.items()):
        print(i, k, v.shape)
        k = k.replace('module.', '')
        state_dict[k] = v
    #     if '_fc' in k or '_se' in k:
    #         continue
    #     elif '_bn' in k:
    #         state_dict[k] = v
    #     else:
    #         _k = k.replace(k.split('.')[-1], 'conv.' + k.split('.')[-1])
    #         state_dict[_k] = v
    # # new_state_dict = defaultdict(list)
    # # new_state_dict['state_dict'] = state_dict
    #
    for i, (k, v) in enumerate(state_dict.items()):
        print(i, k)
    torch.save(state_dict, '/versa/dyy/pretrained_models/vovnet39.pth')

    # model = torch.load('../work_dirs/decoupled_solo_light_b3_fpn_lite/epoch_1.pth')
    # state_dict = OrderedDict()
    # for k, v in model['state_dict'].items():
    #     print(k)
    #     if k.startswith('backbone'):
    #         _k = k.replace('backbone.', '')
    #         state_dict[_k] = v
    #     else:
    #         continue
    # for i, (k, v) in enumerate(state_dict.items()):
    #     print(i, k)
    # torch.save(state_dict, 'pretrained_efficientnet-b3_lite.pth')