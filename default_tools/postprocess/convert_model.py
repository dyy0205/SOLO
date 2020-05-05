import mmcv
import onnx
import torch
from mmcv.runner import load_checkpoint
from onnx import optimizer

from mmdet.models import build_detector

config = '../../cfg/solov2_lite3_fpn_fp16.py'
pth = "../../solo2-407e4836.pth"
input_shape = (3, 800, 1280)

cfg = mmcv.Config.fromfile(config)
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
# print(model)
# model.load_state_dict(torch.load(pth, map_location='cpu')['state_dict'])
# model.eval()
load_checkpoint(model, pth, map_location='cpu')
model.cpu().eval()
if hasattr(model, 'forward_dummy'):
    model.forward = model.forward_dummy
else:
    raise NotImplementedError(
        'ONNX conversion is currently not currently supported with '
        '{}'.format(model.__class__.__name__))

input_data = torch.empty((1, *input_shape),
                         dtype=next(model.parameters()).dtype,
                         device=next(model.parameters()).device)
torch.onnx.export(model, input_data, "../../solo2.onnx", verbose=True)