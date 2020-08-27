import torch

# pretrained_model = r'/workspace/solo/backup_models/model.pth'
# save_model = r'/workspace/solo/backup_models/modified_model_for_enhanced.pth'
#
# model = torch.load(pretrained_model)
# kw = torch.zeros((128, 1025, 3, 3))
# vw = torch.zeros((512, 1025, 3, 3))
# k = model['module.KV_Q_r4.Key.weight']
# v = model['module.KV_Q_r4.Value.weight']
# kw[:, :1024, :, :] = k
# vw[:, :1024, :, :] = v
#
# model['module.KV_Q.Key.weight'] = kw
# model['module.KV_Q.Value.weight'] = vw
# model['module.KV_Q.Key.bias'] = model['module.KV_Q_r4.Key.bias']
# model['module.KV_Q.Value.bias'] = model['module.KV_Q_r4.Value.bias']
# model.pop('module.KV_Q_r4.Key.bias')
# model.pop('module.KV_Q_r4.Value.bias')
# model.pop('module.KV_Q_r4.Key.weight')
# model.pop('module.KV_Q_r4.Value.weight')
#
# torch.save(model, save_model)

# blend enhanced and motion model
enhanced = r'/workspace/solo/backup_models/enhanced2_interval7.pth'
motion = r'/workspace/solo/backup_models/dyy_ckpt_124e.pth'
save = r'/workspace/solo/backup_models/enhanced_motion.pth'

enhanced_model = torch.load(enhanced)
motion_model = torch.load(motion)

enhanced_model.pop('optimizer')
motion_model.pop('optimizer')

for k, v in motion_model['state_dict'].items():
    if k.startswith('module.Memory'):
        print(k)
        enhanced_model['state_dict'][k] = v

torch.save(enhanced_model, save)

