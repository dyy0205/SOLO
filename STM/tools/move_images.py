import shutil
import os

img_set = r'/workspace/dataset/VOS/tianchiyusai/ImageSets/tianchi_val.txt'
ori_dir = r'/workspace/dataset/VOS/tianchiyusai/JPEGImages/'
save_dir = r'/workspace/dataset/VOS/tianchi_val/JPEGImages/'

with open(img_set, 'r') as f:
    set = f.read().splitlines()

print(set)

for name in set:
    if name in os.listdir(ori_dir):
        shutil.copytree(os.path.join(ori_dir, name), os.path.join(save_dir, name))
    else:
        print('{} not found!'.format(name))