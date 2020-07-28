import os
import random

dataset_root = r'/workspace/tianchi/dataset/tianchiyusai/ImageSets/'

davis_set_t = r'davis_train.txt'
davis_set_v = r'davis_val.txt'
youtube_t = 'youtube_train.txt'
youtube_v = 'youtube_val.txt'
versa_t = 'versa_train.txt'
versa_v = 'versa_val.txt'
tianchi_set_t = r'train.txt'

VAL_RATIO = 0.05
VAL_NUM = 50


def compose_dir(file):
    return os.path.join(dataset_root, file)


save_t = r'total_train.txt'
save_v = r'total_val.txt'

with open(compose_dir(tianchi_set_t), 'r') as f:
    tianchi_all = f.read().splitlines()

with open(compose_dir(davis_set_t), 'r') as f:
    davis_train = f.read().splitlines()

with open(compose_dir(davis_set_v), 'r') as f:
    davis_val = f.read().splitlines()

with open(compose_dir(youtube_t), 'r') as f:
    youtube_train = f.read().splitlines()

with open(compose_dir(youtube_v), 'r') as f:
    youtube_val = f.read().splitlines()

with open(compose_dir(versa_t), 'r') as f:
    versa_train = f.read().splitlines()

with open(compose_dir(versa_v), 'r') as f:
    versa_val = f.read().splitlines()


if VAL_NUM is None:
    VAL_NUM = int(VAL_RATIO * len(tianchi_all))
tianchi_val = random.sample(tianchi_all, VAL_NUM)
tianchi_train = []
for i in tianchi_all:
    if i not in tianchi_val:
        tianchi_train.append(i)

total_train = tianchi_train + youtube_train + youtube_val + versa_train + versa_val
total_val = tianchi_val
print('total train:{}, total val:{}'.format(len(total_train), len(total_val)))

with open(compose_dir(save_t), 'w') as f:
    for i in total_train:
        f.write(i)
        f.write('\r\n')

with open(compose_dir(save_v), 'w') as f:
    for i in total_val:
        f.write(i)
        f.write('\r\n')
