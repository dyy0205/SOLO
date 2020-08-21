import os, glob
import cv2
from PIL import Image
import numpy as np

save_dir = '/data/sdv2/project_hkf/STM_test/inference/test/2007171408/merge/'
out_dir = '/data/sdv2/project_hkf/STM_test/inference/test/2007171408/tianchi/'


root = '/data/sdv2/workspace/tianchi/dataset/tianchiyusai'
img_root = os.path.join(root, 'JPEGImages')
ann_root = os.path.join(root, 'Annotations')
with open(os.path.join(root, 'ImageSets/test.txt'), 'r') as f:
    test = f.readlines()
test = [img.strip() for img in test]
print('test videos: ', len(test))

ins_lst = os.listdir(out_dir)
names = []
for name in ins_lst:
    name = name.split('_')[0]
    if name not in names:
        names.append(name)
print(len(names))

for i, name in enumerate(test):
    num_frames = len(glob.glob(os.path.join(img_root, name, '*.jpg')))
    ann_path = os.path.join(ann_root, name, '00000.png')
    mask_f = Image.open(ann_path)
    w, h = mask_f.size
    palette = mask_f.getpalette()
    ins = [ins for ins in ins_lst if ins.startswith(name)]
    print(i, name, len(ins))

    video_dir = os.path.join(save_dir, name)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    if len(ins) == 1:
        for t in range(num_frames):
            path = os.path.join(out_dir, name+'_1', '{:05d}.png'.format(t))
            mask = Image.open(path).convert('P').resize((w, h))
            mask.putpalette(palette)
            mask.save(os.path.join(video_dir, '{:05d}.png'.format(t)))
    else:
        for t in range(num_frames):
            mask = np.zeros((h, w), dtype=np.uint8)
            for j in range(1, len(ins)+1):
                path = os.path.join(out_dir, name + '_{}'.format(j), '{:05d}.png'.format(t))
                temp = np.array(Image.open(path).convert('P').resize((w, h)), dtype=np.uint8)
                temp[temp == 1] = j
                mask += temp
                mask[mask > j] = j
            # print(len(ins), np.unique(mask))
            mask = Image.fromarray(mask)
            mask.putpalette(palette)
            mask.save(os.path.join(video_dir, '{:05d}.png'.format(t)))