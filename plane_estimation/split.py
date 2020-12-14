import os, glob, shutil, random

root = '/versa/dyy/dataset/scannet/'
imgs = os.listdir(os.path.join(root, 'train'))
slice = random.sample(imgs, 5000)
save_dir = os.path.join(root, 'train2')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for img in slice:
    path = os.path.join(root, 'train', img)
    dst = os.path.join(save_dir, img)
    shutil.copy(path, dst)

with open(os.path.join(root, 'train2.txt'), 'w') as f:
    for line in slice:
        f.writelines(line)
        f.write('\n')
