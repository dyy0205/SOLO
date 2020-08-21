import os
import matplotlib.pyplot as plt

img_root = "/cfs/yuanlei/datasets/DAVIS2017/JPEGImages/480p"
txt_root = "/cfs/yuanlei/datasets/DAVIS2017/ImageSets/2016/train.txt"

longest = 0
shortest = 90
shortname = ''
longname = ''
length = [0,0,0,0,0,0,0]
with open(txt_root, 'r') as lines:
    for line in lines:
        video = line.rstrip('\n')
        train_path = os.path.join(img_root, video)
        frame_nums = len(os.listdir(train_path))
        # if frame_nums>longest:
        #     longest = frame_nums
        #     longname = video
        # elif frame_nums<shortest:
        #     shortest = len(os.listdir(train_path))
        #     shortname = video
        if frame_nums>=25 and frame_nums<35:
            length[0] += 1
        elif frame_nums>=35 and frame_nums<45:
            length[1] += 1
        elif frame_nums >= 45 and frame_nums < 55:
            length[2] += 1
        elif frame_nums>=55 and frame_nums<65:
            length[3] += 1
        elif frame_nums>=65 and frame_nums<75:
            length[4] += 1
        elif frame_nums>=75 and frame_nums<85:
            length[5] += 1
        elif frame_nums>=85 and frame_nums<95:
            length[6] += 1
x = list(range(35, 105, 10))

print(length)

plt.bar(x, length)
# plt.xticks(frame_nums, length)
plt.savefig('train_frame_nums.png')
plt.close()

# print('longest:', longest) # 36
# print('longname:', longname)
# print('shortest:', shortest) # 4
# print('shortname:', shortname)
