import os
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from plane_estimation.BTS import BtsModel


parser = argparse.ArgumentParser()
parser.add_argument('--bts_size', default=512)
parser.add_argument('--max_depth', default=10)
parser.add_argument('--encoder', default='densenet161_bts')
parser.add_argument('--dataset', default='nyu')
parser.add_argument('--checkpoint_path', default='./depth_model')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = BtsModel(params=args).cuda()
model = torch.nn.DataParallel(model)

checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.cuda()
model.eval()


if __name__ == '__main__':
    root = '/versa/dyy/dataset/nyu_depth_v2/'
    img_dir = os.path.join(root, 'sync')
    save_dir = os.path.join(root, 'bts_depth')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(root, 'classroom.txt'), 'r') as f:
        data = f.readlines()

    with torch.no_grad():
        for i, line in enumerate(data):
            name, img_path, depth_path = line.strip().split(' ')
            img_path = os.path.join(img_dir, img_path)
            print(i, name)
            input_image = Image.open(img_path)
            raw_w, raw_h = input_image.size
            input_images = transforms(input_image.resize((640, 480)))

            image = input_images.unsqueeze(0).cuda()
            _, _, _, _, feat, depth = model(image)
            feat = feat.squeeze().cpu().numpy()
            depth = depth.squeeze().cpu().numpy() * 1000
            np.savez(os.path.join(save_dir, name.replace('.png', '.npz')),
                     bts_depth=depth, last_feat=feat, img_path=img_path)