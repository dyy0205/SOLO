import os
import json
import skimage.io as io
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


def showAnns(json_file, img_dir):
    coco = COCO(json_file)
    with open(json_file,'r') as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    for img in images:
        img_id = img['id']
        I = io.imread(os.path.join(img_dir, img['file_name']))
        plt.axis('off')
        plt.imshow(I)
        anns = []
        for ann in annotations:
            if ann['image_id'] == img_id:
                anns.append(ann)
        coco.showAnns(anns)
        plt.show()


if __name__ == '__main__':
    json_file = '/Users/dyy/Desktop/human/out.json'
    img_dir = '/Users/dyy/Desktop/human/images/'
    showAnns(json_file, img_dir)

