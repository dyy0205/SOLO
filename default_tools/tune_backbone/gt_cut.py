#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import json
import pycocotools.mask as maskUtils


def get_defect_patch(json_file, img_dir, save_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    for i, img in enumerate(images):
        img_name = img['file_name']
        print(i, img_name)
        img_id = img['id']
        image = cv2.imread(os.path.join(img_dir, img_name))
        height, width = image.shape[:2]
        for j, anno in enumerate(annotations):
            if anno['image_id'] == img_id:
                category = anno['category_id']
                bbox = anno['bbox']
                x, y, w, h = [round(c) for c in bbox]
                if min(w, h) < 100:
                    continue
                try:
                    segm = anno['segmentation']
                    if type(segm) == list:
                        # polygon -- a single object might consist of multiple parts
                        # we merge all parts into one mask rle code
                        rles = maskUtils.frPyObjects(segm, height, width)
                        rle = maskUtils.merge(rles)
                    elif type(segm['counts']) == list:
                        # uncompressed RLE
                        rle = maskUtils.frPyObjects(segm, height, width)
                    else:
                        # rle
                        rle = segm
                    mask = maskUtils.decode(rle)
                    blend = image.copy()
                    blend[mask == 0] = 0
                    patch = blend[y:(y+h), x:(x+w), :]

                    save_path = os.path.join(save_dir, str(category))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, img_name[:-4]+'_'+str(j)+'.jpg'), patch)
                except:
                    continue


if __name__ == '__main__':
    json_file = '/versa/dyy/dataset/ADE/indoor_ins_val.json'
    img_dir = '/versa/dyy/dataset/ADE/panoptic/val'
    save_dir = '/versa/dyy/dataset/ADE/panoptic/val_gt'
    get_defect_patch(json_file, img_dir, save_dir)