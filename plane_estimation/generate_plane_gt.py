import os, glob, shutil
import numpy as np
from PIL import Image
import open3d as o3d
import cv2


FocalLength, centerX, centerY = 517.97, 320, 240
HEIGHT = 480
WIDTH = 640


def planeParameters(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # try:
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd = pcd.select_by_index(ind)
    planeParam, inliers = pcd.segment_plane(distance_threshold=10, ransac_n=10, num_iterations=100)
    return planeParam
    # except:
    #     return None


if __name__ == '__main__':
    root = '/versa/dyy/dataset/nyu_depth_v2/'
    mask_dir = os.path.join(root, '2channels')
    img_dir = os.path.join(root, 'sync')

    save_dir = os.path.join(root, 'train')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(root, 'livingroom.txt'), 'r') as f:
        data = f.readlines()

    for i, line in enumerate(data):
        name, img_path, depth_path = line.strip().split(' ')
        print(i, name)
        mask_path = os.path.join(mask_dir, name)
        img_path = os.path.join(img_dir, img_path)
        depth_path = os.path.join(img_dir, depth_path)

        depth = np.array(Image.open(depth_path))
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        instance = mask[:, :, 0]
        semantic = mask[:, :, 1]
        planes, masks, cates = [], [], []

        points = []
        for v in range(HEIGHT):
            for u in range(WIDTH):
                Z = depth[v][u]
                X = (u - centerX) * Z / FocalLength
                Y = (v - centerY) * Z / FocalLength
                points.append([X, Y, Z])
        points = np.array(points).reshape((HEIGHT, WIDTH, 3))

        for ins_id in np.unique(instance):
            if ins_id == 0:
                continue
            ins_mask = (instance == ins_id)
            cate_id = semantic[ins_mask][0]
            if cate_id in [1, 2, 8]:
                ins_mask = cv2.resize(ins_mask.astype(np.uint8), (WIDTH, HEIGHT))
                ins_points = points[ins_mask == 1]

                if len(ins_points) > 200:
                    planesPar = planeParameters(np.array(ins_points))
                    print(planesPar)
                    if planesPar is not None:
                        planes.append(planesPar[:3].tolist())
                        masks.append(ins_mask)
                        cates.append(cate_id)

        if len(planes) != 0:
            np.savez(os.path.join(save_dir, name.replace('.png', '.npz')),
                     image=img, depth=depth, plane=np.array(planes), num_planes=len(planes),
                     semantics=cates, instances=masks, img_path=img_path)
