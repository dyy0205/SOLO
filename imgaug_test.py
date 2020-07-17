import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
images[:, 64, 64, :] = 255
bbs = [
    [ia.BoundingBox(x1=10.5, y1=15.5, x2=30.5, y2=50.5)],
    [ia.BoundingBox(x1=10.5, y1=20.5, x2=50.5, y2=50.5),
     ia.BoundingBox(x1=40.5, y1=75.5, x2=70.5, y2=100.5)]
]

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.05*255),
    iaa.Affine(translate_px={"x": (1, 5)})
])

images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
print(bbs_aug)
print(bbs_aug[0][0].coords[0])