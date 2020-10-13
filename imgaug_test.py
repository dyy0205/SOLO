import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

# ia.seed(123456)
images = np.zeros((2, 128, 128, 3), dtype=np.uint8)  # two example images
images[:, 64, 64, :] = 255
bbs = [
    [ia.BoundingBox(x1=10.5, y1=15.5, x2=30.5, y2=50.5)],
    [ia.BoundingBox(x1=10.5, y1=20.5, x2=50.5, y2=50.5),
     ia.BoundingBox(x1=40.5, y1=75.5, x2=70.5, y2=100.5)]
]

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    # iaa.PadToFixedSize(width=crop_size[0], height=crop_size[1]),
    # iaa.CropToFixedSize(width=crop_size[0], height=crop_size[1]),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        # scale images to 90-110% of their size, individually per axis
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        # translate by -10 to +10 percent (per axis)
        rotate=(-5, 5),  # rotate by -5 to +5 degrees
        shear=(-3, 3),  # shear by -3 to +3 degrees
    ),
    iaa.Cutout(nb_iterations=(1, 5), size=0.2, cval=0, squared=False),
    # iaa.Sometimes(0.5,
    #                       iaa.OneOf([
    #                           iaa.GaussianBlur((0.0, 3.0)),
    #                           iaa.MotionBlur(k=(3, 20)),
    #                       ]),
    #                       ),
    #         iaa.Sometimes(0.5,
    #                       iaa.OneOf([
    #                           # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    #                           iaa.MultiplyBrightness((0.5, 1.5)),
    #                           iaa.LinearContrast((0.5, 2.0), per_channel=0.2),
    #                           iaa.BlendAlpha((0., 1.), iaa.HistogramEqualization()),
    #                           iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=0.2),
    #                       ]),
    #                       ),
])
# img = np.array(Image.open('test2.png'))[np.newaxis, :, :, :3]
# for i in range(1000):
#     img_aug = seq(images=img)
#     Image.fromarray(np.array(img_aug[0])).save('plot/{}.png'.format(i))
mask = np.array(Image.open('0.png'))[np.newaxis, :, :, np.newaxis]
for i in range(100):
    mask_aug = seq(images=mask)
    Image.fromarray(np.array(mask_aug[0, :, :, 0]) * 255).save('plot/mask_{}.png'.format(i))
# images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
# print(bbs_aug)