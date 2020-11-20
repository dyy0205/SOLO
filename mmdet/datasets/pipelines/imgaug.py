import cv2
import numpy as np

np.random.bit_generator = np.random._bit_generator
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from ..registry import PIPELINES


@PIPELINES.register_module
class ImgAug(object):

    def __init__(self, aug_ratio=0.5):
        self.aug_ratio = aug_ratio
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.

        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                # iaa.Flipud(0.1),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                # sometimes(iaa.CropAndPad(
                #     percent=(-0.05, 0.05),
                #     pad_cval=(0, 255)
                # )),
                # sometimes(iaa.Affine(
                #     scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                #     # scale images to 80-120% of their size, individually per axis
                #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                #     # translate by -20 to +20 percent (per axis)
                #     rotate=(-5, 5),  # rotate by -45 to +45 degrees
                #     shear=(-3, 3),  # shear by -16 to +16 degrees
                #     order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                #     cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                # )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 5)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 5)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                                   iaa.MotionBlur(k=(3, 7))
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(1.0, 2.0)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.0)),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               # iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                               #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               # iaa.OneOf([
                               #     iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=True),
                               # ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.JpegCompression(compression=(50, 90)),
                               # iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                   # change brightness of images (by -10 to 10 of original value)
                                   iaa.Add((-10, 10), per_channel=0.5),
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   # iaa.BlendAlphaFrequencyNoise(
                                   #     exponent=(-4, 0),
                                   #     foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   #     background=iaa.LinearContrast((0.5, 2.0))
                                   # )
                               ]),
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.075)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __call__(self, results):
        if np.random.rand() > self.aug_ratio:
            # img: (h,w,3)  gt_bboxes: (n,4)  gt_masks: (n,h,w)
            img, bboxes, masks = [results[k] for k in ('img', 'gt_bboxes', 'gt_masks')]

            segmap = masks.transpose(1, 2, 0)
            bbs = []
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                bbs.append(BoundingBox(x1, y1, x2, y2))

            img_aug, bbox_aug, seg_aug = self.seq(images=img[np.newaxis, ...],
                                                  bounding_boxes=bbs,
                                                  segmentation_maps=segmap[np.newaxis, ...])

            img_aug = img_aug[0]
            seg_aug = seg_aug[0].transpose(2, 0, 1)
            bbs_aug = []
            for bbox in bbox_aug:
                x1, y1 = bbox.coords[0]
                x2, y2 = bbox.coords[1]
                bbs_aug.append([x1, y1, x2, y2])

            results['img'] = img_aug
            results['gt_bboxes'] = np.array(bbs_aug)
            results['gt_masks'] = seg_aug

            # # Visualize Augmented Results
            # h, w = img_aug.shape[:2]
            # segmap = np.zeros((h, w), dtype=np.uint8)
            # for i in range(len(seg_aug)):
            #     temp = np.array(Image.fromarray(seg_aug[i]).convert('P'), dtype=np.uint8)
            #     segmap[temp == 1] = i + 1
            # segmap = SegmentationMapsOnImage(segmap, shape=img_aug.shape)
            # cv2.imwrite('aug_seg.jpg', segmap.draw_on_image(img_aug)[0])
            # print('seg success')
            #
            # bbox_aug = BoundingBoxesOnImage(bbox_aug, shape=img_aug.shape)
            # cv2.imwrite('aug_bbs.jpg', bbox_aug.draw_on_image(img_aug))
            # print('bbs success')

        return results
