import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import matrix_nms

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageInsDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageInsDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageInsDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if gt_semantic_seg is None:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)
        seg_inputs = outs + (img_meta, self.test_cfg)
        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result  

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.
        If rescale is False, then returned masks will fit the scale of imgs[0].
        """
        ori_shape = img_metas[0][0]['ori_shape'][:2]

        meta_result_list = []
        for img, img_meta in zip(imgs, img_metas):
            x = self.extract_feat(img)
            seg_preds, cate_preds = self.bbox_head(x, eval=True)
            img_shape = img_meta[0]['img_shape']
            img_result_list = self.bbox_head.get_seg_aug(seg_preds, cate_preds, img_shape, self.test_cfg)
            meta_result_list.append(img_result_list)

        img_output = []
        for img_result in zip(*meta_result_list):
            seg_masks, seg_preds, sum_masks, cate_scores, cate_labels = map(list, zip(*img_result))
            unified_size = tuple(seg_masks[0].shape[-2:])
            for i in range(1, len(seg_masks)):
                seg_masks[i] = F.interpolate(seg_masks[i].float().unsqueeze(0),
                                             size=unified_size,
                                             mode='bilinear',
                                             align_corners=False).squeeze(0)
                seg_preds[i] = F.interpolate(seg_preds[i].unsqueeze(0),
                                             size=unified_size,
                                             mode='bilinear',
                                             align_corners=False).squeeze(0)
                if img_metas[i][0]['flip']:
                    seg_masks[i] = torch.flip(seg_masks[i], dims=[2])
                    seg_preds[i] = torch.flip(seg_preds[i], dims=[2])
            seg_masks = torch.cat(seg_masks, dim=0)
            seg_preds = torch.cat(seg_preds, dim=0)
            sum_masks = torch.cat(sum_masks, dim=0)
            cate_scores = torch.cat(cate_scores, dim=0)
            cate_labels = torch.cat(cate_labels, dim=0)
            # import cv2
            # for i, seg_mask in enumerate(seg_masks):
            #     cv2.imwrite('/versa/dyy/SOLO/tta/{}.png'.format(i),
            #                 seg_mask.cpu().numpy().astype(np.uint8) * 255)

            # sort and keep top nms_pre
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > self.test_cfg.nms_pre:
                sort_inds = sort_inds[:self.test_cfg.nms_pre]
            seg_masks = seg_masks[sort_inds, :, :]
            seg_preds = seg_preds[sort_inds, :, :]
            sum_masks = sum_masks[sort_inds]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

            # Matrix NMS
            cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                     kernel=self.test_cfg.kernel, sigma=self.test_cfg.sigma,
                                     sum_masks=sum_masks)

            # filter.
            keep = cate_scores >= self.test_cfg.update_thr
            if keep.sum() == 0:
                return None
            seg_preds = seg_preds[keep, :, :]
            cate_scores = cate_scores[keep]
            cate_labels = cate_labels[keep]

            # sort and keep top_k
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > self.test_cfg.max_per_img:
                sort_inds = sort_inds[:self.test_cfg.max_per_img]
            seg_preds = seg_preds[sort_inds, :, :]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

            seg_masks = F.interpolate(seg_preds.unsqueeze(0),
                                      size=ori_shape,
                                      mode='bilinear',
                                      align_corners=False).squeeze(0)
            seg_masks = seg_masks > self.test_cfg.mask_thr
            output = (seg_masks, cate_labels, cate_scores)
            img_output.append(output)

        return img_output
