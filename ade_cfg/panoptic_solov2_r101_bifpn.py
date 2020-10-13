# model settings
model = dict(
    type='SOLO',
    # pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        # dcn=dict(
        #     type='DCN',
        #     deformable_groups=1,
        #     fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True)
        ),
    neck=dict(
        type='BiFPN_Lite',  # P2 ~ P6
        is_efficientnet=False,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_repeats=1,
        freeze_params=False),
    bbox_head=dict(
        type='SOLOV2HeadPanoptic',
        num_classes=22,
        in_channels=256,
        stacked_convs=4,
        use_dcn_in_tower=True,
        type_dcn='DCN',
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=3.0),
        loss_ssim=dict(
            type='SSIMLoss',
            window_size=11,
            size_average=True,
            loss_weight=2.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_semantic=dict(
            type='CrossEntropyLoss',
            ignore_index=255,
            loss_weight=0.5),
        use_ppm=False,
        freeze_params=False
    ))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/versa/dyy/dataset/ADE/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize',
         img_scale=[(640, 640), (576, 576), (512, 512),
                    (448, 448), (384, 384)],
         multiscale_mode='value',
         keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='ImgAug', aug_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'indoor_ins_train_new.json',
        img_prefix=data_root + 'panoptic/train/',
        seg_prefix=data_root + 'panoptic/train_seg/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'indoor_ins_val.json',
        img_prefix=data_root + 'panoptic/val/',
        seg_prefix=data_root + 'panoptic/val_seg/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'indoor_ins_val.json',
        img_prefix=data_root + 'panoptic/val/',
        seg_prefix=data_root + 'panoptic/val_seg/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ade_indoor_bifpn_panoptic'
load_from = './work_dirs/ade_indoor_bifpn/epoch_36.pth'
# load_from = '../pretrained_models/solov2_r101_3x.pth'
resume_from = None
workflow = [('train', 1)]
