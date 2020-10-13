# model settings
model = dict(
    type='SOLO',
    # pretrained='/home/dingyangyang/pretrained_models/resnest101-22405ba7.pth',
    pretrained='/home/dingyangyang/pretrained_models/resnest101_tuned.pth',
    backbone=dict(
        type='ResNeSt',
        depth=101,
        frozen_stages=1),
    neck=dict(
        type='BiFPN_Lite',  # P2 ~ P6
        is_efficientnet=False,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_repeats=1,
        freeze_params=False),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=0,
    #     num_outs=5),
    bbox_head=dict(
        type='SOLOV2Head',
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
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(640, 640), (576, 576), (512, 512),
                    (448, 448), (384, 384)],
         multiscale_mode='value',
         keep_ratio=False),
    # dict(type='Resize',
    #      img_scale=[(852, 512), (852, 480), (852, 448),
    #                 (852, 416), (852, 384), (852, 352)],
    #      multiscale_mode='value',
    #      keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='ImgAug', aug_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        # img_scale=(852, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='Resize', keep_ratio=True),
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
        ann_file=data_root + 'indoor_ins_train_4.json',
        img_prefix=data_root + 'panoptic/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'indoor_ins_val.json',
        img_prefix=data_root + 'panoptic/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'indoor_ins_val.json',
        img_prefix=data_root + 'panoptic/val/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
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
work_dir = './work_dirs/ade_resnest_stage2_2'
load_from = './work_dirs/ade_resnest_stage1_2/epoch_46.pth'
resume_from = None
workflow = [('train', 1)]
