# model settings
model = dict(
    type='SOLO',
    pretrained='/home/dingyangyang/pretrained_models/tf_efficientnet_lite3_tuned.pth',
    backbone=dict(
        type='EfficientNet_Lite',
        model_name='efficientnet-b3',
        num_stages=7,
        out_indices=(1, 2, 4, 6),  # C2, C3, C4, C5
        frozen_stages=-1),
    neck=dict(
        type='BiFPN_Lite',  # P2 ~ P6
        compound_coef=3,
        num_repeats=1,
        out_channels=160,
        freeze_params=False),
    bbox_head=dict(
        type='SOLOAttentionHead',
        num_classes=5,
        in_channels=160,
        stacked_convs=4,
        seg_feat_channels=160,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        loss_ins=dict(
            type='DiceLoss',
            use_sigmoid=True,
            loss_weight=2.0),
        # loss_mask=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     loss_weight=3.0),
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
data_root = '/home/versa/dataset/MSCOCO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(512, 512), (448, 448), (384, 384)],
         multiscale_mode='value',
         keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
    imgs_per_gpu=11,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
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
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/solov2_attention_label_align2_assim/'
load_from = './work_dirs/solov2_attention_label_align2_assim/stage1_epoch_23_0.392.pth'
# load_from = '/home/dingyangyang/pretrained_models/solo2-lite3_bifpn.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
