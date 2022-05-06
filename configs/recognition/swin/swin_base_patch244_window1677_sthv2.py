_base_ = [
    '../../_base_/models/swin/swin_base.py', '../../_base_/default_runtime.py'
]

_SIZE = 112

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/work/07796/chenwy/maverick2/data/ssv2/videos'
data_root_val = '/work/07796/chenwy/maverick2/data/ssv2/videos'
# ann_file_train = '/work/07796/chenwy/maverick2/data/ssv2/sthv2_train_list_videos.txt'
ann_file_train = '/work/07796/chenwy/maverick2/data/ssv2/sthv2_train_list_videos_imbalance.txt'
# ann_file_train = '/work/07796/chenwy/maverick2/data/ssv2/sthv2_test_list_videos.txt'
ann_file_val = '/work/07796/chenwy/maverick2/data/ssv2/sthv2_val_list_videos.txt'
ann_file_test = '/work/07796/chenwy/maverick2/data/ssv2/sthv2_val_list_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, frame_uniform=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(_SIZE, _SIZE), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Imgaug', transforms=[dict(type='RandAugment', n=4, m=7)]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomErasing', probability=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=_SIZE),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, _SIZE)),
    dict(type='ThreeCrop', crop_size=_SIZE),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=1,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, by_epoch=True, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
# TODO
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05,
# optimizer = dict(type='AdamW', lr=3e-3, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
# total_epochs = 60
total_epochs = 5

# runtime settings
# checkpoint_config = dict(interval=1)
checkpoint_config = dict(interval=1, by_epoch=True, save_optimizer=False, max_keep_ckpts=1, save_last=True)
# work_dir = '/home1/07796/chenwy/Video-Swin-Transformer/work_dirs/baseline' # TODO
work_dir = '/work/07796/chenwy/maverick2/Video-Swin-Transformer/work_dirs/baseline_imbalanced' # TODO
# work_dir = '/work/07796/chenwy/maverick2/Video-Swin-Transformer/work_dirs/lora1_fixed' # TODO
# work_dir = '/work/07796/chenwy/maverick2/Video-Swin-Transformer/work_dirs/lora8' # TODO
# work_dir = '/work/07796/chenwy/maverick2/Video-Swin-Transformer/work_dirs/lora64_nc' # TODO
# work_dir = '/work/07796/chenwy/maverick2/Video-Swin-Transformer/work_dirs/nc_imbalanced' # TODO
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

model=dict(backbone=dict(patch_size=(2,4,4), window_size=(16,7,7), drop_path_rate=0.4),
           cls_head=dict(num_classes=174),
           test_cfg=dict(max_testing_views=2),
           train_cfg=dict(blending=dict(type='LabelSmoothing', num_classes=174, smoothing=0.1)))
