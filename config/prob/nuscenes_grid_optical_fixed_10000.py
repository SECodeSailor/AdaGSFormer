_base_ = [
    "../_base_/misc.py",
    "../_base_/surroundocc.py",
]


# ---------------------------------------------------------------------------
# Dataset and optimization
# ---------------------------------------------------------------------------
data_type = "nus"
input_shape = (1600, 864)
data_aug_conf = dict(
    resize_lim=(1.0, 1.0),
    final_dim=input_shape[::-1],
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True,
)
train_dataset_config = dict(data_aug_conf=data_aug_conf)
val_dataset_config = dict(data_aug_conf=data_aug_conf)

optimizer = dict(
    optimizer=dict(
        type="AdamW",
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={"img_backbone": dict(lr_mult=0.1)},
    ),
)
grad_max_norm = 35

pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scene_size = [50, 50, 4]
occupancy_voxel_size = 0.5

pc_extent = [pc_range[i + 3] - pc_range[i] for i in range(3)]
gaussian_grid_resolution = [
    pc_extent[i] / scene_size[i]
    for i in range(3)
]
occupancy_shape = [
    int(round(pc_extent[i] / occupancy_voxel_size))
    for i in range(3)
]
num_initial_gaussians = scene_size[0] * scene_size[1] * scene_size[2]

overlap_ratio = 1.25
scale_min_ratio = 0.01
gaussian_scale_range = [
    min(gaussian_grid_resolution) * scale_min_ratio,
    max(gaussian_grid_resolution) * overlap_ratio,
]

embed_dims = 128
num_groups = 4
num_levels = 4
num_decoder = 4
num_single_frame_decoder = 1
use_deformable_func = True
semantics = True
semantic_dim = 18
empty_label = 17
empty_semantic_init = 0.6931471805599453
include_opa = True
phi_activation = "sigmoid"
xyz_coordinate = "cartesian"

load_from = "/ckpt/r101_dcn_fcos3d_pretrain.pth"

loss = dict(
    type="MultiLoss",
    loss_cfgs=[
        dict(
            type="BinaryCrossEntropyLoss",
            weight=1.0,
            empty_label=empty_label,
            ignore_label=255,
            class_weights=[1.0, 1.0],
            balance_pos_neg=False,
        ),
        dict(
            type="OccupancyLoss",
            weight=1.0,
            empty_label=empty_label,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0,
            ),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735,
                0.94637502, 1.10087012, 1.26960524, 1.06258364,
                1.18901900, 1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.89364310, 0.85486129,
                0.85278290, 0.50000000,
            ],
            ignore_empty=False,
            lovasz_use_softmax=False,
        ),
    ],
)

loss_input_convertion = dict(
    pred_occ="pred_occ",
    bin_logits="occ_prob",
    # density="density",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask",
)


decoder_stage = [
    "identity",
    "deformable",
    "add",
    "norm",

    "identity",
    "ffn",
    "add",
    "norm",

    "identity",
    "spconv",
    "add",
    "norm",

    "identity",
    "ffn",
    "add",
    "norm",

    "refine",
]
operation_order = decoder_stage * num_decoder


model = dict(
    type="AdaGSFormer",
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        with_cp=True,
        dcn=dict(
            type="DCNv2",
            deform_groups=1,
            fallback_on_stride=False,
        ),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    lifter=dict(
        type="GridGaussianLifter",
        pc_range=pc_range,
        embed_dims=embed_dims,
        scene_size=scene_size,
        scale_range=gaussian_scale_range,
        scale_min_ratio=scale_min_ratio,
        anchor_grad=True,
        feat_grad=True,
        semantics=semantics,
        semantic_dim=semantic_dim,
        empty_label=empty_label,
        empty_semantic_init=empty_semantic_init,
        overlap_ratio=overlap_ratio,
        include_opa=include_opa,
        init_opacity=0.01,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
    ),
    encoder=dict(
        type="GaussianOccEncoder",
        anchor_encoder=dict(
            type="SparseGaussian3DEncoder",
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim,
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.1,
            add_identity=False,
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregation",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="none",
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                embed_dims=embed_dims,
                num_learnable_pts=6,
                learnable_fixed_scale=6.0,
                fix_scale=[
                    [0.0, 0.0, 0.0],
                    [0.45, 0.0, 0.0],
                    [-0.45, 0.0, 0.0],
                    [0.0, 0.45, 0.0],
                    [0.0, -0.45, 0.0],
                    [0.0, 0.0, 0.45],
                    [0.0, 0.0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=gaussian_scale_range,
                xyz_activation="sigmoid",
                scale_activation="sigmoid",
            ),
        ),
        refine_layer=dict(
            type="ResidualGaussianRefinementModule",
            embed_dims=embed_dims,
            pc_range=pc_range,
            scene_size=scene_size,
            scale_range=gaussian_scale_range,
            scale_min_ratio=scale_min_ratio,
            overlap_ratio=overlap_ratio,
            position_step_ratio=0.5,
            scale_delta_factor=1.0,
            rotation_delta_factor=0.5,
            opacity_delta_factor=1.0,
            semantic_delta_factor=1.0,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            semantics_activation="identity",
            xyz_activation="sigmoid",
            scale_activation="sigmoid",
        ),
        spconv_layer=dict(
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.5, 0.5, 0.5], # 0.5
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
            use_multi_layer=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=operation_order,
    ),
    head=dict(
        type="GaussianHead",
        apply_loss_type="fixed_1_2_3",
        num_classes=18,
        empty_label=17,
        empty_args=None,
        with_empty=False,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        use_optical_density=True,
        optical_backend="cuda",
        explicit_empty_semantics=True,
        fuse_explicit_empty_with_occupancy=True,
        combine_geosem=False,
        cuda_kwargs=dict(
            scale_multiplier=5.0,
            H=occupancy_shape[0],
            W=occupancy_shape[1],
            D=occupancy_shape[2],
            pc_min=pc_range[:3],
            grid_size=occupancy_voxel_size,
        ),
    ),
)
