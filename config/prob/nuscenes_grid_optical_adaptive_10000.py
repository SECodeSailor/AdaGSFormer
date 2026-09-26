_base_ = ["./nuscenes_grid_optical_merge_10000.py"]

scene_size = [50, 50, 4]
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
occupancy_voxel_size = 0.5
pc_extent = [pc_range[i + 3] - pc_range[i] for i in range(3)]
gaussian_grid_resolution = [
    pc_extent[i] / scene_size[i]
    for i in range(3)
]
overlap_ratio = 1.25
scale_min_ratio = 0.01
split_offset_ratio = 2.0 ** -0.5
gaussian_scale_range = [
    min(gaussian_grid_resolution) * scale_min_ratio,
    max(gaussian_grid_resolution) * overlap_ratio,
]

embed_dims = 128
num_decoder = 4
empty_label = 17
max_epochs = 35


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
            empty_label=17,
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
        dict(
            type="GaussianComplexityLoss",
            weight=1,
            pc_range=pc_range,
            voxel_size=occupancy_voxel_size,
            num_classes=18,
            empty_label=17,
            ignore_label=255,
            sampling_levels=(-1.0, 0.0, 1.0),
            sampling_scale=1.0,
            min_loss_weight=0.25,
            use_camera_mask=True,
            enable_split_diagnostics=False,
            split_offset_ratio=split_offset_ratio,
            split_gain_eps=1e-8,
            split_gain_ratio_low_threshold=0.05,
        ),
    ],
)

loss_input_convertion = dict(
    pred_occ="pred_occ",
    bin_logits="occ_prob",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask",
    control_result="control_result",
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
operation_order = (
    (decoder_stage + ["adaptive_control"]) * (num_decoder - 1)
    + decoder_stage
)

model = dict(
    encoder=dict(
        adaptive_control_layer=dict(
            type="AdaptiveGaussianControl",
            embed_dims=embed_dims,
            score_hidden_dims=embed_dims,
            absolute_low=0.05,
            absolute_high=0.20,
            k_low=1.0,
            k_high=1.0,
            max_low_ratio=0.30,
            max_high_ratio=0.30,
            warmup_iters=500, # 1000
            pc_range=pc_range,
            scale_range=gaussian_scale_range,
            split_offset_ratio=split_offset_ratio,
            enable_split=True,
            enable_delete=True,
            enable_merge=True,
            opacity_delete_threshold=0.01,
            density_mass_delete_threshold=0.005,
            merge_knn=8,
            merge_chunk_size=512,
            merge_geometry_threshold=0.90,
            merge_semantic_threshold=0.95,
            split_stages=(0, 1, 2),
            delete_stages=(1, 2),
            merge_stages=(1, 2),
            require_consecutive_prune_evidence=True,
            prune_min_consecutive_stages=2,
            split_generator=dict(
                type="LearnedGaussianSplitGenerator",
                embed_dims=embed_dims,
                hidden_dims=embed_dims,
                semantic_dim=18,
                pc_range=pc_range,
                scale_range=gaussian_scale_range,
                split_offset_ratio=split_offset_ratio,
                split_scale_ratio=split_offset_ratio,
            ),
        ),
        share_adaptive_control_layer=True,
        num_decoder=num_decoder,
        operation_order=operation_order,
    ),
)
