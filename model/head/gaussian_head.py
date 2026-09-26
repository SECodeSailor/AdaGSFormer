import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from ..utils.utils import get_rotation_matrix
from .optical_density_g2v import OpticalDensityAggregatorReference


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=17,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        use_optical_density=False,
        optical_backend='cuda',
        optical_reference_kwargs=None,
        combine_geosem=False,
        explicit_empty_semantics=False,
        fuse_explicit_empty_with_occupancy=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        self.num_classes = num_classes
        cuda_kwargs = cuda_kwargs or {}
        self.use_optical_density = bool(use_optical_density)
        self.optical_backend = optical_backend
        if self.use_optical_density:
            self.use_localaggprob = True
            if optical_backend == 'cuda':
                import local_aggregate_optical_tiled
                if getattr(
                    local_aggregate_optical_tiled,
                    'AGGREGATION_MODE',
                    None,
                ) != 'optical_density_tiled':
                    raise RuntimeError(
                        "The local_aggregate_optical_tiled CUDA extension is "
                        "stale. Rebuild model/head/localagg_optical_tiled."
                    )
                self.aggregator = local_aggregate_optical_tiled.LocalAggregator(
                    **cuda_kwargs)
            elif optical_backend == 'torch':
                reference_kwargs = optical_reference_kwargs or {}
                self.aggregator = OpticalDensityAggregatorReference(
                    **reference_kwargs)
            elif optical_backend == 'opacity':
                import local_aggregate_s2go_tiled
                self.aggregator = local_aggregate_s2go_tiled.LocalAggregator(
                    **cuda_kwargs)
            elif optical_backend == 'gf2':
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(
                    **cuda_kwargs)
            else:
                raise ValueError(
                    "optical_backend must be 'cuda' or 'torch', got "
                    f"{optical_backend!r}"
                )
        elif use_localaggprob:
            self.use_localaggprob = True
            if use_localaggprob_fast:
                import local_aggregate_prob_fast
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)
            else:
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)
        else:
            self.use_localaggprob = False
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(**cuda_kwargs)
        
        self.combine_geosem = bool(combine_geosem)
        self.explicit_empty_semantics = bool(explicit_empty_semantics)
        self.fuse_explicit_empty_with_occupancy = bool(
            fuse_explicit_empty_with_occupancy
        )
        if self.explicit_empty_semantics:
            if not self.use_localaggprob:
                raise ValueError(
                    "explicit_empty_semantics requires a probability-style "
                    "aggregator that returns semantics, occupancy and density"
                )
            if with_empty:
                raise ValueError(
                    "explicit_empty_semantics cannot be combined with the "
                    "legacy artificial empty Gaussian"
                )
            if self.combine_geosem:
                raise ValueError(
                    "explicit_empty_semantics already contains the empty "
                    "class and must not use combine_geosem"
                )
        elif self.fuse_explicit_empty_with_occupancy:
            raise ValueError(
                "fuse_explicit_empty_with_occupancy requires "
                "explicit_empty_semantics=True"
            )
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def fuse_semantics_and_occupancy(
        self,
        semantic_logits,
        occupancy_probability,
    ):
        """Build a normalized joint semantic-occupancy distribution.

        ``semantic_logits`` describes the class distribution contributed by
        the Gaussians at each voxel, while ``occupancy_probability`` describes
        whether optical density exists there.  The occupied probability gates
        every semantic class and the complementary probability is assigned to
        the explicit empty class:

            q(c)     = p_occ * p_sem(c),                 c != empty
            q(empty) = 1 - p_occ + p_occ * p_sem(empty)

        This keeps the Gaussian's learnable empty semantic state while making
        low optical density an explicit and differentiable source of empty
        probability.  The result is non-negative and sums to one per voxel.
        """
        if semantic_logits.shape[-1] != self.num_classes:
            raise ValueError(
                "semantic_logits must end with num_classes="
                f"{self.num_classes}, got {semantic_logits.shape}"
            )
        if occupancy_probability.shape != semantic_logits.shape[:-1]:
            raise ValueError(
                "occupancy_probability must match semantic_logits without "
                f"the class dimension, got {occupancy_probability.shape} "
                f"and {semantic_logits.shape}"
            )
        if not 0 <= self.empty_label < self.num_classes:
            raise ValueError(
                f"empty_label={self.empty_label} is outside "
                f"[0, {self.num_classes})"
            )

        semantic_probability = semantic_logits.softmax(dim=-1)
        occupancy_probability = occupancy_probability.clamp(0.0, 1.0)
        empty_basis = torch.zeros(
            self.num_classes,
            dtype=semantic_probability.dtype,
            device=semantic_probability.device,
        )
        empty_basis[self.empty_label] = 1.0
        return (
            occupancy_probability.unsqueeze(-1) * semantic_probability
            + (1.0 - occupancy_probability).unsqueeze(-1) * empty_basis
        )

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        semantic_features = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(
                semantic_features[..., :1], requires_grad=False
            )
        if self.explicit_empty_semantics:
            if semantic_features.shape[-1] != self.num_classes:
                raise ValueError(
                    "Explicit-empty Gaussian semantics must have exactly "
                    f"num_classes={self.num_classes} raw logits, got "
                    f"{semantic_features.shape[-1]}"
                )
            # Keep raw logits.  Optical G2V performs a density-weighted mean
            # and the voxel semantic loss applies softmax/cross-entropy.
        elif self.with_emtpy:
            assert semantic_features.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                semantic_features = torch.cat(
                    [
                        torch.zeros_like(semantic_features[..., :1]),
                        semantic_features,
                    ],
                    dim=-1,
                )
            else:
                semantic_features = torch.cat(
                    [
                        semantic_features,
                        torch.zeros_like(semantic_features[..., :1]),
                    ],
                    dim=-1,
                )
            means = torch.cat([means, self.empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale], dim=1)
            rotations = torch.cat([rotations, self.empty_rot], dim=1)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            semantic_features = torch.cat(
                [semantic_features, empty_sem], dim=1
            )
            origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1)
        elif self.use_localaggprob:
            assert semantic_features.shape[-1] == self.num_classes - 1
            semantic_features = semantic_features.softmax(dim=-1)
            if 'kitti' in self.dataset_type:
                semantic_features = torch.cat(
                    [
                        torch.zeros_like(semantic_features[..., :1]),
                        semantic_features,
                    ],
                    dim=-1,
                )
            else:
                semantic_features = torch.cat(
                    [
                        semantic_features,
                        torch.zeros_like(semantic_features[..., :1]),
                    ],
                    dim=-1,
                )

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = torch.linalg.inv(Cov.float()).to(Cov.dtype)
        aabb_scales = torch.sqrt(
            torch.diagonal(Cov.float(), dim1=-2, dim2=-1).clamp_min(1e-12)
        ).to(scales.dtype)
        return means, origi_opa, semantic_features, aabb_scales, CovInv

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        prediction = []
        semantic_logits = []
        bin_logits = []
        density = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        for idx in apply_loss_layers:
            gaussians = representation[idx]['gaussian']

            (
                means,
                origi_opa,
                semantic_features,
                scales,
                CovInv,
            ) = self.prepare_gaussian_args(gaussians)
            bs, g = means.shape[:2]

            semantics = self.aggregator(
                sampled_xyz.clone().float(), 
                means, 
                origi_opa.reshape(bs, g),
                semantic_features,
                scales,
                CovInv) # 1, c, n
            if self.use_localaggprob:
                if self.combine_geosem:
                    sem = semantics[0][:, :-1] * semantics[1].unsqueeze(-1)
                    geo = 1 - semantics[1].unsqueeze(-1)
                    geosem = torch.cat([sem, geo], dim=-1)
                elif self.fuse_explicit_empty_with_occupancy:
                    geosem = self.fuse_semantics_and_occupancy(
                        semantics[0],
                        semantics[1],
                    )
                else:
                    geosem = semantics[0]

                prediction.append(geosem[None].transpose(1, 2))
                semantic_logits.append(semantics[0][None].transpose(1, 2))
                bin_logits.append(semantics[1][None])
                density.append(semantics[2][None])
            else:
                prediction.append(semantics[None].transpose(1, 2))

        if self.fuse_explicit_empty_with_occupancy:
            # The joint distribution already contains the empty probability;
            # applying another hard occupancy threshold would discard its
            # probabilistic calibration.
            final_prediction = prediction[-1].argmax(dim=1)
        elif self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = prediction[-1].argmax(dim=1)
            final_occupancy = bin_logits[-1] > threshold
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = prediction[-1].argmax(dim=1)
        
        return {
            # Raw conditional semantic logits and optical occupancy outputs
            # remain available for calibration and representation diagnosis.
            'sem_logits': (
                semantic_logits if self.use_localaggprob else prediction
            ),
            'occ_prob': bin_logits,
            # Complete semantic-occupancy prediction consumed by the semantic
            # loss.  With explicit fusion this is a normalized distribution.
            'pred_occ': prediction,
            'joint_occ_prob': (
                prediction
                if self.fuse_explicit_empty_with_occupancy
                else []
            ),
            # Backward-compatible aliases consumed by existing binary losses.
            'bin_logits': bin_logits,
            'density': density,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'final_occ': final_prediction,
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation],
        }
