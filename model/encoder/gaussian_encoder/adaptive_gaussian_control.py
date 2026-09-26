import math
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .complexity_score import GaussianComplexityScore
from ...utils.safe_ops import safe_inverse_sigmoid
from ...utils.utils import get_rotation_matrix


class AdaptiveThresholdGenerator(nn.Module):

    def __init__(
        self,
        absolute_low: float = 0.05,
        absolute_high: float = 0.20,
        k_low: float = 1.0,
        k_high: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= absolute_low < absolute_high <= 1.0:
            raise ValueError(
                "absolute thresholds must satisfy "
                "0 <= absolute_low < absolute_high <= 1"
            )
        if k_low < 0 or k_high < 0:
            raise ValueError("k_low and k_high must be non-negative")
        self.absolute_low = float(absolute_low)
        self.absolute_high = float(absolute_high)
        self.k_low = float(k_low)
        self.k_high = float(k_high)

    def forward(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if scores.ndim != 2:
            raise ValueError(
                f"scores must have shape [B, N], got {scores.shape}"
            )
        if scores.shape[1] == 0:
            raise ValueError("Adaptive control requires at least one Gaussian")

        # Threshold selection is a discrete control decision.  Detaching here
        # keeps Score learning solely governed by its explicit GT supervision.
        statistics_scores = scores.detach().float()
        score_mean = statistics_scores.mean(dim=1, keepdim=True)
        score_variance = (
            statistics_scores - score_mean
        ).square().mean(dim=1, keepdim=True)
        score_std = torch.sqrt(score_variance.clamp_min(0.0))

        absolute_low = torch.full_like(score_mean, self.absolute_low)
        absolute_high = torch.full_like(score_mean, self.absolute_high)
        tau_1 = torch.minimum(
            absolute_low, score_mean - self.k_low * score_std
        ).clamp_(0.0, 1.0)
        tau_2 = torch.maximum(
            absolute_high, score_mean + self.k_high * score_std
        ).clamp_(0.0, 1.0)

        if torch.any(tau_1 >= tau_2):
            raise RuntimeError("Adaptive thresholds must satisfy tau_1 < tau_2")
        return tau_1, tau_2, score_mean, score_std


@MODELS.register_module()
class AdaptiveGaussianControl(BaseModule):
    OP_KEEP = 0
    OP_SPLIT_CHILD = 1
    OP_MERGED = 2

    def __init__(
        self,
        embed_dims: int = 256,
        score_hidden_dims: Optional[int] = None,
        score_dropout: float = 0.0,
        absolute_low: float = 0.05,
        absolute_high: float = 0.20,
        k_low: float = 1.0,
        k_high: float = 1.0,
        max_low_ratio: Optional[float] = 0.20,
        max_high_ratio: Optional[float] = 0.20,
        warmup_iters: int = 5000,
        enable_split: bool = False,
        enable_delete: bool = False,
        enable_merge: bool = False,
        split_stages: Optional[Sequence[int]] = None,
        delete_stages: Optional[Sequence[int]] = None,
        merge_stages: Optional[Sequence[int]] = None,
        require_consecutive_prune_evidence: bool = True,
        prune_min_consecutive_stages: int = 2,
        opacity_delete_threshold: float = 0.005,
        density_mass_delete_threshold: float = 0.001,
        merge_knn: int = 8,
        merge_chunk_size: int = 512,
        merge_geometry_threshold: float = 0.90,
        merge_semantic_threshold: float = 0.95,
        merge_eigenvalue_eps: float = 1e-8,
        merge_mass_eps: float = 1e-12,
        merge_probability_eps: float = 1e-8,
        pc_range: Optional[Sequence[float]] = None,
        scale_range: Optional[Sequence[float]] = None,
        split_offset_ratio: float = 1.0 / math.sqrt(2.0),
        split_generator: Optional[dict] = None,
        include_opa: bool = True,
        xyz_activation: str = "sigmoid",
        scale_activation: str = "sigmoid",
        position_eps: float = 1e-4,
        scale_eps: float = 1e-6,
        opacity_eps: float = 1e-6,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        self._validate_ratio(max_low_ratio, "max_low_ratio")
        self._validate_ratio(max_high_ratio, "max_high_ratio")

        self.max_low_ratio = max_low_ratio
        self.max_high_ratio = max_high_ratio
        self.warmup_iters = int(warmup_iters)
        self.enable_split = bool(enable_split)
        self.enable_delete = bool(enable_delete)
        self.enable_merge = bool(enable_merge)
        self.split_stages = self._normalize_stages(
            split_stages, "split_stages"
        )
        self.delete_stages = self._normalize_stages(
            delete_stages, "delete_stages"
        )
        self.merge_stages = self._normalize_stages(
            merge_stages, "merge_stages"
        )
        self.require_consecutive_prune_evidence = bool(
            require_consecutive_prune_evidence
        )
        if prune_min_consecutive_stages <= 0:
            raise ValueError(
                "prune_min_consecutive_stages must be positive"
            )
        self.prune_min_consecutive_stages = int(
            prune_min_consecutive_stages
        )
        self.include_opa = bool(include_opa)

        if not 0.0 <= opacity_delete_threshold < 1.0:
            raise ValueError(
                "opacity_delete_threshold must lie inside [0, 1)"
            )
        if density_mass_delete_threshold < 0.0:
            raise ValueError(
                "density_mass_delete_threshold must be non-negative"
            )
        self.opacity_delete_threshold = float(opacity_delete_threshold)
        self.density_mass_delete_threshold = float(
            density_mass_delete_threshold
        )

        if merge_knn <= 0:
            raise ValueError("merge_knn must be positive")
        if merge_chunk_size <= 0:
            raise ValueError("merge_chunk_size must be positive")
        if not 0.0 <= merge_geometry_threshold <= 1.0:
            raise ValueError("merge_geometry_threshold must lie in [0, 1]")
        if not 0.0 <= merge_semantic_threshold <= 1.0:
            raise ValueError("merge_semantic_threshold must lie in [0, 1]")
        if merge_eigenvalue_eps <= 0.0:
            raise ValueError("merge_eigenvalue_eps must be positive")
        if merge_mass_eps <= 0.0:
            raise ValueError("merge_mass_eps must be positive")
        if not 0.0 < merge_probability_eps < 1.0:
            raise ValueError("merge_probability_eps must lie inside (0, 1)")
        self.merge_knn = int(merge_knn)
        self.merge_chunk_size = int(merge_chunk_size)
        self.merge_geometry_threshold = float(merge_geometry_threshold)
        self.merge_semantic_threshold = float(merge_semantic_threshold)
        self.merge_eigenvalue_eps = float(merge_eigenvalue_eps)
        self.merge_mass_eps = float(merge_mass_eps)
        self.merge_probability_eps = float(merge_probability_eps)

        if not 0.0 < split_offset_ratio < 1.0:
            raise ValueError("split_offset_ratio must lie strictly inside (0, 1)")
        if xyz_activation not in ("sigmoid", "identity"):
            raise ValueError(f"Unsupported xyz_activation={xyz_activation!r}")
        if scale_activation not in ("sigmoid", "identity"):
            raise ValueError(f"Unsupported scale_activation={scale_activation!r}")
        if position_eps < 0.0 or scale_eps < 0.0:
            raise ValueError("position_eps and scale_eps must be non-negative")
        if not 0.0 < opacity_eps < 0.5:
            raise ValueError("opacity_eps must lie inside (0, 0.5)")

        self.split_offset_ratio = float(split_offset_ratio)
        self.split_scale_ratio = math.sqrt(
            1.0 - self.split_offset_ratio ** 2
        )
        self.xyz_activation = xyz_activation
        self.scale_activation = scale_activation
        self.position_eps = float(position_eps)
        self.scale_eps = float(scale_eps)
        self.opacity_eps = float(opacity_eps)
        self.encoding_unit_eps = 1e-4

        if self.enable_split or self.enable_merge:
            if pc_range is None or len(pc_range) != 6:
                raise ValueError(
                    "pc_range with 6 values is required for split/merge"
                )
            if scale_range is None or len(scale_range) != 2:
                raise ValueError(
                    "scale_range=[min, max] is required for split/merge"
                )
            pc_range_tensor = torch.as_tensor(pc_range, dtype=torch.float32)
            if torch.any(pc_range_tensor[3:] <= pc_range_tensor[:3]):
                raise ValueError("Each pc_range maximum must exceed its minimum")
            scale_min, scale_max = map(float, scale_range)
            if scale_min <= 0.0 or scale_max <= scale_min:
                raise ValueError("scale_range must satisfy 0 < min < max")
        else:
            pc_range_tensor = torch.zeros(6, dtype=torch.float32)
            scale_min, scale_max = 0.0, 1.0

        if (
            self.enable_split or self.enable_delete or self.enable_merge
        ) and not self.include_opa:
            raise ValueError(
                "Optical-density topology control requires include_opa=True"
            )

        self.register_buffer("pc_range", pc_range_tensor, persistent=False)
        self.register_buffer(
            "scale_range",
            torch.tensor([scale_min, scale_max], dtype=torch.float32),
            persistent=False,
        )
        self.score_head = GaussianComplexityScore(
            embed_dims=embed_dims,
            hidden_dims=score_hidden_dims,
            dropout=score_dropout,
        )
        self.threshold_generator = AdaptiveThresholdGenerator(
            absolute_low=absolute_low,
            absolute_high=absolute_high,
            k_low=k_low,
            k_high=k_high,
        )
        self.split_generator = (
            MODELS.build(split_generator)
            if self.enable_split and split_generator is not None
            else None
        )

    @staticmethod
    def _validate_ratio(value: Optional[float], name: str) -> None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be None or lie in [0, 1]")

    @staticmethod
    def _normalize_stages(
        stages: Optional[Sequence[int]], name: str
    ) -> Optional[Tuple[int, ...]]:
        """Validate zero-based adaptive-control stage indices."""
        if stages is None:
            return None
        normalized = tuple(int(stage) for stage in stages)
        if any(stage < 0 for stage in normalized):
            raise ValueError(f"{name} must contain non-negative indices")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must not contain duplicate indices")
        return normalized

    @staticmethod
    def _operation_enabled(
        globally_enabled: bool,
        enabled_stages: Optional[Tuple[int, ...]],
        stage_index: int,
    ) -> bool:
        return globally_enabled and (
            enabled_stages is None or stage_index in enabled_stages
        )

    @staticmethod
    def _global_iter_value(global_iter) -> Optional[int]:
        if global_iter is None:
            return None
        if isinstance(global_iter, torch.Tensor):
            if global_iter.numel() != 1:
                raise ValueError("global_iter tensor must contain one value")
            return int(global_iter.detach().item())
        return int(global_iter)

    @staticmethod
    def _cap_candidates(
        mask: torch.Tensor,
        scores: torch.Tensor,
        max_ratio: Optional[float],
        largest: bool,
    ) -> torch.Tensor:
        """Apply a cap only when threshold candidates exceed the ratio."""
        if max_ratio is None or max_ratio >= 1.0:
            return mask

        num_gaussians = mask.shape[1]
        if max_ratio <= 0.0:
            return torch.zeros_like(mask)
        max_candidates = max(1, int(num_gaussians * max_ratio))
        if max_candidates >= num_gaussians:
            return mask

        detached_scores = scores.detach().float()
        fill_value = -torch.inf if largest else torch.inf
        ranking_scores = torch.where(
            mask, detached_scores, torch.full_like(detached_scores, fill_value)
        )
        candidate_indices = torch.topk(
            ranking_scores,
            k=max_candidates,
            dim=1,
            largest=largest,
            sorted=False,
        ).indices
        selected_are_valid = mask.gather(1, candidate_indices)
        capped_mask = torch.zeros_like(mask)
        capped_mask.scatter_(1, candidate_indices, selected_are_valid)
        return capped_mask

    def _candidates_enabled(self, global_iter) -> bool:
        iteration = self._global_iter_value(global_iter)
        # Validation during training passes global_iter and must obey the same
        # warmup as training.  Standalone evaluation has no iteration and is
        # treated as evaluation of a trained checkpoint, so control is active.
        if iteration is not None:
            return iteration >= self.warmup_iters
        if not self.training:
            return True
        # During training an absent iteration is treated conservatively.  The
        # Score loss still trains, but no topology candidate is activated.
        return False

    def _encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        pc_min = self.pc_range[:3].to(device=xyz.device, dtype=xyz.dtype)
        pc_extent = (
            self.pc_range[3:] - self.pc_range[:3]
        ).to(device=xyz.device, dtype=xyz.dtype)
        xyz_unit = (xyz - pc_min) / pc_extent
        if self.xyz_activation == "sigmoid":
            return safe_inverse_sigmoid(xyz_unit)
        return xyz_unit.clamp(min=1e-6, max=1.0 - 1e-6)

    def _encode_scale(self, scales: torch.Tensor) -> torch.Tensor:
        scale_range = self.scale_range.to(
            device=scales.device, dtype=scales.dtype
        )
        scale_unit = (
            scales - scale_range[0]
        ) / (
            scale_range[1] - scale_range[0]
        )
        if self.scale_activation == "sigmoid":
            return safe_inverse_sigmoid(scale_unit)
        return scale_unit.clamp(min=1e-6, max=1.0 - 1e-6)

    def _split_feasibility(self, gaussian):
        """Return scale/position validity masks and proposed child geometry."""
        means = gaussian.means
        scales = gaussian.scales
        rotations = gaussian.rotations
        if means.ndim != 3 or means.shape[-1] != 3:
            raise ValueError(f"gaussian.means must be [B, N, 3], got {means.shape}")
        if scales.shape != means.shape:
            raise ValueError(
                f"gaussian.scales must match means, got {scales.shape}"
            )
        if rotations.shape != (*means.shape[:2], 4):
            raise ValueError(
                "gaussian.rotations must be [B, N, 4], got "
                f"{rotations.shape}"
            )

        largest_axis = scales.argmax(dim=-1)
        largest_scale = scales.gather(
            -1, largest_axis.unsqueeze(-1)
        ).squeeze(-1)

        # GaussianHead forms Cov = R^T diag(scale^2) R, hence a local
        # principal axis is a row of R in world coordinates.
        rotation_matrix = get_rotation_matrix(rotations)
        axis_direction = rotation_matrix.gather(
            -2,
            largest_axis[..., None, None].expand(*largest_axis.shape, 1, 3),
        ).squeeze(-2)
        offset = (
            largest_scale * self.split_offset_ratio
        ).unsqueeze(-1) * axis_direction
        child_mean_positive = means + offset
        child_mean_negative = means - offset

        child_scales = scales.clone()
        child_axis_scale = largest_scale * self.split_scale_ratio
        child_scales.scatter_(
            -1, largest_axis.unsqueeze(-1), child_axis_scale.unsqueeze(-1)
        )

        scale_range = self.scale_range.to(
            device=scales.device, dtype=scales.dtype
        )
        minimum_encodable_scale = scale_range[0] + (
            scale_range[1] - scale_range[0]
        ) * self.encoding_unit_eps
        scale_feasible = (
            child_axis_scale >= minimum_encodable_scale + self.scale_eps
        )

        pc_min = self.pc_range[:3].to(device=means.device, dtype=means.dtype)
        pc_max = self.pc_range[3:].to(device=means.device, dtype=means.dtype)
        encoding_margin = (pc_max - pc_min) * self.encoding_unit_eps
        lower = pc_min + encoding_margin + self.position_eps
        upper = pc_max - encoding_margin - self.position_eps
        positive_inside = (
            (child_mean_positive >= lower) & (child_mean_positive <= upper)
        ).all(dim=-1)
        negative_inside = (
            (child_mean_negative >= lower) & (child_mean_negative <= upper)
        ).all(dim=-1)
        children_inside = positive_inside & negative_inside
        return (
            scale_feasible,
            children_inside,
            child_mean_positive,
            child_mean_negative,
            child_scales,
        )

    @staticmethod
    def _select_gaussian(gaussian, selected: torch.Tensor):
        """Gather physical Gaussian fields for selected parents only."""
        if selected.ndim != 1 or selected.dtype != torch.bool:
            raise ValueError("selected must be a one-dimensional bool mask")
        return SimpleNamespace(
            means=gaussian.means[:, selected],
            scales=gaussian.scales[:, selected],
            rotations=gaussian.rotations[:, selected],
            opacities=gaussian.opacities[:, selected],
        )

    @staticmethod
    def _physical_covariance(
        scales: torch.Tensor, rotations: torch.Tensor
    ) -> torch.Tensor:
        """Build ``R^T diag(scale^2) R``, matching GaussianHead exactly."""
        rotation_matrix = get_rotation_matrix(rotations)
        return (
            rotation_matrix.transpose(-1, -2)
            @ torch.diag_embed(scales.square())
            @ rotation_matrix
        )

    def _moment_match_geometry(
        self,
        mean_i: torch.Tensor,
        mean_j: torch.Tensor,
        covariance_i: torch.Tensor,
        covariance_j: torch.Tensor,
        mass_i: torch.Tensor,
        mass_j: torch.Tensor,
    ):
        """Return mass weights, mean and covariance of a two-Gaussian mixture.

        ``mass_i`` and ``mass_j`` are proportional to the true integrals of
        optical density: the omitted constant ``(2*pi)^(3/2)`` cancels in the
        weights.  The between-mean outer products are required for second-
        moment conservation; merely averaging scale and rotation is invalid.
        """
        total_mass = mass_i + mass_j
        safe_total_mass = total_mass.clamp_min(self.merge_mass_eps)
        weight_i = mass_i / safe_total_mass
        weight_i = torch.where(
            total_mass > self.merge_mass_eps,
            weight_i,
            torch.full_like(weight_i, 0.5),
        )
        weight_j = 1.0 - weight_i
        merged_mean = (
            weight_i.unsqueeze(-1) * mean_i
            + weight_j.unsqueeze(-1) * mean_j
        )
        delta_i = mean_i - merged_mean
        delta_j = mean_j - merged_mean
        merged_covariance = (
            weight_i[..., None, None]
            * (covariance_i + delta_i.unsqueeze(-1) * delta_i.unsqueeze(-2))
            + weight_j[..., None, None]
            * (covariance_j + delta_j.unsqueeze(-1) * delta_j.unsqueeze(-2))
        )
        # Remove the tiny antisymmetric part introduced by finite precision.
        merged_covariance = 0.5 * (
            merged_covariance + merged_covariance.transpose(-1, -2)
        )
        return weight_i, weight_j, total_mass, merged_mean, merged_covariance

    @torch.no_grad()
    def _select_merge_pairs(
        self,
        gaussian,
        anchor: torch.Tensor,
        merge_candidate_mask: torch.Tensor,
    ):
        """Select disjoint conservative pairs with chunked KNN + mutual best.

        KNN is only a proposal mechanism.  A pair must additionally pass:

        1. Bhattacharyya similarity of the two full anisotropic Gaussians;
        2. cosine similarity of their semantic probabilities (18 classes,
           including the explicit empty class, in the new optical setup);
        3. exact encodability of the moment-matched mean, scale and opacity.

        Among valid neighbours each candidate chooses the largest product of
        geometry and semantic similarity.  Only mutual choices are retained,
        which prevents one-to-many conflicts without a global matching pass.
        """
        batch_size, num_gaussians = merge_candidate_mask.shape
        if batch_size != 1:
            raise NotImplementedError(
                "Dynamic Gaussian topology control requires batch_size=1; "
                f"got batch_size={batch_size}"
            )
        empty_pairs = torch.empty(
            (0, 2), dtype=torch.long, device=merge_candidate_mask.device
        )
        empty_similarity = torch.empty(
            (0,), dtype=torch.float32, device=merge_candidate_mask.device
        )

        candidate_indices = torch.nonzero(
            merge_candidate_mask[0], as_tuple=False
        ).squeeze(-1)
        num_candidates = int(candidate_indices.numel())
        if num_candidates < 2:
            return empty_pairs, empty_similarity, empty_similarity.clone()
        if anchor.shape[0] != 1 or anchor.shape[1] != num_gaussians:
            raise ValueError("anchor and merge_candidate_mask are misaligned")
        if anchor.shape[-1] <= 11:
            raise ValueError(
                "Merging requires semantic logits at anchor[..., 11:]"
            )

        # Pair selection is discrete, so all proposal statistics stay in
        # float32 and detached from both Score and occupancy gradients.
        means = gaussian.means[0, candidate_indices].detach().float()
        scales = gaussian.scales[0, candidate_indices].detach().float()
        rotations = gaussian.rotations[0, candidate_indices].detach().float()
        alpha = gaussian.opacities[0, candidate_indices, 0].detach().float()
        alpha = alpha.clamp(min=0.0, max=1.0 - self.opacity_eps)
        rho = -torch.log1p(-alpha)
        mass = rho * scales.prod(dim=-1)
        covariance = self._physical_covariance(scales, rotations)
        semantic_probability = F.softmax(
            anchor[0, candidate_indices, 11:].detach().float(), dim=-1
        )

        neighbour_count = min(self.merge_knn, num_candidates - 1)
        best_neighbour = torch.full(
            (num_candidates,), -1, dtype=torch.long, device=means.device
        )
        best_geometry = torch.zeros(
            num_candidates, dtype=torch.float32, device=means.device
        )
        best_semantic = torch.zeros_like(best_geometry)
        identity = torch.eye(3, dtype=torch.float32, device=means.device)

        scale_range = self.scale_range.to(device=means.device, dtype=torch.float32)
        scale_margin = (
            scale_range[1] - scale_range[0]
        ) * self.encoding_unit_eps
        scale_lower = scale_range[0] + scale_margin + self.scale_eps
        scale_upper = scale_range[1] - scale_margin - self.scale_eps
        pc_min = self.pc_range[:3].to(device=means.device, dtype=torch.float32)
        pc_max = self.pc_range[3:].to(device=means.device, dtype=torch.float32)
        position_margin = (pc_max - pc_min) * self.encoding_unit_eps
        position_lower = pc_min + position_margin + self.position_eps
        position_upper = pc_max - position_margin - self.position_eps

        for start in range(0, num_candidates, self.merge_chunk_size):
            end = min(start + self.merge_chunk_size, num_candidates)
            query_count = end - start
            distances = torch.cdist(means[start:end], means)
            query_rows = torch.arange(query_count, device=means.device)
            self_columns = torch.arange(start, end, device=means.device)
            distances[query_rows, self_columns] = torch.inf
            neighbour = torch.topk(
                distances,
                k=neighbour_count,
                dim=1,
                largest=False,
                sorted=False,
            ).indices

            mean_i = means[start:end, None, :].expand(
                -1, neighbour_count, -1
            )
            mean_j = means[neighbour]
            covariance_i = covariance[start:end, None, :, :].expand(
                -1, neighbour_count, -1, -1
            )
            covariance_j = covariance[neighbour]

            # Bhattacharyya distance accounts for center, scale and rotation.
            # Unlike response-at-the-other-center, it rejects concentric but
            # radically different Gaussians.
            covariance_middle = 0.5 * (covariance_i + covariance_j)
            covariance_middle = covariance_middle + (
                self.merge_eigenvalue_eps * identity
            )
            delta = mean_i - mean_j
            solved_delta = torch.linalg.solve(
                covariance_middle, delta.unsqueeze(-1)
            ).squeeze(-1)
            quadratic = (delta * solved_delta).sum(dim=-1)
            logdet_middle = torch.linalg.slogdet(covariance_middle).logabsdet
            logdet_i = 2.0 * torch.log(
                scales[start:end].clamp_min(self.merge_eigenvalue_eps)
            ).sum(dim=-1, keepdim=True)
            logdet_j = 2.0 * torch.log(
                scales[neighbour].clamp_min(self.merge_eigenvalue_eps)
            ).sum(dim=-1)
            bhattacharyya_distance = (
                0.125 * quadratic
                + 0.5
                * (logdet_middle - 0.5 * (logdet_i + logdet_j))
            ).clamp_min(0.0)
            geometry_similarity = torch.exp(-bhattacharyya_distance)
            semantic_similarity = F.cosine_similarity(
                semantic_probability[start:end, None, :],
                semantic_probability[neighbour],
                dim=-1,
                eps=self.merge_probability_eps,
            )

            mass_i = mass[start:end, None].expand(-1, neighbour_count)
            mass_j = mass[neighbour]
            (
                _,
                _,
                merged_mass,
                merged_mean,
                merged_covariance,
            ) = self._moment_match_geometry(
                mean_i,
                mean_j,
                covariance_i,
                covariance_j,
                mass_i,
                mass_j,
            )
            eigenvalues = torch.linalg.eigvalsh(merged_covariance)
            merged_scales = torch.sqrt(
                eigenvalues.clamp_min(self.merge_eigenvalue_eps)
            )
            merged_rho = merged_mass / merged_scales.prod(dim=-1).clamp_min(
                self.merge_mass_eps
            )
            merged_alpha = -torch.expm1(-merged_rho)

            scale_encodable = (
                (merged_scales >= scale_lower)
                & (merged_scales <= scale_upper)
            ).all(dim=-1)
            mean_encodable = (
                (merged_mean >= position_lower)
                & (merged_mean <= position_upper)
            ).all(dim=-1)
            opacity_encodable = (
                (merged_alpha >= self.encoding_unit_eps)
                & (merged_alpha <= 1.0 - self.encoding_unit_eps)
            )
            finite = (
                torch.isfinite(geometry_similarity)
                & torch.isfinite(semantic_similarity)
                & torch.isfinite(merged_scales).all(dim=-1)
                & torch.isfinite(merged_alpha)
            )
            valid = (
                (geometry_similarity >= self.merge_geometry_threshold)
                & (semantic_similarity >= self.merge_semantic_threshold)
                & scale_encodable
                & mean_encodable
                & opacity_encodable
                & finite
            )
            pair_quality = geometry_similarity * semantic_similarity
            pair_quality = pair_quality.masked_fill(~valid, -torch.inf)
            best_quality, best_column = pair_quality.max(dim=1)
            row_has_match = torch.isfinite(best_quality)
            selected_neighbour = neighbour.gather(
                1, best_column.unsqueeze(-1)
            ).squeeze(-1)
            best_neighbour[start:end] = torch.where(
                row_has_match,
                selected_neighbour,
                torch.full_like(selected_neighbour, -1),
            )
            best_geometry[start:end] = geometry_similarity.gather(
                1, best_column.unsqueeze(-1)
            ).squeeze(-1)
            best_semantic[start:end] = semantic_similarity.gather(
                1, best_column.unsqueeze(-1)
            ).squeeze(-1)

        local_index = torch.arange(num_candidates, device=means.device)
        has_neighbour = best_neighbour >= 0
        safe_neighbour = best_neighbour.clamp_min(0)
        mutual = (
            has_neighbour
            & (best_neighbour[safe_neighbour] == local_index)
            & (local_index < best_neighbour)
        )
        left_local = local_index[mutual]
        right_local = best_neighbour[left_local]
        if left_local.numel() == 0:
            return empty_pairs, empty_similarity, empty_similarity.clone()
        merge_pairs = torch.stack(
            [candidate_indices[left_local], candidate_indices[right_local]],
            dim=-1,
        )
        return (
            merge_pairs,
            best_geometry[left_local],
            best_semantic[left_local],
        )

    @staticmethod
    def _matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to normalized ``[w, x, y, z]`` values."""
        if rotation_matrix.shape[-2:] != (3, 3):
            raise ValueError("rotation_matrix must end in [3, 3]")
        m00 = rotation_matrix[..., 0, 0]
        m11 = rotation_matrix[..., 1, 1]
        m22 = rotation_matrix[..., 2, 2]
        q_abs = torch.sqrt(
            torch.clamp(
                torch.stack(
                    [
                        1.0 + m00 + m11 + m22,
                        1.0 + m00 - m11 - m22,
                        1.0 - m00 + m11 - m22,
                        1.0 - m00 - m11 + m22,
                    ],
                    dim=-1,
                ),
                min=0.0,
            )
        )
        candidates = torch.stack(
            [
                torch.stack(
                    [
                        q_abs[..., 0].square(),
                        rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
                        rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
                        rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
                        q_abs[..., 1].square(),
                        rotation_matrix[..., 1, 0] + rotation_matrix[..., 0, 1],
                        rotation_matrix[..., 0, 2] + rotation_matrix[..., 2, 0],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
                        rotation_matrix[..., 1, 0] + rotation_matrix[..., 0, 1],
                        q_abs[..., 2].square(),
                        rotation_matrix[..., 2, 1] + rotation_matrix[..., 1, 2],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
                        rotation_matrix[..., 0, 2] + rotation_matrix[..., 2, 0],
                        rotation_matrix[..., 2, 1] + rotation_matrix[..., 1, 2],
                        q_abs[..., 3].square(),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        candidates = candidates / (
            2.0 * q_abs[..., :, None].clamp_min(1e-8)
        )
        best = q_abs.argmax(dim=-1)
        gather_index = best[..., None, None].expand(*best.shape, 1, 4)
        quaternion = candidates.gather(-2, gather_index).squeeze(-2)
        return F.normalize(quaternion, dim=-1)

    def _build_merged_gaussians(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian,
        merge_pairs: torch.Tensor,
    ):
        """Moment-match selected pairs and encode the merged anchors."""
        index_i = merge_pairs[:, 0]
        index_j = merge_pairs[:, 1]
        mean_i = gaussian.means[0, index_i].float()
        mean_j = gaussian.means[0, index_j].float()
        scale_i = gaussian.scales[0, index_i].float()
        scale_j = gaussian.scales[0, index_j].float()
        covariance_i = self._physical_covariance(
            scale_i, gaussian.rotations[0, index_i].float()
        )
        covariance_j = self._physical_covariance(
            scale_j, gaussian.rotations[0, index_j].float()
        )
        alpha_i = gaussian.opacities[0, index_i, 0].float().clamp(
            min=0.0, max=1.0 - self.opacity_eps
        )
        alpha_j = gaussian.opacities[0, index_j, 0].float().clamp(
            min=0.0, max=1.0 - self.opacity_eps
        )
        mass_i = -torch.log1p(-alpha_i) * scale_i.prod(dim=-1)
        mass_j = -torch.log1p(-alpha_j) * scale_j.prod(dim=-1)
        (
            weight_i,
            weight_j,
            merged_mass,
            merged_mean,
            merged_covariance,
        ) = self._moment_match_geometry(
            mean_i,
            mean_j,
            covariance_i,
            covariance_j,
            mass_i,
            mass_j,
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(merged_covariance)
        merged_scales = torch.sqrt(
            eigenvalues.clamp_min(self.merge_eigenvalue_eps)
        )
        # eigh returns Sigma = V diag(lambda) V^T, whereas the rest of this
        # project uses Sigma = R^T diag(scale^2) R.  Therefore R must be V^T.
        # Eigenvector derivatives contain inverse eigenvalue gaps and become
        # unstable for isotropic/nearly isotropic covariances.  The rotation
        # remains exactly moment matched in the forward pass, but is treated
        # as a discrete topology result.  Mean, eigenvalue-derived scales,
        # opacity, semantics and instance features remain differentiable.
        merged_rotation_matrix = eigenvectors.detach().transpose(-1, -2)
        determinant = torch.linalg.det(merged_rotation_matrix)
        row_sign = torch.ones_like(merged_scales)
        row_sign[..., -1] = torch.where(
            determinant < 0.0,
            -torch.ones_like(determinant),
            torch.ones_like(determinant),
        )
        merged_rotation_matrix = (
            row_sign.unsqueeze(-1) * merged_rotation_matrix
        )
        merged_rotation = self._matrix_to_quaternion(merged_rotation_matrix)

        merged_rho = merged_mass / merged_scales.prod(dim=-1).clamp_min(
            self.merge_mass_eps
        )
        merged_alpha = (-torch.expm1(-merged_rho)).clamp(
            min=self.encoding_unit_eps,
            max=1.0 - self.encoding_unit_eps,
        )

        semantic_i = F.softmax(anchor[0, index_i, 11:].float(), dim=-1)
        semantic_j = F.softmax(anchor[0, index_j, 11:].float(), dim=-1)
        merged_probability = (
            weight_i.unsqueeze(-1) * semantic_i
            + weight_j.unsqueeze(-1) * semantic_j
        )
        merged_probability = merged_probability / merged_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.merge_probability_eps)
        merged_semantic_logits = torch.log(
            merged_probability.clamp_min(self.merge_probability_eps)
        )

        merged_anchor = anchor[:, index_i].clone()
        merged_anchor[..., :3] = self._encode_xyz(
            merged_mean.unsqueeze(0).to(dtype=anchor.dtype)
        )
        merged_anchor[..., 3:6] = self._encode_scale(
            merged_scales.unsqueeze(0).to(dtype=anchor.dtype)
        )
        merged_anchor[..., 6:10] = merged_rotation.unsqueeze(0).to(
            dtype=anchor.dtype
        )
        merged_anchor[..., 10:11] = safe_inverse_sigmoid(
            merged_alpha.unsqueeze(0).unsqueeze(-1).to(dtype=anchor.dtype)
        )
        merged_anchor[..., 11:] = merged_semantic_logits.unsqueeze(0).to(
            dtype=anchor.dtype
        )

        feature_i = instance_feature[:, index_i]
        feature_j = instance_feature[:, index_j]
        merged_feature = (
            weight_i[None, :, None].to(dtype=feature_i.dtype) * feature_i
            + weight_j[None, :, None].to(dtype=feature_j.dtype) * feature_j
        )
        return merged_feature, merged_anchor

    def _apply_topology_edits(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        last_operation: torch.Tensor,
        operation_age: torch.Tensor,
        prune_evidence: torch.Tensor,
        gaussian,
        split_mask: torch.Tensor,
        delete_mask: torch.Tensor,
        child_mean_positive: torch.Tensor,
        child_mean_negative: torch.Tensor,
        child_scales: torch.Tensor,
        merge_pairs: torch.Tensor,
        generated_child_anchor: Optional[torch.Tensor] = None,
        generated_child_feature: Optional[torch.Tensor] = None,
    ):
        """Apply edits atomically and keep provenance aligned with topology."""
        batch_size, num_gaussians = split_mask.shape
        if delete_mask.shape != split_mask.shape:
            raise ValueError("split_mask and delete_mask must have the same shape")
        if batch_size != 1:
            raise NotImplementedError(
                "Dynamic Gaussian topology control requires batch_size=1; "
                f"got batch_size={batch_size}"
            )
        if last_operation.shape != split_mask.shape:
            raise ValueError(
                "last_operation must have shape [B, N], got "
                f"{last_operation.shape} for {split_mask.shape}"
            )
        if last_operation.dtype != torch.long:
            raise ValueError("last_operation must use torch.long dtype")
        if operation_age.shape != split_mask.shape:
            raise ValueError(
                "operation_age must have shape [B, N], got "
                f"{operation_age.shape} for {split_mask.shape}"
            )
        if operation_age.dtype != torch.long:
            raise ValueError("operation_age must use torch.long dtype")
        if torch.any(operation_age < 0):
            raise ValueError("operation_age must be non-negative")
        if prune_evidence.shape != split_mask.shape:
            raise ValueError(
                "prune_evidence must have shape [B, N], got "
                f"{prune_evidence.shape} for {split_mask.shape}"
            )
        if prune_evidence.dtype != torch.long:
            raise ValueError("prune_evidence must use torch.long dtype")
        if torch.any(prune_evidence < 0):
            raise ValueError("prune_evidence must be non-negative")

        selected_split = split_mask[0]
        selected_delete = delete_mask[0]
        if torch.any(selected_split & selected_delete):
            raise RuntimeError("A Gaussian cannot be split and deleted together")

        if merge_pairs.ndim != 2 or merge_pairs.shape[-1] != 2:
            raise ValueError("merge_pairs must have shape [P, 2]")
        if merge_pairs.dtype != torch.long:
            raise ValueError("merge_pairs must contain long indices")
        if merge_pairs.numel() > 0:
            if torch.any(merge_pairs < 0) or torch.any(
                merge_pairs >= num_gaussians
            ):
                raise ValueError("merge_pairs contains an out-of-range index")
            flattened_pairs = merge_pairs.flatten()
            if torch.unique(flattened_pairs).numel() != flattened_pairs.numel():
                raise RuntimeError("Merge pairs must be disjoint")
        merge_member = torch.zeros_like(selected_split)
        if merge_pairs.numel() > 0:
            merge_member[merge_pairs.flatten()] = True
        if torch.any(merge_member & (selected_split | selected_delete)):
            raise RuntimeError(
                "A Gaussian cannot be merged and split/deleted together"
            )

        num_split = int(selected_split.sum().item())
        num_deleted = int(selected_delete.sum().item())
        num_merge_pairs = int(merge_pairs.shape[0])
        if num_split == 0 and num_deleted == 0 and num_merge_pairs == 0:
            next_age = torch.where(
                last_operation == self.OP_KEEP,
                torch.zeros_like(operation_age),
                operation_age + 1,
            )
            return (
                instance_feature,
                anchor,
                last_operation,
                next_age,
                prune_evidence,
            )

        # A split parent contributes two new children; both parents of a merge
        # contribute one moment-matched Gaussian.  Building all outputs from
        # the same pre-edit representation avoids order-dependent decisions.
        keep = ~(selected_split | selected_delete | merge_member)
        kept_anchor = anchor[:, keep]
        kept_feature = instance_feature[:, keep]
        output_anchors = [kept_anchor]
        output_features = [kept_feature]
        kept_operation = last_operation[:, keep]
        kept_age = operation_age[:, keep]
        kept_age = torch.where(
            kept_operation == self.OP_KEEP,
            torch.zeros_like(kept_age),
            kept_age + 1,
        )
        output_operations = [kept_operation]
        output_ages = [kept_age]
        output_prune_evidence = [prune_evidence[:, keep]]

        if num_split > 0:
            if generated_child_anchor is not None:
                if generated_child_feature is None:
                    raise ValueError(
                        "generated_child_feature is required with learned "
                        "child anchors"
                    )
                expected_anchor_shape = (
                    batch_size,
                    num_split,
                    2,
                    anchor.shape[-1],
                )
                expected_feature_shape = (
                    batch_size,
                    num_split,
                    2,
                    instance_feature.shape[-1],
                )
                if generated_child_anchor.shape != expected_anchor_shape:
                    raise ValueError(
                        "generated_child_anchor must have shape "
                        f"{expected_anchor_shape}, got "
                        f"{generated_child_anchor.shape}"
                    )
                if generated_child_feature.shape != expected_feature_shape:
                    raise ValueError(
                        "generated_child_feature must have shape "
                        f"{expected_feature_shape}, got "
                        f"{generated_child_feature.shape}"
                    )
                output_anchors.append(
                    generated_child_anchor.flatten(1, 2)
                )
                output_features.append(
                    generated_child_feature.flatten(1, 2)
                )
            else:
                parent_anchor = anchor[:, selected_split]
                parent_feature = instance_feature[:, selected_split]

                # Interleave the positive/negative child of every parent.
                child_anchor = parent_anchor.unsqueeze(2).expand(
                    -1, -1, 2, -1
                ).clone()
                selected_means = torch.stack(
                    [child_mean_positive, child_mean_negative], dim=2
                )
                selected_scales = child_scales.unsqueeze(2).expand(
                    -1, -1, 2, -1
                )
                if selected_means.shape != (
                    batch_size,
                    num_split,
                    2,
                    3,
                ):
                    raise ValueError(
                        "Analytical child means must contain selected "
                        f"parents only, got {selected_means.shape}"
                    )
                if selected_scales.shape != selected_means.shape:
                    raise ValueError(
                        "Analytical child scales must match child means"
                    )
                child_anchor[..., :3] = self._encode_xyz(selected_means)
                child_anchor[..., 3:6] = self._encode_scale(selected_scales)

                parent_alpha = gaussian.opacities[:, selected_split]
                if parent_alpha.ndim != 3 or parent_alpha.shape[-1] != 1:
                    raise ValueError(
                        "gaussian.opacities must be [B, N, 1] for splitting, "
                        f"got {parent_alpha.shape}"
                    )
                parent_alpha = parent_alpha.float().clamp(
                    min=0.0, max=1.0 - self.opacity_eps
                )
                parent_rho = -torch.log1p(-parent_alpha)
                # Legacy path: preserve integrated optical-density mass.
                child_rho = parent_rho / (2.0 * self.split_scale_ratio)
                child_alpha = -torch.expm1(-child_rho)
                child_alpha = child_alpha.clamp(
                    min=self.opacity_eps, max=1.0 - self.opacity_eps
                ).to(dtype=anchor.dtype)
                child_opacity_anchor = safe_inverse_sigmoid(child_alpha)
                child_anchor[..., 10:11] = child_opacity_anchor.unsqueeze(2)

                output_anchors.append(child_anchor.flatten(1, 2))
                output_features.append(
                    parent_feature.unsqueeze(2).expand(
                        -1, -1, 2, -1
                    ).reshape(1, 2 * num_split, -1)
                )
            output_operations.append(
                torch.full(
                    (batch_size, 2 * num_split),
                    self.OP_SPLIT_CHILD,
                    dtype=torch.long,
                    device=anchor.device,
                )
            )
            output_ages.append(
                torch.zeros(
                    (batch_size, 2 * num_split),
                    dtype=torch.long,
                    device=anchor.device,
                )
            )
            output_prune_evidence.append(
                torch.zeros(
                    (batch_size, 2 * num_split),
                    dtype=torch.long,
                    device=anchor.device,
                )
            )

        if num_merge_pairs > 0:
            merged_feature, merged_anchor = self._build_merged_gaussians(
                instance_feature=instance_feature,
                anchor=anchor,
                gaussian=gaussian,
                merge_pairs=merge_pairs,
            )
            output_anchors.append(merged_anchor)
            output_features.append(merged_feature)
            output_operations.append(
                torch.full(
                    (batch_size, num_merge_pairs),
                    self.OP_MERGED,
                    dtype=torch.long,
                    device=anchor.device,
                )
            )
            output_ages.append(
                torch.zeros(
                    (batch_size, num_merge_pairs),
                    dtype=torch.long,
                    device=anchor.device,
                )
            )
            output_prune_evidence.append(
                torch.zeros(
                    (batch_size, num_merge_pairs),
                    dtype=torch.long,
                    device=anchor.device,
                )
            )

        output_anchor = torch.cat(output_anchors, dim=1)
        output_feature = torch.cat(output_features, dim=1)
        output_last_operation = torch.cat(output_operations, dim=1)
        output_operation_age = torch.cat(output_ages, dim=1)
        output_prune_evidence = torch.cat(
            output_prune_evidence, dim=1
        )
        expected_count = (
            num_gaussians + num_split - num_deleted - num_merge_pairs
        )
        if output_anchor.shape[1] != expected_count:
            raise RuntimeError(
                "Topology count mismatch: expected "
                f"{expected_count}, got {output_anchor.shape[1]}"
            )
        if output_last_operation.shape != output_anchor.shape[:2]:
            raise RuntimeError(
                "Topology provenance became misaligned with output anchors"
            )
        if output_operation_age.shape != output_anchor.shape[:2]:
            raise RuntimeError(
                "Topology age became misaligned with output anchors"
            )
        if output_prune_evidence.shape != output_anchor.shape[:2]:
            raise RuntimeError(
                "Prune evidence became misaligned with output anchors"
            )
        return (
            output_feature,
            output_anchor,
            output_last_operation,
            output_operation_age,
            output_prune_evidence,
        )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        gaussian=None,
        stage_index: int = 0,
        last_operation: Optional[torch.Tensor] = None,
        operation_age: Optional[torch.Tensor] = None,
        prune_evidence: Optional[torch.Tensor] = None,
        global_iter=None,
        disable_split_at_eval: bool = False,
        collect_split_diagnostics: bool = False,
    ):
        if disable_split_at_eval and self.training:
            raise RuntimeError(
                "disable_split_at_eval is an evaluation-only diagnostic "
                "switch and must not be used while the model is training"
            )
        stage_index = int(stage_index)
        if stage_index < 0:
            raise ValueError("stage_index must be non-negative")
        split_enabled = self._operation_enabled(
            self.enable_split, self.split_stages, stage_index
        ) and not disable_split_at_eval
        delete_enabled = self._operation_enabled(
            self.enable_delete, self.delete_stages, stage_index
        )
        merge_enabled = self._operation_enabled(
            self.enable_merge, self.merge_stages, stage_index
        )

        if anchor.shape[:2] != instance_feature.shape[:2]:
            raise ValueError(
                "anchor and instance_feature must share [B, N], got "
                f"{anchor.shape[:2]} and {instance_feature.shape[:2]}"
            )
        if last_operation is None:
            last_operation = torch.full(
                anchor.shape[:2],
                self.OP_KEEP,
                dtype=torch.long,
                device=anchor.device,
            )
        if operation_age is None:
            operation_age = torch.zeros(
                anchor.shape[:2], dtype=torch.long, device=anchor.device
            )
        if prune_evidence is None:
            prune_evidence = torch.zeros(
                anchor.shape[:2], dtype=torch.long, device=anchor.device
            )
        if last_operation.shape != anchor.shape[:2]:
            raise ValueError(
                "last_operation must share anchor [B, N], got "
                f"{last_operation.shape} and {anchor.shape[:2]}"
            )
        if last_operation.dtype != torch.long:
            raise ValueError("last_operation must use torch.long dtype")
        if last_operation.device != anchor.device:
            raise ValueError("last_operation and anchor must share a device")
        if operation_age.shape != anchor.shape[:2]:
            raise ValueError(
                "operation_age must share anchor [B, N], got "
                f"{operation_age.shape} and {anchor.shape[:2]}"
            )
        if operation_age.dtype != torch.long:
            raise ValueError("operation_age must use torch.long dtype")
        if operation_age.device != anchor.device:
            raise ValueError("operation_age and anchor must share a device")
        if torch.any(operation_age < 0):
            raise ValueError("operation_age must be non-negative")
        if prune_evidence.shape != anchor.shape[:2]:
            raise ValueError(
                "prune_evidence must share anchor [B, N], got "
                f"{prune_evidence.shape} and {anchor.shape[:2]}"
            )
        if prune_evidence.dtype != torch.long:
            raise ValueError("prune_evidence must use torch.long dtype")
        if prune_evidence.device != anchor.device:
            raise ValueError("prune_evidence and anchor must share a device")
        if torch.any(prune_evidence < 0):
            raise ValueError("prune_evidence must be non-negative")
        valid_operation = (
            (last_operation == self.OP_KEEP)
            | (last_operation == self.OP_SPLIT_CHILD)
            | (last_operation == self.OP_MERGED)
        )
        if not bool(torch.all(valid_operation).item()):
            raise ValueError("last_operation contains an unknown operation id")

        split_provenance_mask = last_operation == self.OP_SPLIT_CHILD
        merge_provenance_mask = last_operation == self.OP_MERGED

        score_logits = self.score_head(instance_feature, anchor_embed)
        scores = score_logits.sigmoid()
        tau_1, tau_2, score_mean, score_std = self.threshold_generator(scores)

        candidates_enabled = self._candidates_enabled(global_iter)
        if candidates_enabled:
            low_mask = scores.detach() <= tau_1
            high_mask = scores.detach() >= tau_2
            low_mask = self._cap_candidates(
                low_mask,
                scores,
                max_ratio=self.max_low_ratio,
                largest=False,
            )
            high_mask = self._cap_candidates(
                high_mask,
                scores,
                max_ratio=self.max_high_ratio,
                largest=True,
            )
        else:
            low_mask = torch.zeros_like(scores, dtype=torch.bool)
            high_mask = torch.zeros_like(scores, dtype=torch.bool)

        if (split_enabled or delete_enabled or merge_enabled) and gaussian is None:
            raise ValueError(
                "The refined physical gaussian is required for topology control"
            )

        batch_size, num_gaussians = scores.shape
        if gaussian is not None:
            opacity = gaussian.opacities
            scales = gaussian.scales
            if opacity.shape != (batch_size, num_gaussians, 1):
                raise ValueError(
                    "gaussian.opacities must be [B, N, 1], got "
                    f"{opacity.shape}"
                )
            if scales.shape != (batch_size, num_gaussians, 3):
                raise ValueError(
                    "gaussian.scales must be [B, N, 3], got "
                    f"{scales.shape}"
                )
            # Deletion is a non-differentiable topology decision.  Keep its
            # contribution statistics detached from Score and occupancy loss.
            opacity_for_control = opacity.detach().float().squeeze(-1).clamp(
                min=0.0, max=1.0 - self.opacity_eps
            )
            optical_rho = -torch.log1p(-opacity_for_control)
            density_mass = optical_rho * scales.detach().float().prod(dim=-1)
        else:
            opacity_for_control = torch.zeros_like(scores, dtype=torch.float32)
            density_mass = torch.zeros_like(scores, dtype=torch.float32)

        low_opacity_mask = (
            opacity_for_control <= self.opacity_delete_threshold
        )
        low_density_mass_mask = (
            density_mass <= self.density_mass_delete_threshold
        )
        # Candidate classification and topology execution are separate.  A
        # merge-only experiment still needs to recognize negligible
        # Gaussians and exclude them from merging, although it deliberately
        # keeps them in the representation for an attributable experiment.
        delete_candidate_mask = torch.zeros_like(low_mask)
        if candidates_enabled:
            delete_candidate_mask = (
                low_mask & low_opacity_mask & low_density_mass_mask
            )
        # Evidence must be consecutive: any stage which does not classify the
        # Gaussian as a delete candidate resets its counter to zero.
        next_prune_evidence_candidate = torch.where(
            delete_candidate_mask,
            (prune_evidence + 1).clamp_max(
                self.prune_min_consecutive_stages
            ),
            torch.zeros_like(prune_evidence),
        )
        topology_lineage_mask = last_operation != self.OP_KEEP
        delete_protected_mask = torch.zeros_like(delete_candidate_mask)
        if self.require_consecutive_prune_evidence:
            delete_protected_mask = (
                delete_candidate_mask
                & topology_lineage_mask
                & (
                    next_prune_evidence_candidate
                    < self.prune_min_consecutive_stages
                )
            )
        delete_mask = torch.zeros_like(low_mask)
        if delete_enabled:
            delete_mask = delete_candidate_mask & ~delete_protected_mask

        scale_feasible_mask = torch.ones_like(high_mask)
        children_inside_mask = torch.ones_like(high_mask)
        split_mask = torch.zeros_like(high_mask)
        split_reverse_block_mask = high_mask & merge_provenance_mask
        split_candidate_mask = high_mask & ~merge_provenance_mask
        child_mean_positive = None
        child_mean_negative = None
        child_scales = None
        generated_child_anchor = None
        generated_child_feature = None
        learned_split_mass_ratio = torch.ones_like(
            scores, dtype=torch.float32
        )
        learned_split_position_clamped = torch.zeros_like(
            high_mask, dtype=torch.bool
        )
        sibling_diagnostics = {}
        # Skip all physical split geometry during Score warmup and when the
        # learned threshold yields no high-complexity candidate.
        should_prepare_split = (
            split_enabled
            and candidates_enabled
            and bool(torch.any(split_candidate_mask).item())
        )
        if should_prepare_split:
            if batch_size != 1:
                raise NotImplementedError(
                    "Dynamic Gaussian splitting requires batch_size=1; "
                    f"got batch_size={batch_size}"
                )
            if self.split_generator is not None:
                # Determine the final parent mask before running SplitMLP.
                # Learned positions are clipped to pc_range, so the only
                # cheap precondition is that at least one parent scale has a
                # non-degenerate span above scale_min.
                scale_min = self.scale_range[0].to(
                    device=gaussian.scales.device,
                    dtype=gaussian.scales.dtype,
                )
                scale_feasible_mask = (
                    gaussian.scales > scale_min + self.scale_eps
                ).any(dim=-1)
                split_mask = split_candidate_mask & scale_feasible_mask
                selected_split = split_mask[0]
                if bool(torch.any(selected_split).item()):
                    selected_gaussian = self._select_gaussian(
                        gaussian, selected_split
                    )
                    generated_split = self.split_generator(
                        instance_feature=instance_feature[:, selected_split],
                        anchor=anchor[:, selected_split],
                        anchor_embed=anchor_embed[:, selected_split],
                        gaussian=selected_gaussian,
                        collect_diagnostics=collect_split_diagnostics,
                    )
                    generated_child_anchor = generated_split["child_anchor"]
                    generated_child_feature = generated_split["child_feature"]
                    if not torch.all(
                        generated_split["scale_feasible_mask"]
                    ):
                        raise RuntimeError(
                            "Preselected learned split contains an "
                            "infeasible parent scale"
                        )
                    if not torch.all(
                        generated_split[
                            "children_inside_pc_range_mask"
                        ]
                    ):
                        raise RuntimeError(
                            "Learned split generator returned an "
                            "out-of-range child"
                        )
                    learned_split_mass_ratio[:, selected_split] = (
                        generated_split["density_mass_ratio"]
                        .detach()
                        .float()
                    )
                    learned_split_position_clamped[:, selected_split] = (
                        generated_split["position_was_clamped_mask"].detach()
                    )
                    for name, values in generated_split.get(
                        "sibling_diagnostics", {}
                    ).items():
                        full_values = torch.zeros(
                            scores.shape,
                            dtype=values.dtype,
                            device=values.device,
                        )
                        full_values[:, selected_split] = values.detach()
                        sibling_diagnostics[name] = full_values
            else:
                # The legacy analytical path is also evaluated only for the
                # high-score candidates.  A second local filter removes the
                # rare candidates whose children cannot be encoded legally.
                selected_high = split_candidate_mask[0]
                selected_gaussian = self._select_gaussian(
                    gaussian, selected_high
                )
                (
                    candidate_scale_feasible,
                    candidate_children_inside,
                    candidate_mean_positive,
                    candidate_mean_negative,
                    candidate_scales,
                ) = self._split_feasibility(selected_gaussian)
                scale_feasible_mask[:, selected_high] = (
                    candidate_scale_feasible
                )
                children_inside_mask[:, selected_high] = (
                    candidate_children_inside
                )
                candidate_valid = (
                    candidate_scale_feasible & candidate_children_inside
                )[0]
                high_indices = torch.nonzero(
                    selected_high, as_tuple=False
                ).squeeze(-1)
                split_mask[0, high_indices[candidate_valid]] = True
                child_mean_positive = candidate_mean_positive[
                    :, candidate_valid
                ]
                child_mean_negative = candidate_mean_negative[
                    :, candidate_valid
                ]
                child_scales = candidate_scales[:, candidate_valid]

        if torch.any(split_mask & delete_mask):
            raise RuntimeError("Split and delete masks must be disjoint")

        merge_candidate_mask = torch.zeros_like(low_mask)
        merge_member_mask = torch.zeros_like(low_mask)
        merge_partner_index = torch.full_like(
            low_mask, -1, dtype=torch.long
        )
        merge_pairs = torch.empty(
            (0, 2), dtype=torch.long, device=scores.device
        )
        merge_geometry_similarity = torch.empty(
            (0,), dtype=torch.float32, device=scores.device
        )
        merge_semantic_similarity = torch.empty_like(
            merge_geometry_similarity
        )
        merge_reverse_block_mask = low_mask & split_provenance_mask
        should_prepare_merge = (
            merge_enabled
            and candidates_enabled
            and bool(
                torch.any(
                    low_mask
                    & ~delete_candidate_mask
                    & ~split_provenance_mask
                ).item()
            )
        )
        if should_prepare_merge:
            # Delete-candidate classification always has priority, even in a
            # merge-only ablation where deletion itself is disabled.  Split
            # candidates are high-score and already disjoint from this set.
            merge_candidate_mask = (
                low_mask
                & ~delete_candidate_mask
                & ~split_provenance_mask
            )
            (
                merge_pairs,
                merge_geometry_similarity,
                merge_semantic_similarity,
            ) = self._select_merge_pairs(
                gaussian=gaussian,
                anchor=anchor,
                merge_candidate_mask=merge_candidate_mask,
            )
            if merge_pairs.numel() > 0:
                left = merge_pairs[:, 0]
                right = merge_pairs[:, 1]
                merge_member_mask[0, merge_pairs.flatten()] = True
                merge_partner_index[0, left] = right
                merge_partner_index[0, right] = left

        if torch.any(
            merge_member_mask
            & (split_mask | delete_mask | delete_candidate_mask)
        ):
            raise RuntimeError(
                "Merge pairs overlap split/delete candidates or decisions"
            )
        # Always advance age, even if this stage made no physical edit.  The
        # structural provenance itself persists to block later reversal.
        (
            instance_feature,
            anchor,
            next_last_operation,
            next_operation_age,
            next_prune_evidence,
        ) = self._apply_topology_edits(
            instance_feature=instance_feature,
            anchor=anchor,
            last_operation=last_operation,
            operation_age=operation_age,
            prune_evidence=next_prune_evidence_candidate,
            gaussian=gaussian,
            split_mask=split_mask,
            delete_mask=delete_mask,
            child_mean_positive=child_mean_positive,
            child_mean_negative=child_mean_negative,
            child_scales=child_scales,
            merge_pairs=merge_pairs,
            generated_child_anchor=generated_child_anchor,
            generated_child_feature=generated_child_feature,
        )

        if self.split_generator is not None:
            # Split execution is data dependent: it is skipped during Score
            # warmup and may also be skipped when a rank has no high-score
            # candidate.  With DDP(find_unused_parameters=False), completely
            # bypassing SplitMLP on only some iterations/ranks leaves its
            # reduction bucket unfinished and fails at the next forward.
            #
            # Attach an exactly-zero dependency on every SplitMLP parameter
            # to the returned feature.  This does not run the MLP, alter any
            # prediction, or create a fake optimization signal; it only makes
            # autograd produce an explicit zero gradient for inactive split
            # iterations.  Real split gradients are accumulated normally
            # whenever generated children participate in the occupancy loss.
            split_ddp_zero = None
            for parameter in self.split_generator.parameters():
                if not parameter.requires_grad:
                    continue
                parameter_zero = parameter.reshape(-1)[0] * 0.0
                split_ddp_zero = (
                    parameter_zero
                    if split_ddp_zero is None
                    else split_ddp_zero + parameter_zero
                )
            if split_ddp_zero is not None:
                instance_feature = instance_feature + split_ddp_zero.to(
                    dtype=instance_feature.dtype
                )
                anchor = anchor + split_ddp_zero.to(dtype=anchor.dtype)

        num_split = split_mask.sum(dim=1)
        num_deleted = delete_mask.sum(dim=1)
        num_merge_pairs = torch.full(
            (batch_size,),
            int(merge_pairs.shape[0]),
            dtype=torch.long,
            device=scores.device,
        )
        num_before = torch.full(
            (batch_size,),
            num_gaussians,
            dtype=torch.long,
            device=scores.device,
        )
        control_result = {
            "score_logits": score_logits,
            "score": scores,
            "tau_1": tau_1,
            "tau_2": tau_2,
            "score_mean": score_mean,
            "score_std": score_std,
            "low_candidate_mask": low_mask,
            "high_candidate_mask": high_mask,
            "last_operation_before": last_operation,
            "operation_age_before": operation_age,
            "prune_evidence_before": prune_evidence,
            "prune_evidence_after_candidate": (
                next_prune_evidence_candidate
            ),
            "split_provenance_mask": split_provenance_mask,
            "merge_provenance_mask": merge_provenance_mask,
            "split_reverse_block_mask": split_reverse_block_mask,
            "scale_feasible_mask": scale_feasible_mask,
            "children_inside_pc_range_mask": children_inside_mask,
            "split_mask": split_mask,
            "learned_split_mass_ratio": learned_split_mass_ratio,
            "learned_split_position_clamped_mask": (
                learned_split_position_clamped
            ),
            "gaussian_opacity": opacity_for_control,
            "density_mass": density_mass,
            "low_opacity_mask": low_opacity_mask,
            "low_density_mass_mask": low_density_mass_mask,
            "delete_candidate_mask": delete_candidate_mask,
            "delete_protected_mask": delete_protected_mask,
            "delete_mask": delete_mask,
            "merge_candidate_mask": merge_candidate_mask,
            "merge_reverse_block_mask": merge_reverse_block_mask,
            "merge_member_mask": merge_member_mask,
            "merge_partner_index": merge_partner_index,
            "merge_pairs": merge_pairs,
            "merge_geometry_similarity": merge_geometry_similarity,
            "merge_semantic_similarity": merge_semantic_similarity,
            "opacity_delete_threshold": torch.full(
                (batch_size, 1),
                self.opacity_delete_threshold,
                dtype=torch.float32,
                device=scores.device,
            ),
            "density_mass_delete_threshold": torch.full(
                (batch_size, 1),
                self.density_mass_delete_threshold,
                dtype=torch.float32,
                device=scores.device,
            ),
            "require_consecutive_prune_evidence": torch.full(
                (batch_size,),
                self.require_consecutive_prune_evidence,
                dtype=torch.bool,
                device=scores.device,
            ),
            "prune_min_consecutive_stages": torch.full(
                (batch_size, 1),
                self.prune_min_consecutive_stages,
                dtype=torch.long,
                device=scores.device,
            ),
            "merge_geometry_threshold": torch.full(
                (batch_size, 1),
                self.merge_geometry_threshold,
                dtype=torch.float32,
                device=scores.device,
            ),
            "merge_semantic_threshold": torch.full(
                (batch_size, 1),
                self.merge_semantic_threshold,
                dtype=torch.float32,
                device=scores.device,
            ),
            "num_gaussians_before": num_before,
            "num_split": num_split,
            "num_deleted": num_deleted,
            "num_merge_pairs": num_merge_pairs,
            "num_split_reverse_blocked": split_reverse_block_mask.sum(dim=1),
            "num_delete_protected": delete_protected_mask.sum(dim=1),
            "num_merge_reverse_blocked": merge_reverse_block_mask.sum(dim=1),
            "num_gaussians_after": (
                num_before + num_split - num_deleted - num_merge_pairs
            ),
            "control_enabled": torch.full(
                (batch_size,),
                candidates_enabled,
                dtype=torch.bool,
                device=scores.device,
            ),
            "split_enabled": torch.full(
                (batch_size,),
                split_enabled and candidates_enabled,
                dtype=torch.bool,
                device=scores.device,
            ),
            "learned_split_enabled": torch.full(
                (batch_size,),
                (
                    split_enabled
                    and candidates_enabled
                    and self.split_generator is not None
                ),
                dtype=torch.bool,
                device=scores.device,
            ),
            "delete_enabled": torch.full(
                (batch_size,),
                delete_enabled and candidates_enabled,
                dtype=torch.bool,
                device=scores.device,
            ),
            "merge_enabled": torch.full(
                (batch_size,),
                merge_enabled and candidates_enabled,
                dtype=torch.bool,
                device=scores.device,
            ),
            "control_stage": torch.full(
                (batch_size,),
                int(stage_index),
                dtype=torch.long,
                device=scores.device,
            ),
        }
        # These parameter-free, detached tensors are requested only by the
        # checkpoint diagnostic.  Keeping one value per parent lets the
        # caller select precisely the Gaussians in split_mask.
        control_result.update(sibling_diagnostics)
        return (
            instance_feature,
            anchor,
            next_last_operation,
            next_operation_age,
            next_prune_evidence,
            control_result,
        )


# import math
# from typing import Optional, Sequence, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from mmseg.registry import MODELS

# from .complexity_score import GaussianComplexityScore
# from ...utils.safe_ops import safe_inverse_sigmoid
# from ...utils.utils import get_rotation_matrix


# class AdaptiveThresholdGenerator(nn.Module):
#     """Generate conservative per-scene thresholds from Score statistics.

#     The absolute thresholds keep the learned score calibrated across scenes,
#     while mean/std statistics adapt candidate selection to each scene.

#     This module intentionally has no learnable parameters: hard candidate
#     masks do not provide a stable gradient for a learned threshold head.
#     """

#     def __init__(
#         self,
#         absolute_low: float = 0.05,
#         absolute_high: float = 0.20,
#         k_low: float = 1.0,
#         k_high: float = 1.0,
#     ) -> None:
#         super().__init__()
#         if not 0.0 <= absolute_low < absolute_high <= 1.0:
#             raise ValueError(
#                 "absolute thresholds must satisfy "
#                 "0 <= absolute_low < absolute_high <= 1"
#             )
#         if k_low < 0 or k_high < 0:
#             raise ValueError("k_low and k_high must be non-negative")
#         self.absolute_low = float(absolute_low)
#         self.absolute_high = float(absolute_high)
#         self.k_low = float(k_low)
#         self.k_high = float(k_high)

#     def forward(
#         self, scores: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         if scores.ndim != 2:
#             raise ValueError(
#                 f"scores must have shape [B, N], got {scores.shape}"
#             )
#         if scores.shape[1] == 0:
#             raise ValueError("Adaptive control requires at least one Gaussian")

#         # Threshold selection is a discrete control decision.  Detaching here
#         # keeps Score learning solely governed by its explicit GT supervision.
#         statistics_scores = scores.detach().float()
#         score_mean = statistics_scores.mean(dim=1, keepdim=True)
#         score_variance = (
#             statistics_scores - score_mean
#         ).square().mean(dim=1, keepdim=True)
#         score_std = torch.sqrt(score_variance.clamp_min(0.0))

#         absolute_low = torch.full_like(score_mean, self.absolute_low)
#         absolute_high = torch.full_like(score_mean, self.absolute_high)
#         tau_1 = torch.minimum(
#             absolute_low, score_mean - self.k_low * score_std
#         ).clamp_(0.0, 1.0)
#         tau_2 = torch.maximum(
#             absolute_high, score_mean + self.k_high * score_std
#         ).clamp_(0.0, 1.0)

#         # The absolute bounds guarantee this unless an invalid configuration
#         # bypasses __init__, but retain a direct assertion near mask creation.
#         if torch.any(tau_1 >= tau_2):
#             raise RuntimeError("Adaptive thresholds must satisfy tau_1 < tau_2")
#         return tau_1, tau_2, score_mean, score_std


# @MODELS.register_module()
# class AdaptiveGaussianControl(BaseModule):
#     """Score Gaussians and apply conservative split/delete/merge decisions.

#     The high-score mask expresses the learned local-complexity decision.  It
#     is not filtered by opacity or optical-density mass.  Before applying the
#     decision, this module only checks whether the two children can be encoded
#     legally: their reduced scale must remain inside ``scale_range`` and both
#     centers must remain inside ``pc_range``.

#     A selected parent is replaced by two children along its longest principal
#     axis.  For the default offset ``1 / sqrt(2)``, the child standard deviation
#     on that axis is also ``1 / sqrt(2)`` of the parent.  Each child's optical
#     strength is ``rho_parent / sqrt(2)``.  This preserves the Gaussian-mixture
#     mean, second moment and integrated optical-density mass.

#     A low-score Gaussian is deleted only when both its peak opacity and its
#     integrated optical-density mass ``rho * sx * sy * sz`` are below explicit
#     thresholds.  Low complexity alone never makes a Gaussian removable: large
#     uniform road/surface Gaussians must remain in the representation.

#     Merging is deliberately more conservative than deletion.  Only low-score
#     Gaussians which survive deletion are considered.  A chunked Euclidean KNN
#     search proposes nearby pairs, then Bhattacharyya similarity checks their
#     complete anisotropic geometry while cosine similarity checks their
#     semantic distributions.  Mutual-best matching makes the result disjoint,
#     so every Gaussian participates in at most one merge.

#     The merged Gaussian preserves integrated optical-density mass, mean and
#     covariance by moment matching.  Its semantic distribution and instance
#     feature are weighted by the same optical-density masses.  This is the
#     quantity conserved by the optical-density G2V head.

#     Dynamic topology control currently requires batch size one, matching the
#     GaussianFormer occupancy head and loss path.
#     """

#     def __init__(
#         self,
#         embed_dims: int = 256,
#         score_hidden_dims: Optional[int] = None,
#         score_dropout: float = 0.0,
#         absolute_low: float = 0.05,
#         absolute_high: float = 0.20,
#         k_low: float = 1.0,
#         k_high: float = 1.0,
#         max_low_ratio: Optional[float] = 0.20,
#         max_high_ratio: Optional[float] = 0.20,
#         warmup_iters: int = 5000,
#         enable_split: bool = False,
#         enable_delete: bool = False,
#         enable_merge: bool = False,
#         opacity_delete_threshold: float = 0.005,
#         density_mass_delete_threshold: float = 0.001,
#         merge_knn: int = 8,
#         merge_chunk_size: int = 512,
#         merge_geometry_threshold: float = 0.90,
#         merge_semantic_threshold: float = 0.95,
#         merge_eigenvalue_eps: float = 1e-8,
#         merge_mass_eps: float = 1e-12,
#         merge_probability_eps: float = 1e-8,
#         pc_range: Optional[Sequence[float]] = None,
#         scale_range: Optional[Sequence[float]] = None,
#         split_offset_ratio: float = 1.0 / math.sqrt(2.0),
#         include_opa: bool = True,
#         xyz_activation: str = "sigmoid",
#         scale_activation: str = "sigmoid",
#         position_eps: float = 1e-4,
#         scale_eps: float = 1e-6,
#         opacity_eps: float = 1e-6,
#         init_cfg=None,
#         **kwargs,
#     ) -> None:
#         super().__init__(init_cfg=init_cfg)
#         if warmup_iters < 0:
#             raise ValueError("warmup_iters must be non-negative")
#         self._validate_ratio(max_low_ratio, "max_low_ratio")
#         self._validate_ratio(max_high_ratio, "max_high_ratio")

#         self.max_low_ratio = max_low_ratio
#         self.max_high_ratio = max_high_ratio
#         self.warmup_iters = int(warmup_iters)
#         self.enable_split = bool(enable_split)
#         self.enable_delete = bool(enable_delete)
#         self.enable_merge = bool(enable_merge)
#         self.include_opa = bool(include_opa)

#         if not 0.0 <= opacity_delete_threshold < 1.0:
#             raise ValueError(
#                 "opacity_delete_threshold must lie inside [0, 1)"
#             )
#         if density_mass_delete_threshold < 0.0:
#             raise ValueError(
#                 "density_mass_delete_threshold must be non-negative"
#             )
#         self.opacity_delete_threshold = float(opacity_delete_threshold)
#         self.density_mass_delete_threshold = float(
#             density_mass_delete_threshold
#         )

#         if merge_knn <= 0:
#             raise ValueError("merge_knn must be positive")
#         if merge_chunk_size <= 0:
#             raise ValueError("merge_chunk_size must be positive")
#         if not 0.0 <= merge_geometry_threshold <= 1.0:
#             raise ValueError("merge_geometry_threshold must lie in [0, 1]")
#         if not 0.0 <= merge_semantic_threshold <= 1.0:
#             raise ValueError("merge_semantic_threshold must lie in [0, 1]")
#         if merge_eigenvalue_eps <= 0.0:
#             raise ValueError("merge_eigenvalue_eps must be positive")
#         if merge_mass_eps <= 0.0:
#             raise ValueError("merge_mass_eps must be positive")
#         if not 0.0 < merge_probability_eps < 1.0:
#             raise ValueError("merge_probability_eps must lie inside (0, 1)")
#         self.merge_knn = int(merge_knn)
#         self.merge_chunk_size = int(merge_chunk_size)
#         self.merge_geometry_threshold = float(merge_geometry_threshold)
#         self.merge_semantic_threshold = float(merge_semantic_threshold)
#         self.merge_eigenvalue_eps = float(merge_eigenvalue_eps)
#         self.merge_mass_eps = float(merge_mass_eps)
#         self.merge_probability_eps = float(merge_probability_eps)

#         if not 0.0 < split_offset_ratio < 1.0:
#             raise ValueError("split_offset_ratio must lie strictly inside (0, 1)")
#         if xyz_activation not in ("sigmoid", "identity"):
#             raise ValueError(f"Unsupported xyz_activation={xyz_activation!r}")
#         if scale_activation not in ("sigmoid", "identity"):
#             raise ValueError(f"Unsupported scale_activation={scale_activation!r}")
#         if position_eps < 0.0 or scale_eps < 0.0:
#             raise ValueError("position_eps and scale_eps must be non-negative")
#         if not 0.0 < opacity_eps < 0.5:
#             raise ValueError("opacity_eps must lie inside (0, 0.5)")

#         self.split_offset_ratio = float(split_offset_ratio)
#         self.split_scale_ratio = math.sqrt(
#             1.0 - self.split_offset_ratio ** 2
#         )
#         self.xyz_activation = xyz_activation
#         self.scale_activation = scale_activation
#         self.position_eps = float(position_eps)
#         self.scale_eps = float(scale_eps)
#         self.opacity_eps = float(opacity_eps)
#         # safe_inverse_sigmoid clamps its unit-domain input to [1e-4, .9999].
#         # Feasibility uses the same margin so a supposedly valid child is not
#         # silently moved or enlarged by the encoder clamp.
#         self.encoding_unit_eps = 1e-4

#         if self.enable_split or self.enable_merge:
#             if pc_range is None or len(pc_range) != 6:
#                 raise ValueError(
#                     "pc_range with 6 values is required for split/merge"
#                 )
#             if scale_range is None or len(scale_range) != 2:
#                 raise ValueError(
#                     "scale_range=[min, max] is required for split/merge"
#                 )
#             pc_range_tensor = torch.as_tensor(pc_range, dtype=torch.float32)
#             if torch.any(pc_range_tensor[3:] <= pc_range_tensor[:3]):
#                 raise ValueError("Each pc_range maximum must exceed its minimum")
#             scale_min, scale_max = map(float, scale_range)
#             if scale_min <= 0.0 or scale_max <= scale_min:
#                 raise ValueError("scale_range must satisfy 0 < min < max")
#         else:
#             # Buffers still have stable shapes so state/device handling remains
#             # uniform when the module is used only for Score supervision.
#             pc_range_tensor = torch.zeros(6, dtype=torch.float32)
#             scale_min, scale_max = 0.0, 1.0

#         if (
#             self.enable_split or self.enable_delete or self.enable_merge
#         ) and not self.include_opa:
#             raise ValueError(
#                 "Optical-density topology control requires include_opa=True"
#             )

#         self.register_buffer("pc_range", pc_range_tensor, persistent=False)
#         self.register_buffer(
#             "scale_range",
#             torch.tensor([scale_min, scale_max], dtype=torch.float32),
#             persistent=False,
#         )
#         self.score_head = GaussianComplexityScore(
#             embed_dims=embed_dims,
#             hidden_dims=score_hidden_dims,
#             dropout=score_dropout,
#         )
#         self.threshold_generator = AdaptiveThresholdGenerator(
#             absolute_low=absolute_low,
#             absolute_high=absolute_high,
#             k_low=k_low,
#             k_high=k_high,
#         )

#     @staticmethod
#     def _validate_ratio(value: Optional[float], name: str) -> None:
#         if value is not None and not 0.0 <= value <= 1.0:
#             raise ValueError(f"{name} must be None or lie in [0, 1]")

#     @staticmethod
#     def _global_iter_value(global_iter) -> Optional[int]:
#         if global_iter is None:
#             return None
#         if isinstance(global_iter, torch.Tensor):
#             if global_iter.numel() != 1:
#                 raise ValueError("global_iter tensor must contain one value")
#             return int(global_iter.detach().item())
#         return int(global_iter)

#     @staticmethod
#     def _cap_candidates(
#         mask: torch.Tensor,
#         scores: torch.Tensor,
#         max_ratio: Optional[float],
#         largest: bool,
#     ) -> torch.Tensor:
#         """Apply a cap only when threshold candidates exceed the ratio."""
#         if max_ratio is None or max_ratio >= 1.0:
#             return mask

#         num_gaussians = mask.shape[1]
#         if max_ratio <= 0.0:
#             return torch.zeros_like(mask)
#         max_candidates = max(1, int(num_gaussians * max_ratio))
#         if max_candidates >= num_gaussians:
#             return mask

#         detached_scores = scores.detach().float()
#         fill_value = -torch.inf if largest else torch.inf
#         ranking_scores = torch.where(
#             mask, detached_scores, torch.full_like(detached_scores, fill_value)
#         )
#         candidate_indices = torch.topk(
#             ranking_scores,
#             k=max_candidates,
#             dim=1,
#             largest=largest,
#             sorted=False,
#         ).indices
#         selected_are_valid = mask.gather(1, candidate_indices)
#         capped_mask = torch.zeros_like(mask)
#         capped_mask.scatter_(1, candidate_indices, selected_are_valid)
#         return capped_mask

#     def _candidates_enabled(self, global_iter) -> bool:
#         iteration = self._global_iter_value(global_iter)
#         # Validation during training passes global_iter and must obey the same
#         # warmup as training.  Standalone evaluation has no iteration and is
#         # treated as evaluation of a trained checkpoint, so control is active.
#         if iteration is not None:
#             return iteration >= self.warmup_iters
#         if not self.training:
#             return True
#         # During training an absent iteration is treated conservatively.  The
#         # Score loss still trains, but no topology candidate is activated.
#         return False

#     def _encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
#         pc_min = self.pc_range[:3].to(device=xyz.device, dtype=xyz.dtype)
#         pc_extent = (
#             self.pc_range[3:] - self.pc_range[:3]
#         ).to(device=xyz.device, dtype=xyz.dtype)
#         xyz_unit = (xyz - pc_min) / pc_extent
#         if self.xyz_activation == "sigmoid":
#             return safe_inverse_sigmoid(xyz_unit)
#         return xyz_unit.clamp(min=1e-6, max=1.0 - 1e-6)

#     def _encode_scale(self, scales: torch.Tensor) -> torch.Tensor:
#         scale_range = self.scale_range.to(
#             device=scales.device, dtype=scales.dtype
#         )
#         scale_unit = (
#             scales - scale_range[0]
#         ) / (
#             scale_range[1] - scale_range[0]
#         )
#         if self.scale_activation == "sigmoid":
#             return safe_inverse_sigmoid(scale_unit)
#         return scale_unit.clamp(min=1e-6, max=1.0 - 1e-6)

#     def _split_feasibility(self, gaussian):
#         """Return scale/position validity masks and proposed child geometry."""
#         means = gaussian.means
#         scales = gaussian.scales
#         rotations = gaussian.rotations
#         if means.ndim != 3 or means.shape[-1] != 3:
#             raise ValueError(f"gaussian.means must be [B, N, 3], got {means.shape}")
#         if scales.shape != means.shape:
#             raise ValueError(
#                 f"gaussian.scales must match means, got {scales.shape}"
#             )
#         if rotations.shape != (*means.shape[:2], 4):
#             raise ValueError(
#                 "gaussian.rotations must be [B, N, 4], got "
#                 f"{rotations.shape}"
#             )

#         largest_axis = scales.argmax(dim=-1)
#         largest_scale = scales.gather(
#             -1, largest_axis.unsqueeze(-1)
#         ).squeeze(-1)

#         # GaussianHead forms Cov = R^T diag(scale^2) R, hence a local
#         # principal axis is a row of R in world coordinates.
#         rotation_matrix = get_rotation_matrix(rotations)
#         axis_direction = rotation_matrix.gather(
#             -2,
#             largest_axis[..., None, None].expand(*largest_axis.shape, 1, 3),
#         ).squeeze(-2)
#         offset = (
#             largest_scale * self.split_offset_ratio
#         ).unsqueeze(-1) * axis_direction
#         child_mean_positive = means + offset
#         child_mean_negative = means - offset

#         child_scales = scales.clone()
#         child_axis_scale = largest_scale * self.split_scale_ratio
#         child_scales.scatter_(
#             -1, largest_axis.unsqueeze(-1), child_axis_scale.unsqueeze(-1)
#         )

#         scale_range = self.scale_range.to(
#             device=scales.device, dtype=scales.dtype
#         )
#         minimum_encodable_scale = scale_range[0] + (
#             scale_range[1] - scale_range[0]
#         ) * self.encoding_unit_eps
#         scale_feasible = (
#             child_axis_scale >= minimum_encodable_scale + self.scale_eps
#         )

#         pc_min = self.pc_range[:3].to(device=means.device, dtype=means.dtype)
#         pc_max = self.pc_range[3:].to(device=means.device, dtype=means.dtype)
#         encoding_margin = (pc_max - pc_min) * self.encoding_unit_eps
#         lower = pc_min + encoding_margin + self.position_eps
#         upper = pc_max - encoding_margin - self.position_eps
#         positive_inside = (
#             (child_mean_positive >= lower) & (child_mean_positive <= upper)
#         ).all(dim=-1)
#         negative_inside = (
#             (child_mean_negative >= lower) & (child_mean_negative <= upper)
#         ).all(dim=-1)
#         children_inside = positive_inside & negative_inside
#         return (
#             scale_feasible,
#             children_inside,
#             child_mean_positive,
#             child_mean_negative,
#             child_scales,
#         )

#     @staticmethod
#     def _physical_covariance(
#         scales: torch.Tensor, rotations: torch.Tensor
#     ) -> torch.Tensor:
#         """Build ``R^T diag(scale^2) R``, matching GaussianHead exactly."""
#         rotation_matrix = get_rotation_matrix(rotations)
#         return (
#             rotation_matrix.transpose(-1, -2)
#             @ torch.diag_embed(scales.square())
#             @ rotation_matrix
#         )

#     def _moment_match_geometry(
#         self,
#         mean_i: torch.Tensor,
#         mean_j: torch.Tensor,
#         covariance_i: torch.Tensor,
#         covariance_j: torch.Tensor,
#         mass_i: torch.Tensor,
#         mass_j: torch.Tensor,
#     ):
#         """Return mass weights, mean and covariance of a two-Gaussian mixture.

#         ``mass_i`` and ``mass_j`` are proportional to the true integrals of
#         optical density: the omitted constant ``(2*pi)^(3/2)`` cancels in the
#         weights.  The between-mean outer products are required for second-
#         moment conservation; merely averaging scale and rotation is invalid.
#         """
#         total_mass = mass_i + mass_j
#         safe_total_mass = total_mass.clamp_min(self.merge_mass_eps)
#         weight_i = mass_i / safe_total_mass
#         weight_i = torch.where(
#             total_mass > self.merge_mass_eps,
#             weight_i,
#             torch.full_like(weight_i, 0.5),
#         )
#         weight_j = 1.0 - weight_i
#         merged_mean = (
#             weight_i.unsqueeze(-1) * mean_i
#             + weight_j.unsqueeze(-1) * mean_j
#         )
#         delta_i = mean_i - merged_mean
#         delta_j = mean_j - merged_mean
#         merged_covariance = (
#             weight_i[..., None, None]
#             * (covariance_i + delta_i.unsqueeze(-1) * delta_i.unsqueeze(-2))
#             + weight_j[..., None, None]
#             * (covariance_j + delta_j.unsqueeze(-1) * delta_j.unsqueeze(-2))
#         )
#         # Remove the tiny antisymmetric part introduced by finite precision.
#         merged_covariance = 0.5 * (
#             merged_covariance + merged_covariance.transpose(-1, -2)
#         )
#         return weight_i, weight_j, total_mass, merged_mean, merged_covariance

#     @torch.no_grad()
#     def _select_merge_pairs(
#         self,
#         gaussian,
#         anchor: torch.Tensor,
#         merge_candidate_mask: torch.Tensor,
#     ):
#         """Select disjoint conservative pairs with chunked KNN + mutual best.

#         KNN is only a proposal mechanism.  A pair must additionally pass:

#         1. Bhattacharyya similarity of the two full anisotropic Gaussians;
#         2. cosine similarity of their semantic probabilities (18 classes,
#            including the explicit empty class, in the new optical setup);
#         3. exact encodability of the moment-matched mean, scale and opacity.

#         Among valid neighbours each candidate chooses the largest product of
#         geometry and semantic similarity.  Only mutual choices are retained,
#         which prevents one-to-many conflicts without a global matching pass.
#         """
#         batch_size, num_gaussians = merge_candidate_mask.shape
#         if batch_size != 1:
#             raise NotImplementedError(
#                 "Dynamic Gaussian topology control requires batch_size=1; "
#                 f"got batch_size={batch_size}"
#             )
#         empty_pairs = torch.empty(
#             (0, 2), dtype=torch.long, device=merge_candidate_mask.device
#         )
#         empty_similarity = torch.empty(
#             (0,), dtype=torch.float32, device=merge_candidate_mask.device
#         )

#         candidate_indices = torch.nonzero(
#             merge_candidate_mask[0], as_tuple=False
#         ).squeeze(-1)
#         num_candidates = int(candidate_indices.numel())
#         if num_candidates < 2:
#             return empty_pairs, empty_similarity, empty_similarity.clone()
#         if anchor.shape[0] != 1 or anchor.shape[1] != num_gaussians:
#             raise ValueError("anchor and merge_candidate_mask are misaligned")
#         if anchor.shape[-1] <= 11:
#             raise ValueError(
#                 "Merging requires semantic logits at anchor[..., 11:]"
#             )

#         # Pair selection is discrete, so all proposal statistics stay in
#         # float32 and detached from both Score and occupancy gradients.
#         means = gaussian.means[0, candidate_indices].detach().float()
#         scales = gaussian.scales[0, candidate_indices].detach().float()
#         rotations = gaussian.rotations[0, candidate_indices].detach().float()
#         alpha = gaussian.opacities[0, candidate_indices, 0].detach().float()
#         alpha = alpha.clamp(min=0.0, max=1.0 - self.opacity_eps)
#         rho = -torch.log1p(-alpha)
#         mass = rho * scales.prod(dim=-1)
#         covariance = self._physical_covariance(scales, rotations)
#         semantic_probability = F.softmax(
#             anchor[0, candidate_indices, 11:].detach().float(), dim=-1
#         )

#         neighbour_count = min(self.merge_knn, num_candidates - 1)
#         best_neighbour = torch.full(
#             (num_candidates,), -1, dtype=torch.long, device=means.device
#         )
#         best_geometry = torch.zeros(
#             num_candidates, dtype=torch.float32, device=means.device
#         )
#         best_semantic = torch.zeros_like(best_geometry)
#         identity = torch.eye(3, dtype=torch.float32, device=means.device)

#         scale_range = self.scale_range.to(device=means.device, dtype=torch.float32)
#         scale_margin = (
#             scale_range[1] - scale_range[0]
#         ) * self.encoding_unit_eps
#         scale_lower = scale_range[0] + scale_margin + self.scale_eps
#         scale_upper = scale_range[1] - scale_margin - self.scale_eps
#         pc_min = self.pc_range[:3].to(device=means.device, dtype=torch.float32)
#         pc_max = self.pc_range[3:].to(device=means.device, dtype=torch.float32)
#         position_margin = (pc_max - pc_min) * self.encoding_unit_eps
#         position_lower = pc_min + position_margin + self.position_eps
#         position_upper = pc_max - position_margin - self.position_eps

#         for start in range(0, num_candidates, self.merge_chunk_size):
#             end = min(start + self.merge_chunk_size, num_candidates)
#             query_count = end - start
#             distances = torch.cdist(means[start:end], means)
#             query_rows = torch.arange(query_count, device=means.device)
#             self_columns = torch.arange(start, end, device=means.device)
#             distances[query_rows, self_columns] = torch.inf
#             neighbour = torch.topk(
#                 distances,
#                 k=neighbour_count,
#                 dim=1,
#                 largest=False,
#                 sorted=False,
#             ).indices

#             mean_i = means[start:end, None, :].expand(
#                 -1, neighbour_count, -1
#             )
#             mean_j = means[neighbour]
#             covariance_i = covariance[start:end, None, :, :].expand(
#                 -1, neighbour_count, -1, -1
#             )
#             covariance_j = covariance[neighbour]

#             # Bhattacharyya distance accounts for center, scale and rotation.
#             # Unlike response-at-the-other-center, it rejects concentric but
#             # radically different Gaussians.
#             covariance_middle = 0.5 * (covariance_i + covariance_j)
#             covariance_middle = covariance_middle + (
#                 self.merge_eigenvalue_eps * identity
#             )
#             delta = mean_i - mean_j
#             solved_delta = torch.linalg.solve(
#                 covariance_middle, delta.unsqueeze(-1)
#             ).squeeze(-1)
#             quadratic = (delta * solved_delta).sum(dim=-1)
#             logdet_middle = torch.linalg.slogdet(covariance_middle).logabsdet
#             logdet_i = 2.0 * torch.log(
#                 scales[start:end].clamp_min(self.merge_eigenvalue_eps)
#             ).sum(dim=-1, keepdim=True)
#             logdet_j = 2.0 * torch.log(
#                 scales[neighbour].clamp_min(self.merge_eigenvalue_eps)
#             ).sum(dim=-1)
#             bhattacharyya_distance = (
#                 0.125 * quadratic
#                 + 0.5
#                 * (logdet_middle - 0.5 * (logdet_i + logdet_j))
#             ).clamp_min(0.0)
#             geometry_similarity = torch.exp(-bhattacharyya_distance)
#             semantic_similarity = F.cosine_similarity(
#                 semantic_probability[start:end, None, :],
#                 semantic_probability[neighbour],
#                 dim=-1,
#                 eps=self.merge_probability_eps,
#             )

#             mass_i = mass[start:end, None].expand(-1, neighbour_count)
#             mass_j = mass[neighbour]
#             (
#                 _,
#                 _,
#                 merged_mass,
#                 merged_mean,
#                 merged_covariance,
#             ) = self._moment_match_geometry(
#                 mean_i,
#                 mean_j,
#                 covariance_i,
#                 covariance_j,
#                 mass_i,
#                 mass_j,
#             )
#             eigenvalues = torch.linalg.eigvalsh(merged_covariance)
#             merged_scales = torch.sqrt(
#                 eigenvalues.clamp_min(self.merge_eigenvalue_eps)
#             )
#             merged_rho = merged_mass / merged_scales.prod(dim=-1).clamp_min(
#                 self.merge_mass_eps
#             )
#             merged_alpha = -torch.expm1(-merged_rho)

#             scale_encodable = (
#                 (merged_scales >= scale_lower)
#                 & (merged_scales <= scale_upper)
#             ).all(dim=-1)
#             mean_encodable = (
#                 (merged_mean >= position_lower)
#                 & (merged_mean <= position_upper)
#             ).all(dim=-1)
#             opacity_encodable = (
#                 (merged_alpha >= self.encoding_unit_eps)
#                 & (merged_alpha <= 1.0 - self.encoding_unit_eps)
#             )
#             finite = (
#                 torch.isfinite(geometry_similarity)
#                 & torch.isfinite(semantic_similarity)
#                 & torch.isfinite(merged_scales).all(dim=-1)
#                 & torch.isfinite(merged_alpha)
#             )
#             valid = (
#                 (geometry_similarity >= self.merge_geometry_threshold)
#                 & (semantic_similarity >= self.merge_semantic_threshold)
#                 & scale_encodable
#                 & mean_encodable
#                 & opacity_encodable
#                 & finite
#             )
#             pair_quality = geometry_similarity * semantic_similarity
#             pair_quality = pair_quality.masked_fill(~valid, -torch.inf)
#             best_quality, best_column = pair_quality.max(dim=1)
#             row_has_match = torch.isfinite(best_quality)
#             selected_neighbour = neighbour.gather(
#                 1, best_column.unsqueeze(-1)
#             ).squeeze(-1)
#             best_neighbour[start:end] = torch.where(
#                 row_has_match,
#                 selected_neighbour,
#                 torch.full_like(selected_neighbour, -1),
#             )
#             best_geometry[start:end] = geometry_similarity.gather(
#                 1, best_column.unsqueeze(-1)
#             ).squeeze(-1)
#             best_semantic[start:end] = semantic_similarity.gather(
#                 1, best_column.unsqueeze(-1)
#             ).squeeze(-1)

#         local_index = torch.arange(num_candidates, device=means.device)
#         has_neighbour = best_neighbour >= 0
#         safe_neighbour = best_neighbour.clamp_min(0)
#         mutual = (
#             has_neighbour
#             & (best_neighbour[safe_neighbour] == local_index)
#             & (local_index < best_neighbour)
#         )
#         left_local = local_index[mutual]
#         right_local = best_neighbour[left_local]
#         if left_local.numel() == 0:
#             return empty_pairs, empty_similarity, empty_similarity.clone()
#         merge_pairs = torch.stack(
#             [candidate_indices[left_local], candidate_indices[right_local]],
#             dim=-1,
#         )
#         return (
#             merge_pairs,
#             best_geometry[left_local],
#             best_semantic[left_local],
#         )

#     @staticmethod
#     def _matrix_to_quaternion(rotation_matrix: torch.Tensor) -> torch.Tensor:
#         """Convert rotation matrices to normalized ``[w, x, y, z]`` values."""
#         if rotation_matrix.shape[-2:] != (3, 3):
#             raise ValueError("rotation_matrix must end in [3, 3]")
#         m00 = rotation_matrix[..., 0, 0]
#         m11 = rotation_matrix[..., 1, 1]
#         m22 = rotation_matrix[..., 2, 2]
#         q_abs = torch.sqrt(
#             torch.clamp(
#                 torch.stack(
#                     [
#                         1.0 + m00 + m11 + m22,
#                         1.0 + m00 - m11 - m22,
#                         1.0 - m00 + m11 - m22,
#                         1.0 - m00 - m11 + m22,
#                     ],
#                     dim=-1,
#                 ),
#                 min=0.0,
#             )
#         )
#         candidates = torch.stack(
#             [
#                 torch.stack(
#                     [
#                         q_abs[..., 0].square(),
#                         rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
#                         rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
#                         rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
#                     ],
#                     dim=-1,
#                 ),
#                 torch.stack(
#                     [
#                         rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
#                         q_abs[..., 1].square(),
#                         rotation_matrix[..., 1, 0] + rotation_matrix[..., 0, 1],
#                         rotation_matrix[..., 0, 2] + rotation_matrix[..., 2, 0],
#                     ],
#                     dim=-1,
#                 ),
#                 torch.stack(
#                     [
#                         rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
#                         rotation_matrix[..., 1, 0] + rotation_matrix[..., 0, 1],
#                         q_abs[..., 2].square(),
#                         rotation_matrix[..., 2, 1] + rotation_matrix[..., 1, 2],
#                     ],
#                     dim=-1,
#                 ),
#                 torch.stack(
#                     [
#                         rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
#                         rotation_matrix[..., 0, 2] + rotation_matrix[..., 2, 0],
#                         rotation_matrix[..., 2, 1] + rotation_matrix[..., 1, 2],
#                         q_abs[..., 3].square(),
#                     ],
#                     dim=-1,
#                 ),
#             ],
#             dim=-2,
#         )
#         candidates = candidates / (
#             2.0 * q_abs[..., :, None].clamp_min(1e-8)
#         )
#         best = q_abs.argmax(dim=-1)
#         gather_index = best[..., None, None].expand(*best.shape, 1, 4)
#         quaternion = candidates.gather(-2, gather_index).squeeze(-2)
#         return F.normalize(quaternion, dim=-1)

#     def _build_merged_gaussians(
#         self,
#         instance_feature: torch.Tensor,
#         anchor: torch.Tensor,
#         gaussian,
#         merge_pairs: torch.Tensor,
#     ):
#         """Moment-match selected pairs and encode the merged anchors."""
#         index_i = merge_pairs[:, 0]
#         index_j = merge_pairs[:, 1]
#         mean_i = gaussian.means[0, index_i].float()
#         mean_j = gaussian.means[0, index_j].float()
#         scale_i = gaussian.scales[0, index_i].float()
#         scale_j = gaussian.scales[0, index_j].float()
#         covariance_i = self._physical_covariance(
#             scale_i, gaussian.rotations[0, index_i].float()
#         )
#         covariance_j = self._physical_covariance(
#             scale_j, gaussian.rotations[0, index_j].float()
#         )
#         alpha_i = gaussian.opacities[0, index_i, 0].float().clamp(
#             min=0.0, max=1.0 - self.opacity_eps
#         )
#         alpha_j = gaussian.opacities[0, index_j, 0].float().clamp(
#             min=0.0, max=1.0 - self.opacity_eps
#         )
#         mass_i = -torch.log1p(-alpha_i) * scale_i.prod(dim=-1)
#         mass_j = -torch.log1p(-alpha_j) * scale_j.prod(dim=-1)
#         (
#             weight_i,
#             weight_j,
#             merged_mass,
#             merged_mean,
#             merged_covariance,
#         ) = self._moment_match_geometry(
#             mean_i,
#             mean_j,
#             covariance_i,
#             covariance_j,
#             mass_i,
#             mass_j,
#         )

#         eigenvalues, eigenvectors = torch.linalg.eigh(merged_covariance)
#         merged_scales = torch.sqrt(
#             eigenvalues.clamp_min(self.merge_eigenvalue_eps)
#         )
#         # eigh returns Sigma = V diag(lambda) V^T, whereas the rest of this
#         # project uses Sigma = R^T diag(scale^2) R.  Therefore R must be V^T.
#         # Eigenvector derivatives contain inverse eigenvalue gaps and become
#         # unstable for isotropic/nearly isotropic covariances.  The rotation
#         # remains exactly moment matched in the forward pass, but is treated
#         # as a discrete topology result.  Mean, eigenvalue-derived scales,
#         # opacity, semantics and instance features remain differentiable.
#         merged_rotation_matrix = eigenvectors.detach().transpose(-1, -2)
#         determinant = torch.linalg.det(merged_rotation_matrix)
#         row_sign = torch.ones_like(merged_scales)
#         row_sign[..., -1] = torch.where(
#             determinant < 0.0,
#             -torch.ones_like(determinant),
#             torch.ones_like(determinant),
#         )
#         merged_rotation_matrix = (
#             row_sign.unsqueeze(-1) * merged_rotation_matrix
#         )
#         merged_rotation = self._matrix_to_quaternion(merged_rotation_matrix)

#         merged_rho = merged_mass / merged_scales.prod(dim=-1).clamp_min(
#             self.merge_mass_eps
#         )
#         merged_alpha = (-torch.expm1(-merged_rho)).clamp(
#             min=self.encoding_unit_eps,
#             max=1.0 - self.encoding_unit_eps,
#         )

#         semantic_i = F.softmax(anchor[0, index_i, 11:].float(), dim=-1)
#         semantic_j = F.softmax(anchor[0, index_j, 11:].float(), dim=-1)
#         merged_probability = (
#             weight_i.unsqueeze(-1) * semantic_i
#             + weight_j.unsqueeze(-1) * semantic_j
#         )
#         merged_probability = merged_probability / merged_probability.sum(
#             dim=-1, keepdim=True
#         ).clamp_min(self.merge_probability_eps)
#         merged_semantic_logits = torch.log(
#             merged_probability.clamp_min(self.merge_probability_eps)
#         )

#         merged_anchor = anchor[:, index_i].clone()
#         merged_anchor[..., :3] = self._encode_xyz(
#             merged_mean.unsqueeze(0).to(dtype=anchor.dtype)
#         )
#         merged_anchor[..., 3:6] = self._encode_scale(
#             merged_scales.unsqueeze(0).to(dtype=anchor.dtype)
#         )
#         merged_anchor[..., 6:10] = merged_rotation.unsqueeze(0).to(
#             dtype=anchor.dtype
#         )
#         merged_anchor[..., 10:11] = safe_inverse_sigmoid(
#             merged_alpha.unsqueeze(0).unsqueeze(-1).to(dtype=anchor.dtype)
#         )
#         merged_anchor[..., 11:] = merged_semantic_logits.unsqueeze(0).to(
#             dtype=anchor.dtype
#         )

#         feature_i = instance_feature[:, index_i]
#         feature_j = instance_feature[:, index_j]
#         merged_feature = (
#             weight_i[None, :, None].to(dtype=feature_i.dtype) * feature_i
#             + weight_j[None, :, None].to(dtype=feature_j.dtype) * feature_j
#         )
#         return merged_feature, merged_anchor

#     def _apply_topology_edits(
#         self,
#         instance_feature: torch.Tensor,
#         anchor: torch.Tensor,
#         gaussian,
#         split_mask: torch.Tensor,
#         delete_mask: torch.Tensor,
#         child_mean_positive: torch.Tensor,
#         child_mean_negative: torch.Tensor,
#         child_scales: torch.Tensor,
#         merge_pairs: torch.Tensor,
#     ):
#         """Apply all edits atomically, before the next decoder operation."""
#         batch_size, num_gaussians = split_mask.shape
#         if delete_mask.shape != split_mask.shape:
#             raise ValueError("split_mask and delete_mask must have the same shape")
#         if batch_size != 1:
#             raise NotImplementedError(
#                 "Dynamic Gaussian topology control requires batch_size=1; "
#                 f"got batch_size={batch_size}"
#             )

#         selected_split = split_mask[0]
#         selected_delete = delete_mask[0]
#         if torch.any(selected_split & selected_delete):
#             raise RuntimeError("A Gaussian cannot be split and deleted together")

#         if merge_pairs.ndim != 2 or merge_pairs.shape[-1] != 2:
#             raise ValueError("merge_pairs must have shape [P, 2]")
#         if merge_pairs.dtype != torch.long:
#             raise ValueError("merge_pairs must contain long indices")
#         if merge_pairs.numel() > 0:
#             if torch.any(merge_pairs < 0) or torch.any(
#                 merge_pairs >= num_gaussians
#             ):
#                 raise ValueError("merge_pairs contains an out-of-range index")
#             flattened_pairs = merge_pairs.flatten()
#             if torch.unique(flattened_pairs).numel() != flattened_pairs.numel():
#                 raise RuntimeError("Merge pairs must be disjoint")
#         merge_member = torch.zeros_like(selected_split)
#         if merge_pairs.numel() > 0:
#             merge_member[merge_pairs.flatten()] = True
#         if torch.any(merge_member & (selected_split | selected_delete)):
#             raise RuntimeError(
#                 "A Gaussian cannot be merged and split/deleted together"
#             )

#         num_split = int(selected_split.sum().item())
#         num_deleted = int(selected_delete.sum().item())
#         num_merge_pairs = int(merge_pairs.shape[0])
#         if num_split == 0 and num_deleted == 0 and num_merge_pairs == 0:
#             return instance_feature, anchor

#         # A split parent contributes two new children; both parents of a merge
#         # contribute one moment-matched Gaussian.  Building all outputs from
#         # the same pre-edit representation avoids order-dependent decisions.
#         keep = ~(selected_split | selected_delete | merge_member)
#         kept_anchor = anchor[:, keep]
#         kept_feature = instance_feature[:, keep]
#         output_anchors = [kept_anchor]
#         output_features = [kept_feature]

#         if num_split > 0:
#             parent_anchor = anchor[:, selected_split]
#             parent_feature = instance_feature[:, selected_split]

#             # Interleave the positive/negative child of every selected parent.
#             child_anchor = parent_anchor.unsqueeze(2).expand(
#                 -1, -1, 2, -1
#             ).clone()
#             selected_positive = child_mean_positive[:, selected_split]
#             selected_negative = child_mean_negative[:, selected_split]
#             selected_means = torch.stack(
#                 [selected_positive, selected_negative], dim=2
#             )
#             selected_scales = child_scales[:, selected_split].unsqueeze(2).expand(
#                 -1, -1, 2, -1
#             )
#             child_anchor[..., :3] = self._encode_xyz(selected_means)
#             child_anchor[..., 3:6] = self._encode_scale(selected_scales)

#             parent_alpha = gaussian.opacities[:, selected_split]
#             if parent_alpha.ndim != 3 or parent_alpha.shape[-1] != 1:
#                 raise ValueError(
#                     "gaussian.opacities must be [B, N, 1] for splitting, got "
#                     f"{parent_alpha.shape}"
#                 )
#             parent_alpha = parent_alpha.float().clamp(
#                 min=0.0, max=1.0 - self.opacity_eps
#             )
#             parent_rho = -torch.log1p(-parent_alpha)
#             # Each child volume is split_scale_ratio times the parent volume.
#             # Dividing rho by 2*ratio preserves total integrated density mass.
#             child_rho = parent_rho / (2.0 * self.split_scale_ratio)
#             child_alpha = -torch.expm1(-child_rho)
#             child_alpha = child_alpha.clamp(
#                 min=self.opacity_eps, max=1.0 - self.opacity_eps
#             ).to(dtype=anchor.dtype)
#             child_opacity_anchor = safe_inverse_sigmoid(child_alpha)
#             child_anchor[..., 10:11] = child_opacity_anchor.unsqueeze(2)

#             output_anchors.append(child_anchor.flatten(1, 2))
#             output_features.append(
#                 parent_feature.unsqueeze(2).expand(-1, -1, 2, -1).reshape(
#                     1, 2 * num_split, -1
#                 )
#             )

#         if num_merge_pairs > 0:
#             merged_feature, merged_anchor = self._build_merged_gaussians(
#                 instance_feature=instance_feature,
#                 anchor=anchor,
#                 gaussian=gaussian,
#                 merge_pairs=merge_pairs,
#             )
#             output_anchors.append(merged_anchor)
#             output_features.append(merged_feature)

#         output_anchor = torch.cat(output_anchors, dim=1)
#         output_feature = torch.cat(output_features, dim=1)
#         expected_count = (
#             num_gaussians + num_split - num_deleted - num_merge_pairs
#         )
#         if output_anchor.shape[1] != expected_count:
#             raise RuntimeError(
#                 "Topology count mismatch: expected "
#                 f"{expected_count}, got {output_anchor.shape[1]}"
#             )
#         return output_feature, output_anchor

#     def forward(
#         self,
#         instance_feature: torch.Tensor,
#         anchor: torch.Tensor,
#         anchor_embed: torch.Tensor,
#         gaussian=None,
#         stage_index: int = 0,
#         global_iter=None,
#         disable_split_at_eval: bool = False,
#     ):
#         if disable_split_at_eval and self.training:
#             raise RuntimeError(
#                 "disable_split_at_eval is an evaluation-only diagnostic "
#                 "switch and must not be used while the model is training"
#             )
#         split_enabled = self.enable_split and not disable_split_at_eval

#         if anchor.shape[:2] != instance_feature.shape[:2]:
#             raise ValueError(
#                 "anchor and instance_feature must share [B, N], got "
#                 f"{anchor.shape[:2]} and {instance_feature.shape[:2]}"
#             )

#         score_logits = self.score_head(instance_feature, anchor_embed)
#         scores = score_logits.sigmoid()
#         tau_1, tau_2, score_mean, score_std = self.threshold_generator(scores)

#         candidates_enabled = self._candidates_enabled(global_iter)
#         if candidates_enabled:
#             low_mask = scores.detach() <= tau_1
#             high_mask = scores.detach() >= tau_2
#             low_mask = self._cap_candidates(
#                 low_mask,
#                 scores,
#                 max_ratio=self.max_low_ratio,
#                 largest=False,
#             )
#             high_mask = self._cap_candidates(
#                 high_mask,
#                 scores,
#                 max_ratio=self.max_high_ratio,
#                 largest=True,
#             )
#         else:
#             low_mask = torch.zeros_like(scores, dtype=torch.bool)
#             high_mask = torch.zeros_like(scores, dtype=torch.bool)

#         if (
#             self.enable_split or self.enable_delete or self.enable_merge
#         ) and gaussian is None:
#             raise ValueError(
#                 "The refined physical gaussian is required for topology control"
#             )

#         batch_size, num_gaussians = scores.shape
#         if gaussian is not None:
#             opacity = gaussian.opacities
#             scales = gaussian.scales
#             if opacity.shape != (batch_size, num_gaussians, 1):
#                 raise ValueError(
#                     "gaussian.opacities must be [B, N, 1], got "
#                     f"{opacity.shape}"
#                 )
#             if scales.shape != (batch_size, num_gaussians, 3):
#                 raise ValueError(
#                     "gaussian.scales must be [B, N, 3], got "
#                     f"{scales.shape}"
#                 )
#             # Deletion is a non-differentiable topology decision.  Keep its
#             # contribution statistics detached from Score and occupancy loss.
#             opacity_for_control = opacity.detach().float().squeeze(-1).clamp(
#                 min=0.0, max=1.0 - self.opacity_eps
#             )
#             optical_rho = -torch.log1p(-opacity_for_control)
#             density_mass = optical_rho * scales.detach().float().prod(dim=-1)
#         else:
#             opacity_for_control = torch.zeros_like(scores, dtype=torch.float32)
#             density_mass = torch.zeros_like(scores, dtype=torch.float32)

#         low_opacity_mask = (
#             opacity_for_control <= self.opacity_delete_threshold
#         )
#         low_density_mass_mask = (
#             density_mass <= self.density_mass_delete_threshold
#         )
#         # Candidate classification and topology execution are separate.  A
#         # merge-only experiment still needs to recognize negligible
#         # Gaussians and exclude them from merging, although it deliberately
#         # keeps them in the representation for an attributable experiment.
#         delete_candidate_mask = torch.zeros_like(low_mask)
#         if candidates_enabled:
#             delete_candidate_mask = (
#                 low_mask & low_opacity_mask & low_density_mass_mask
#             )
#         delete_mask = torch.zeros_like(low_mask)
#         if self.enable_delete:
#             delete_mask = delete_candidate_mask

#         scale_feasible_mask = torch.ones_like(high_mask)
#         children_inside_mask = torch.ones_like(high_mask)
#         split_mask = torch.zeros_like(high_mask)
#         child_mean_positive = None
#         child_mean_negative = None
#         child_scales = None
#         # Skip all physical split geometry during Score warmup and when the
#         # learned threshold yields no high-complexity candidate.
#         should_prepare_split = (
#             split_enabled
#             and candidates_enabled
#             and bool(torch.any(high_mask).item())
#         )
#         if should_prepare_split:
#             (
#                 scale_feasible_mask,
#                 children_inside_mask,
#                 child_mean_positive,
#                 child_mean_negative,
#                 child_scales,
#             ) = self._split_feasibility(gaussian)
#             split_mask = (
#                 high_mask & scale_feasible_mask & children_inside_mask
#             )

#         if torch.any(split_mask & delete_mask):
#             raise RuntimeError("Split and delete masks must be disjoint")

#         merge_candidate_mask = torch.zeros_like(low_mask)
#         merge_member_mask = torch.zeros_like(low_mask)
#         merge_partner_index = torch.full_like(
#             low_mask, -1, dtype=torch.long
#         )
#         merge_pairs = torch.empty(
#             (0, 2), dtype=torch.long, device=scores.device
#         )
#         merge_geometry_similarity = torch.empty(
#             (0,), dtype=torch.float32, device=scores.device
#         )
#         merge_semantic_similarity = torch.empty_like(
#             merge_geometry_similarity
#         )
#         should_prepare_merge = (
#             self.enable_merge
#             and candidates_enabled
#             and bool(torch.any(low_mask & ~delete_candidate_mask).item())
#         )
#         if should_prepare_merge:
#             # Delete-candidate classification always has priority, even in a
#             # merge-only ablation where deletion itself is disabled.  Split
#             # candidates are high-score and already disjoint from this set.
#             merge_candidate_mask = low_mask & ~delete_candidate_mask
#             (
#                 merge_pairs,
#                 merge_geometry_similarity,
#                 merge_semantic_similarity,
#             ) = self._select_merge_pairs(
#                 gaussian=gaussian,
#                 anchor=anchor,
#                 merge_candidate_mask=merge_candidate_mask,
#             )
#             if merge_pairs.numel() > 0:
#                 left = merge_pairs[:, 0]
#                 right = merge_pairs[:, 1]
#                 merge_member_mask[0, merge_pairs.flatten()] = True
#                 merge_partner_index[0, left] = right
#                 merge_partner_index[0, right] = left

#         if torch.any(
#             merge_member_mask
#             & (split_mask | delete_mask | delete_candidate_mask)
#         ):
#             raise RuntimeError(
#                 "Merge pairs overlap split/delete candidates or decisions"
#             )
#         has_topology_edit = bool(
#             torch.any(split_mask | delete_mask).item()
#         ) or merge_pairs.shape[0] > 0
#         if has_topology_edit:
#             instance_feature, anchor = self._apply_topology_edits(
#                 instance_feature=instance_feature,
#                 anchor=anchor,
#                 gaussian=gaussian,
#                 split_mask=split_mask,
#                 delete_mask=delete_mask,
#                 child_mean_positive=child_mean_positive,
#                 child_mean_negative=child_mean_negative,
#                 child_scales=child_scales,
#                 merge_pairs=merge_pairs,
#             )

#         num_split = split_mask.sum(dim=1)
#         num_deleted = delete_mask.sum(dim=1)
#         num_merge_pairs = torch.full(
#             (batch_size,),
#             int(merge_pairs.shape[0]),
#             dtype=torch.long,
#             device=scores.device,
#         )
#         num_before = torch.full(
#             (batch_size,),
#             num_gaussians,
#             dtype=torch.long,
#             device=scores.device,
#         )
#         control_result = {
#             "score_logits": score_logits,
#             "score": scores,
#             "tau_1": tau_1,
#             "tau_2": tau_2,
#             "score_mean": score_mean,
#             "score_std": score_std,
#             "low_candidate_mask": low_mask,
#             "high_candidate_mask": high_mask,
#             "scale_feasible_mask": scale_feasible_mask,
#             "children_inside_pc_range_mask": children_inside_mask,
#             "split_mask": split_mask,
#             "gaussian_opacity": opacity_for_control,
#             "density_mass": density_mass,
#             "low_opacity_mask": low_opacity_mask,
#             "low_density_mass_mask": low_density_mass_mask,
#             "delete_candidate_mask": delete_candidate_mask,
#             "delete_mask": delete_mask,
#             "merge_candidate_mask": merge_candidate_mask,
#             "merge_member_mask": merge_member_mask,
#             "merge_partner_index": merge_partner_index,
#             "merge_pairs": merge_pairs,
#             "merge_geometry_similarity": merge_geometry_similarity,
#             "merge_semantic_similarity": merge_semantic_similarity,
#             "opacity_delete_threshold": torch.full(
#                 (batch_size, 1),
#                 self.opacity_delete_threshold,
#                 dtype=torch.float32,
#                 device=scores.device,
#             ),
#             "density_mass_delete_threshold": torch.full(
#                 (batch_size, 1),
#                 self.density_mass_delete_threshold,
#                 dtype=torch.float32,
#                 device=scores.device,
#             ),
#             "merge_geometry_threshold": torch.full(
#                 (batch_size, 1),
#                 self.merge_geometry_threshold,
#                 dtype=torch.float32,
#                 device=scores.device,
#             ),
#             "merge_semantic_threshold": torch.full(
#                 (batch_size, 1),
#                 self.merge_semantic_threshold,
#                 dtype=torch.float32,
#                 device=scores.device,
#             ),
#             "num_gaussians_before": num_before,
#             "num_split": num_split,
#             "num_deleted": num_deleted,
#             "num_merge_pairs": num_merge_pairs,
#             "num_gaussians_after": (
#                 num_before + num_split - num_deleted - num_merge_pairs
#             ),
#             "control_enabled": torch.full(
#                 (batch_size,),
#                 candidates_enabled,
#                 dtype=torch.bool,
#                 device=scores.device,
#             ),
#             "split_enabled": torch.full(
#                 (batch_size,),
#                 split_enabled and candidates_enabled,
#                 dtype=torch.bool,
#                 device=scores.device,
#             ),
#             "delete_enabled": torch.full(
#                 (batch_size,),
#                 self.enable_delete and candidates_enabled,
#                 dtype=torch.bool,
#                 device=scores.device,
#             ),
#             "merge_enabled": torch.full(
#                 (batch_size,),
#                 self.enable_merge and candidates_enabled,
#                 dtype=torch.bool,
#                 device=scores.device,
#             ),
#             "control_stage": torch.full(
#                 (batch_size,),
#                 int(stage_index),
#                 dtype=torch.long,
#                 device=scores.device,
#             ),
#         }
#         return instance_feature, anchor, control_result
