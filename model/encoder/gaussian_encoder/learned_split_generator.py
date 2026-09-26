import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ...utils.safe_ops import (
    safe_inverse_sigmoid,
    safe_sigmoid,
)
from ...utils.utils import get_rotation_matrix


@MODELS.register_module()
class LearnedGaussianSplitGenerator(BaseModule):
    """Replace one selected parent with two learned child Gaussians.

    The generator consumes the same parent ``instance_feature + anchor_embed``
    context used by refinement.  Following the residual child parameterization
    in SAGFormer, one MLP predicts independent position, positive scale,
    opacity, semantic and feature updates for both children.  Rotation is
    copied from the parent.  The two children are returned to
    ``AdaptiveGaussianControl``, where they atomically replace the parent.

    Position offsets are bounded in the parent's local ellipsoid.  Scale is a
    learned positive multiplier, opacity is updated in logit space, and
    semantics/features are residual updates.  No optical-density mass
    conservation constraint is imposed.

    Anchor layout:
        xyz (3) + scale (3) + quaternion (4) + opacity (1) + semantics (C)
    """

    def __init__(
        self,
        embed_dims: int = 256,
        hidden_dims: Optional[int] = None,
        semantic_dim: int = 18,
        pc_range: Optional[Sequence[float]] = None,
        scale_range: Optional[Sequence[float]] = None,
        split_position_limit_ratio: float = 1.0,
        split_offset_ratio: float = 1.0 / math.sqrt(2.0),
        split_scale_ratio: Optional[float] = None,
        opacity_delta_factor: float = 1.0,
        semantic_delta_factor: float = 1.0,
        feature_delta_factor: float = 1.0,
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
        hidden_dims = int(hidden_dims or embed_dims)
        if embed_dims <= 0 or hidden_dims <= 0:
            raise ValueError("embed_dims and hidden_dims must be positive")
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        if pc_range is None or len(pc_range) != 6:
            raise ValueError("pc_range must contain 6 values")
        if scale_range is None or len(scale_range) != 2:
            raise ValueError("scale_range must be [min, max]")
        if not 0.0 < split_position_limit_ratio <= 1.0:
            raise ValueError(
                "split_position_limit_ratio must lie inside (0, 1]"
            )
        if not 0.0 < split_offset_ratio < split_position_limit_ratio:
            raise ValueError(
                "split_offset_ratio must be smaller than "
                "split_position_limit_ratio"
            )
        if split_scale_ratio is None:
            split_scale_ratio = math.sqrt(1.0 - split_offset_ratio ** 2)
        if not 0.0 < split_scale_ratio < 1.0:
            raise ValueError("split_scale_ratio must lie inside (0, 1)")
        if opacity_delta_factor < 0.0:
            raise ValueError("opacity_delta_factor must be non-negative")
        if semantic_delta_factor < 0.0:
            raise ValueError("semantic_delta_factor must be non-negative")
        if feature_delta_factor < 0.0:
            raise ValueError("feature_delta_factor must be non-negative")
        if not include_opa:
            raise ValueError(
                "Learned optical-density splitting requires include_opa=True"
            )
        if xyz_activation not in ("sigmoid", "identity"):
            raise ValueError(f"Unsupported xyz_activation={xyz_activation!r}")
        if scale_activation not in ("sigmoid", "identity"):
            raise ValueError(
                f"Unsupported scale_activation={scale_activation!r}"
            )
        if position_eps < 0.0 or scale_eps < 0.0:
            raise ValueError("position_eps and scale_eps must be non-negative")
        if not 0.0 < opacity_eps < 0.5:
            raise ValueError("opacity_eps must lie inside (0, 0.5)")

        pc_range_tensor = torch.as_tensor(pc_range, dtype=torch.float32)
        if torch.any(pc_range_tensor[3:] <= pc_range_tensor[:3]):
            raise ValueError("Each pc_range maximum must exceed its minimum")
        scale_min, scale_max = map(float, scale_range)
        if scale_min <= 0.0 or scale_max <= scale_min:
            raise ValueError("scale_range must satisfy 0 < min < max")

        self.embed_dims = int(embed_dims)
        self.hidden_dims = hidden_dims
        self.semantic_dim = int(semantic_dim)
        self.anchor_dim = 11 + self.semantic_dim
        self.split_position_limit_ratio = float(
            split_position_limit_ratio
        )
        self.split_offset_ratio = float(split_offset_ratio)
        self.split_scale_ratio = float(split_scale_ratio)
        self.opacity_delta_factor = float(opacity_delta_factor)
        self.semantic_delta_factor = float(semantic_delta_factor)
        self.feature_delta_factor = float(feature_delta_factor)
        self.xyz_activation = xyz_activation
        self.scale_activation = scale_activation
        self.position_eps = float(position_eps)
        self.scale_eps = float(scale_eps)
        self.opacity_eps = float(opacity_eps)
        self.encoding_unit_eps = 1e-4

        # Per child: local xyz offset, positive scale multiplier, opacity
        # residual, semantic residual and instance-feature residual.  Rotation
        # is intentionally copied from the parent.
        self.child_output_dim = (
            3 + 3 + 1 + self.semantic_dim + self.embed_dims
        )
        self.context_norm = nn.LayerNorm(self.embed_dims)
        self.split_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            # nn.LayerNorm(self.hidden_dims),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            # nn.LayerNorm(self.hidden_dims),
            nn.Linear(self.hidden_dims, 2 * self.child_output_dim),
        )

        self.register_buffer("pc_range", pc_range_tensor, persistent=False)
        self.register_buffer(
            "scale_range",
            torch.tensor([scale_min, scale_max], dtype=torch.float32),
            persistent=False,
        )

    def init_weight(self) -> None:
        """Start near the parent while breaking child-exchange symmetry."""
        for module in self.split_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        final_linear = self.split_mlp[-1]
        # Residual branches start at zero.  Only the two independent position
        # blocks receive a small random initialization, otherwise two exactly
        # identical children receive symmetric gradients and remain collapsed.
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        final_weight = final_linear.weight.view(
            2, self.child_output_dim, self.hidden_dims
        )
        position_init_std = (
            0.5 * self.split_offset_ratio / math.sqrt(self.hidden_dims)
        )
        nn.init.normal_(
            final_weight[:, :3], mean=0.0, std=position_init_std
        )

        # beta = softplus(raw_scale) is initialized to split_scale_ratio.
        scale_bias = math.log(math.expm1(self.split_scale_ratio))
        final_bias = final_linear.bias.view(2, self.child_output_dim)
        nn.init.constant_(final_bias[:, 3:6], scale_bias)
        nn.init.ones_(self.context_norm.weight)
        nn.init.zeros_(self.context_norm.bias)

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

    def _split_output(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        cursor = 0
        result = {}
        for name, width in (
            ("position", 3),
            ("scale", 3),
            ("opacity", 1),
            ("semantic", self.semantic_dim),
            ("feature", self.embed_dims),
        ):
            result[name] = output[..., cursor : cursor + width]
            cursor += width
        if cursor != self.child_output_dim:
            raise RuntimeError("Learned split output layout is inconsistent")
        return result

    @staticmethod
    @torch.no_grad()
    def _sibling_diagnostics(
        child_means: torch.Tensor,
        child_scales: torch.Tensor,
        child_rotations: torch.Tensor,
        child_rho: torch.Tensor,
        child_alpha: torch.Tensor,
        child_semantics: torch.Tensor,
        child_features: torch.Tensor,
        normalized_local_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Measure whether the two learned children actually specialize.

        Every returned tensor is detached and has shape ``[B, N]``.  The
        caller later indexes it with the actual ``split_mask``; diagnostics
        for unselected parents therefore never enter the reported result.

        Bhattacharyya similarity measures complete geometric overlap and is
        one for identical Gaussians.  Semantic Jensen--Shannon divergence is
        normalized by log(2), so it lies in [0, 1].
        """
        means = child_means.detach().float()
        scales = child_scales.detach().float().clamp_min(1e-8)
        rotations = F.normalize(
            child_rotations.detach().float(), p=2, dim=-1
        )
        rho = child_rho.detach().float().squeeze(-1).clamp_min(1e-8)
        alpha = child_alpha.detach().float().squeeze(-1)
        semantics = child_semantics.detach().float()
        features = child_features.detach().float()
        normalized_positions = normalized_local_positions.detach().float()

        center_delta = means[:, :, 0] - means[:, :, 1]
        center_distance = torch.linalg.vector_norm(center_delta, dim=-1)
        normalized_center_distance = torch.linalg.vector_norm(
            normalized_positions[:, :, 0]
            - normalized_positions[:, :, 1],
            dim=-1,
        )

        rotation_matrix = get_rotation_matrix(rotations)
        covariance = (
            rotation_matrix.transpose(-1, -2)
            @ torch.diag_embed(scales.square())
            @ rotation_matrix
        )
        covariance_0 = covariance[:, :, 0]
        covariance_1 = covariance[:, :, 1]
        covariance_mean = 0.5 * (covariance_0 + covariance_1)
        delta_column = center_delta.unsqueeze(-1)
        solved_delta = torch.linalg.solve(
            covariance_mean, delta_column
        )
        mahalanobis_term = (
            center_delta.unsqueeze(-2) @ solved_delta
        ).squeeze(-1).squeeze(-1) / 8.0
        _, logdet_0 = torch.linalg.slogdet(covariance_0)
        _, logdet_1 = torch.linalg.slogdet(covariance_1)
        _, logdet_mean = torch.linalg.slogdet(covariance_mean)
        determinant_term = 0.5 * (
            logdet_mean - 0.5 * (logdet_0 + logdet_1)
        )
        bhattacharyya_distance = (
            mahalanobis_term + determinant_term
        ).clamp_min(0.0)
        geometry_similarity = torch.exp(-bhattacharyya_distance).clamp(
            min=0.0, max=1.0
        )

        semantic_probability = semantics.softmax(dim=-1)
        probability_0 = semantic_probability[:, :, 0]
        probability_1 = semantic_probability[:, :, 1]
        probability_mean = 0.5 * (probability_0 + probability_1)
        probability_eps = 1e-8
        kl_0 = (
            probability_0
            * (
                probability_0.clamp_min(probability_eps).log()
                - probability_mean.clamp_min(probability_eps).log()
            )
        ).sum(dim=-1)
        kl_1 = (
            probability_1
            * (
                probability_1.clamp_min(probability_eps).log()
                - probability_mean.clamp_min(probability_eps).log()
            )
        ).sum(dim=-1)
        semantic_js = (0.5 * (kl_0 + kl_1) / math.log(2.0)).clamp(
            min=0.0, max=1.0
        )
        semantic_argmax_disagreement = (
            probability_0.argmax(dim=-1)
            != probability_1.argmax(dim=-1)
        )

        feature_cosine = F.cosine_similarity(
            features[:, :, 0], features[:, :, 1], dim=-1, eps=1e-8
        ).clamp(min=-1.0, max=1.0)
        opacity_difference = (
            alpha[:, :, 0] - alpha[:, :, 1]
        ).abs()
        log_density_difference = (
            rho[:, :, 0].log() - rho[:, :, 1].log()
        ).abs()
        scale_log_distance = torch.linalg.vector_norm(
            scales[:, :, 0].log() - scales[:, :, 1].log(), dim=-1
        )
        quaternion_dot = (
            rotations[:, :, 0] * rotations[:, :, 1]
        ).sum(dim=-1).abs().clamp(max=1.0)
        rotation_angle_degrees = (
            2.0 * torch.acos(quaternion_dot) * (180.0 / math.pi)
        )

        return {
            "sibling_center_distance": center_distance,
            "sibling_center_distance_normalized": (
                normalized_center_distance
            ),
            "sibling_geometry_similarity": geometry_similarity,
            "sibling_semantic_js_divergence": semantic_js,
            "sibling_semantic_argmax_disagreement": (
                semantic_argmax_disagreement
            ),
            "sibling_feature_cosine_similarity": feature_cosine,
            "sibling_opacity_abs_difference": opacity_difference,
            "sibling_log_density_abs_difference": (
                log_density_difference
            ),
            "sibling_scale_log_distance": scale_log_distance,
            "sibling_rotation_angle_degrees": rotation_angle_degrees,
        }

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        gaussian,
        collect_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if instance_feature.shape != anchor_embed.shape:
            raise ValueError(
                "instance_feature and anchor_embed must have the same shape, "
                f"got {instance_feature.shape} and {anchor_embed.shape}"
            )
        if anchor.shape[:2] != instance_feature.shape[:2]:
            raise ValueError(
                "anchor and instance_feature must share [B, N], got "
                f"{anchor.shape[:2]} and {instance_feature.shape[:2]}"
            )
        if anchor.shape[-1] != self.anchor_dim:
            raise ValueError(
                f"Expected anchor dim {self.anchor_dim}, got {anchor.shape[-1]}"
            )

        batch_size, num_gaussians = anchor.shape[:2]
        means = gaussian.means.float()
        scales = gaussian.scales.float()
        rotations = F.normalize(
            gaussian.rotations.float(), p=2, dim=-1
        )
        opacities = gaussian.opacities.float()
        if means.shape != (batch_size, num_gaussians, 3):
            raise ValueError("gaussian.means must be [B, N, 3]")
        if scales.shape != means.shape:
            raise ValueError("gaussian.scales must be [B, N, 3]")
        if rotations.shape != (batch_size, num_gaussians, 4):
            raise ValueError("gaussian.rotations must be [B, N, 4]")
        if opacities.shape != (batch_size, num_gaussians, 1):
            raise ValueError("gaussian.opacities must be [B, N, 1]")

        context = self.context_norm(instance_feature + anchor_embed)
        raw_output = self.split_mlp(context).reshape(
            batch_size,
            num_gaussians,
            2,
            self.child_output_dim,
        )
        residual = self._split_output(raw_output)

        # Learn an arbitrary direction for each child, but keep its center
        # inside the parent's oriented local ellipsoid.  Normalizing vectors
        # whose norm exceeds one enforces ||u||_2 <= 1.
        local_unit = torch.tanh(residual["position"].float())
        local_norm = torch.linalg.vector_norm(
            local_unit, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        local_unit = local_unit / torch.maximum(
            local_norm, torch.ones_like(local_norm)
        )
        local_unit = self.split_position_limit_ratio * local_unit
        local_offset = scales.unsqueeze(2) * local_unit
        rotation_matrix = get_rotation_matrix(rotations)
        world_offset = torch.einsum(
            "bnki,bnij->bnkj", local_offset, rotation_matrix
        )
        unclamped_child_means = means.unsqueeze(2) + world_offset

        pc_min = self.pc_range[:3].to(
            device=means.device, dtype=means.dtype
        )
        pc_max = self.pc_range[3:].to(
            device=means.device, dtype=means.dtype
        )
        encoding_margin = (pc_max - pc_min) * self.encoding_unit_eps
        lower = pc_min + encoding_margin + self.position_eps
        upper = pc_max - encoding_margin - self.position_eps
        child_means = torch.maximum(
            torch.minimum(unclamped_child_means, upper), lower
        )
        position_was_clamped = (
            child_means != unclamped_child_means
        ).any(dim=-1)

        # SAGFormer-style positive, independently learned scale multiplier.
        # It starts at split_scale_ratio but is not constrained below one.
        scale_multiplier = F.softplus(
            residual["scale"].float()
        ) + self.scale_eps
        scale_min = self.scale_range[0].to(
            device=scales.device, dtype=scales.dtype
        )
        scale_max = self.scale_range[1].to(
            device=scales.device, dtype=scales.dtype
        )
        child_scales = scales.unsqueeze(2) * scale_multiplier
        child_scales = child_scales.clamp(
            min=scale_min + self.scale_eps,
            max=scale_max - self.scale_eps,
        )
        scale_feasible = (
            scales > scale_min + self.scale_eps
        ).any(dim=-1)

        # Rotation is deliberately inherited.  All other appearance/state
        # attributes receive independent residuals for the two children.
        child_rotations = rotations.unsqueeze(2).expand(-1, -1, 2, -1)
        parent_alpha = opacities.clamp(
            min=self.opacity_eps, max=1.0 - self.opacity_eps
        )
        parent_opacity_logit = safe_inverse_sigmoid(parent_alpha)
        child_opacity_logit = (
            parent_opacity_logit.unsqueeze(2)
            + self.opacity_delta_factor
            * residual["opacity"].float()
        )
        child_alpha = safe_sigmoid(child_opacity_logit).clamp(
            min=self.opacity_eps, max=1.0 - self.opacity_eps
        )

        parent_semantics = anchor[
            ..., 11 : 11 + self.semantic_dim
        ].float()
        child_semantics = (
            parent_semantics.unsqueeze(2)
            + self.semantic_delta_factor
            * residual["semantic"].float()
        )
        child_features = (
            instance_feature.unsqueeze(2)
            + self.feature_delta_factor * residual["feature"]
        )

        child_anchor = torch.cat(
            [
                self._encode_xyz(child_means),
                self._encode_scale(child_scales),
                child_rotations,
                safe_inverse_sigmoid(child_alpha),
                child_semantics,
            ],
            dim=-1,
        ).to(dtype=anchor.dtype)
        child_features = child_features.to(dtype=instance_feature.dtype)

        # Diagnostic only: unlike the legacy analytical split, this ratio is
        # measured rather than forced to one.
        parent_rho = -torch.log1p(-parent_alpha)
        child_rho = -torch.log1p(-child_alpha)
        parent_mass = (
            parent_rho.squeeze(-1) * scales.prod(dim=-1)
        )
        child_mass = (
            child_rho.squeeze(-1) * child_scales.prod(dim=-1)
        ).sum(dim=2)
        density_mass_ratio = child_mass / parent_mass.clamp_min(
            self.opacity_eps
        )

        children_inside = (
            (child_means >= lower) & (child_means <= upper)
        ).all(dim=-1).all(dim=2)
        result = {
            "child_anchor": child_anchor,
            "child_feature": child_features,
            "scale_feasible_mask": scale_feasible,
            "children_inside_pc_range_mask": children_inside,
            "position_was_clamped_mask": position_was_clamped.any(dim=2),
            "density_mass_ratio": density_mass_ratio,
            "child_opacity": child_alpha.squeeze(-1),
        }
        if collect_diagnostics:
            result["sibling_diagnostics"] = self._sibling_diagnostics(
                child_means=child_means,
                child_scales=child_scales,
                child_rotations=child_rotations,
                child_rho=child_rho,
                child_alpha=child_alpha,
                child_semantics=child_semantics,
                child_features=child_features,
                normalized_local_positions=local_unit,
            )
        return result