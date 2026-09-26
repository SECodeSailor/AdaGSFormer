from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .utils import GaussianPrediction, cartesian, linear_relu_ln, reverse_cartesian
from ...utils.safe_ops import safe_sigmoid

'''
    refine_layer=dict(
        type="ResidualGaussianRefinementModule",
        embed_dims=embed_dims,
        pc_range=pc_range,
        scene_size=scene_size,
        semantics=True,
        semantic_dim=17,
        include_opa=True,
        semantics_activation="identity",
    )
    
    adaptive_control_layer=dict(
        type="AdaptiveGaussianControl",
        embed_dims=embed_dims,
    )
    
    share_adaptive_control_layer=True
'''

@MODELS.register_module()
class ResidualGaussianRefinementModule(BaseModule):
    """Refine every Gaussian parameter as a residual of its current value.

    Unlike the original GaussianFormer refiner, this module does not replace
    scale, rotation, opacity and semantic logits at every decoder stage.  This
    is important for grid initialization and for future split children: a zero
    predicted residual preserves the input Gaussian.

    The physical xyz update is bounded.  When ``max_xyz_offset`` is omitted,
    it is derived from ``pc_range`` and ``scene_size`` so changing the coarse
    grid resolution does not require editing another geometry parameter.

    Anchor layout:
        xyz (3) + scale (3) + quaternion (4) + opacity (0/1) + semantics (C)
    """

    def __init__(
        self,
        embed_dims: int = 256,
        pc_range: Sequence[float] = None,
        scene_size: Optional[Sequence[int]] = None,
        scale_range: Optional[Sequence[float]] = None,
        scale_min_ratio: float = 0.01,
        scale_max_ratio: Optional[float] = None,
        overlap_ratio: float = 1.25,
        max_xyz_offset: Optional[Sequence[float]] = None,
        position_step_ratio: float = 0.5,
        scale_delta_factor: float = 1.0,
        rotation_delta_factor: float = 0.5,
        opacity_delta_factor: float = 1.0,
        semantic_delta_factor: float = 1.0,
        semantics: bool = True,
        semantic_dim: Optional[int] = 17,
        include_opa: bool = True,
        semantics_activation: str = "identity",
        xyz_activation: str = "sigmoid",
        scale_activation: str = "sigmoid",
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if pc_range is None or len(pc_range) != 6:
            raise ValueError("pc_range must contain 6 values")
        if embed_dims <= 0:
            raise ValueError(f"embed_dims must be positive, got {embed_dims}")
        if semantics and (semantic_dim is None or semantic_dim <= 0):
            raise ValueError(
                "semantic_dim must be positive when semantics=True"
            )
        if xyz_activation not in ("sigmoid", "identity"):
            raise ValueError(f"Unsupported xyz_activation={xyz_activation!r}")
        if scale_activation not in ("sigmoid", "identity"):
            raise ValueError(f"Unsupported scale_activation={scale_activation!r}")
        if semantics_activation not in ("identity", "softmax", "softplus"):
            raise ValueError(
                f"Unsupported semantics_activation={semantics_activation!r}"
            )
        if position_step_ratio <= 0:
            raise ValueError("position_step_ratio must be positive")
        if scale_min_ratio <= 0:
            raise ValueError("scale_min_ratio must be positive")
        if scale_max_ratio is not None and scale_max_ratio <= 0:
            raise ValueError("scale_max_ratio must be positive")
        if overlap_ratio <= 0:
            raise ValueError("overlap_ratio must be positive")

        pc_range_tensor = torch.as_tensor(pc_range, dtype=torch.float32)
        pc_extent = pc_range_tensor[3:] - pc_range_tensor[:3]
        if torch.any(pc_extent <= 0):
            raise ValueError("Each pc_range maximum must exceed its minimum")

        grid_resolution = None
        if scene_size is not None:
            if len(scene_size) != 3:
                raise ValueError("scene_size must contain 3 integers")
            scene_size_tensor = torch.as_tensor(scene_size, dtype=torch.float32)
            if torch.any(scene_size_tensor <= 0):
                raise ValueError("scene_size must be positive")
            if not torch.equal(scene_size_tensor, scene_size_tensor.round()):
                raise ValueError("scene_size must contain integers")
            grid_resolution = pc_extent / scene_size_tensor

        if max_xyz_offset is None:
            if grid_resolution is None:
                raise ValueError(
                    "Provide scene_size or max_xyz_offset so the bounded xyz "
                    "update can be determined"
                )
            max_xyz_offset_tensor = grid_resolution * float(position_step_ratio)
        else:
            max_xyz_offset_tensor = torch.as_tensor(
                max_xyz_offset, dtype=torch.float32
            )
            if max_xyz_offset_tensor.numel() == 1:
                max_xyz_offset_tensor = max_xyz_offset_tensor.repeat(3)
            if max_xyz_offset_tensor.numel() != 3:
                raise ValueError("max_xyz_offset must be a scalar or 3 values")
        if torch.any(max_xyz_offset_tensor <= 0):
            raise ValueError("max_xyz_offset must be positive")

        if scale_range is None:
            if grid_resolution is None:
                raise ValueError(
                    "Provide scene_size or scale_range so physical scales can "
                    "be decoded"
                )
            resolved_max_ratio = (
                float(overlap_ratio)
                if scale_max_ratio is None
                else float(scale_max_ratio)
            )
            scale_min = float(grid_resolution.min()) * float(scale_min_ratio)
            scale_max = float(grid_resolution.max()) * resolved_max_ratio
        else:
            if len(scale_range) != 2:
                raise ValueError("scale_range must be [scale_min, scale_max]")
            scale_min, scale_max = map(float, scale_range)
        if scale_min <= 0 or scale_max <= scale_min:
            raise ValueError("scale_range must satisfy 0 < min < max")

        self.embed_dims = int(embed_dims)
        self.semantic_dim = int(semantic_dim) if semantics else 0
        self.include_opa = bool(include_opa)
        self.semantic_start = 10 + int(self.include_opa)
        self.output_dim = self.semantic_start + self.semantic_dim
        self.semantics_activation = semantics_activation
        self.xyz_activation = xyz_activation
        self.scale_activation = scale_activation
        self.scale_range = (scale_min, scale_max)
        self.scale_delta_factor = float(scale_delta_factor)
        self.rotation_delta_factor = float(rotation_delta_factor)
        self.opacity_delta_factor = float(opacity_delta_factor)
        self.semantic_delta_factor = float(semantic_delta_factor)

        self.register_buffer("pc_range", pc_range_tensor, persistent=False)
        self.register_buffer(
            "max_xyz_offset", max_xyz_offset_tensor, persistent=False
        )
        if grid_resolution is not None:
            self.register_buffer(
                "grid_resolution", grid_resolution, persistent=False
            )

        self.delta_mlp = nn.Sequential(
            *linear_relu_ln(self.embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

    def init_weight(self) -> None:
        """Use an identity refinement at initialization."""
        final_linear = self.delta_mlp[-2]
        for module in self.delta_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        with torch.no_grad():
            self.delta_mlp[-1].scale.fill_(1.0)

    def _decode_gaussian(
        self,
        anchor: torch.Tensor,
        original_xyz: torch.Tensor,
        delta_xyz: torch.Tensor,
    ) -> GaussianPrediction:
        xyz = self._get_xyz(anchor[..., :3])

        raw_scale = anchor[..., 3:6]
        if self.scale_activation == "sigmoid":
            scale_unit = safe_sigmoid(raw_scale)
        else:
            scale_unit = raw_scale.clamp(min=1e-6, max=1.0 - 1e-6)
        scales = self.scale_range[0] + (
            self.scale_range[1] - self.scale_range[0]
        ) * scale_unit

        semantic_logits = anchor[
            ..., self.semantic_start : self.semantic_start + self.semantic_dim
        ]
        if self.semantics_activation == "softmax":
            semantics = semantic_logits.softmax(dim=-1)
        elif self.semantics_activation == "softplus":
            semantics = F.softplus(semantic_logits)
        else:
            semantics = semantic_logits

        opacity_logits = anchor[..., 10 : self.semantic_start]
        opacities = (
            safe_sigmoid(opacity_logits)
            if self.include_opa
            else opacity_logits
        )

        return GaussianPrediction(
            means=xyz,
            scales=scales,
            rotations=anchor[..., 6:10],
            opacities=opacities,
            semantics=semantics,
            original_means=original_xyz,
            delta_means=delta_xyz,
        )

    def _get_xyz(self, anchor_xyz: torch.Tensor) -> torch.Tensor:
        return cartesian(
            anchor_xyz,
            pc_range=self.pc_range,
            use_sigmoid=(self.xyz_activation == "sigmoid"),
        )

    def _reverse_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        return reverse_cartesian(
            xyz,
            pc_range=self.pc_range,
            use_sigmoid=(self.xyz_activation == "sigmoid"),
        )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
    ):
        if anchor.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected anchor dim {self.output_dim}, got {anchor.shape[-1]}"
            )

        delta = self.delta_mlp(instance_feature + anchor_embed)

        original_xyz = self._get_xyz(anchor[..., :3])
        delta_xyz = torch.tanh(delta[..., :3]) * self.max_xyz_offset
        refined_xyz = self._reverse_xyz(original_xyz + delta_xyz)

        refined_scale = anchor[..., 3:6] + self.scale_delta_factor * torch.tanh(
            delta[..., 3:6]
        )
        refined_rotation = F.normalize(
            anchor[..., 6:10]
            + self.rotation_delta_factor * delta[..., 6:10],
            p=2,
            dim=-1,
        )

        cursor = 10
        parts = [refined_xyz, refined_scale, refined_rotation]
        if self.include_opa:
            refined_opacity = anchor[..., cursor : cursor + 1] + (
                self.opacity_delta_factor * delta[..., cursor : cursor + 1]
            )
            parts.append(refined_opacity)
            cursor += 1

        if self.semantic_dim:
            refined_semantics = anchor[..., cursor:] + (
                self.semantic_delta_factor * delta[..., cursor:]
            )
            parts.append(refined_semantics)

        refined_anchor = torch.cat(parts, dim=-1)
        gaussian = self._decode_gaussian(
            refined_anchor, original_xyz=original_xyz, delta_xyz=delta_xyz
        )
        return refined_anchor, gaussian
