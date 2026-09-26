from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmseg.registry import MODELS

from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid

'''
    lifter=dict(
        type="GridGaussianLifter",
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        embed_dims=128,
        scene_size=[25, 25, 2],
    
        semantics=True,
        semantic_dim=18,
        empty_label=17,
        empty_semantic_init=0.6931471805599453,
        include_opa=True,
        init_opacity=0.01,
        overlap_ratio=1.25,
    
        anchor_grad=True,
        feat_grad=True,
    )
'''


Number = Union[int, float]


@MODELS.register_module()
class GridGaussianLifter(BaseLifter):
    """Initialize Gaussian candidates on a regular 3D grid.

    ``scene_size`` is the number of Gaussian cells along x/y/z.  It is not
    the final occupancy voxel grid.  Gaussian centers always cover the full
    ``pc_range`` and are placed at the centers of these coarse cells.

    Anchor layout:
        xyz (3) + scale (3) + quaternion (4) + opacity (0/1) + semantics (C)

    Args:
        pc_range: Physical range ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
        embed_dims: Dimension of each Gaussian instance feature.
        scene_size: Number of coarse Gaussian cells along x/y/z.  If None,
            it is derived from ``reso``.
        reso: Optional physical stride of the Gaussian grid.  When both
            ``scene_size`` and ``reso`` are given, they must describe the
            same grid.  The final occupancy voxel size (for example 0.5 m)
            should not be passed here unless the Gaussian grid is equally
            dense.
        scale_range: Physical min/max Gaussian standard deviation.
            When omitted, it is derived from the coarse grid resolution.
        scale_min_ratio: Automatic minimum scale divided by the smallest
            coarse-grid stride.
        scale_max_ratio: Automatic maximum scale divided by the largest
            coarse-grid stride.  Defaults to ``overlap_ratio``, leaving the
            initial scale near the middle rather than at the sigmoid limit.
        overlap_ratio: Initial scale divided by half of the grid stride.
        init_opacity: Initial occupancy strength of every Gaussian candidate.
            A low value is important because grid anchors are candidates, not
            pre-declared occupied cells.
        empty_label: Optional semantic index representing empty space.  When
            provided, that logit is initialized with ``empty_semantic_init``.
        empty_semantic_init: Initial raw logit for ``empty_label``.  The
            S2GO-style value ``log(2)`` gives empty a weak prior while leaving
            every grid Gaussian free to refine into a non-empty class.
    """

    def __init__(
        self,
        pc_range: Sequence[Number],
        embed_dims: int,
        scene_size: Optional[Sequence[int]] = (25, 25, 2),
        reso: Optional[Union[Number, Sequence[Number]]] = None,
        scale_range: Optional[Sequence[Number]] = None,
        scale_min_ratio: float = 0.01,
        scale_max_ratio: Optional[float] = None,
        anchor_grad: bool = True,
        feat_grad: bool = True,
        semantics: bool = True,
        semantic_dim: Optional[int] = 17,
        empty_label: Optional[int] = None,
        # empty_semantic_init: float = 0.0,
        overlap_ratio: float = 1.25,
        include_opa: bool = True,
        init_opacity: float = 0.01,
        xyz_activation: str = "sigmoid",
        scale_activation: str = "sigmoid",
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if len(pc_range) != 6:
            raise ValueError(f"pc_range must contain 6 values, got {pc_range}")
        if scale_range is not None and len(scale_range) != 2:
            raise ValueError("scale_range must be [scale_min, scale_max]")
        if xyz_activation not in ("sigmoid", "identity"):
            raise ValueError(
                f"Unsupported xyz_activation={xyz_activation!r}; "
                "expected 'sigmoid' or 'identity'"
            )
        if scale_activation not in ("sigmoid", "identity"):
            raise ValueError(
                f"Unsupported scale_activation={scale_activation!r}; "
                "expected 'sigmoid' or 'identity'"
            )
        if embed_dims <= 0:
            raise ValueError(f"embed_dims must be positive, got {embed_dims}")
        if overlap_ratio <= 0:
            raise ValueError(
                f"overlap_ratio must be positive, got {overlap_ratio}"
            )
        if scale_min_ratio <= 0:
            raise ValueError(
                f"scale_min_ratio must be positive, got {scale_min_ratio}"
            )
        if scale_max_ratio is not None and scale_max_ratio <= 0:
            raise ValueError(
                f"scale_max_ratio must be positive, got {scale_max_ratio}"
            )
        if include_opa and not 0.0 < init_opacity < 1.0:
            raise ValueError(
                f"init_opacity must be in (0, 1), got {init_opacity}"
            )

        pc_range_tensor = torch.as_tensor(pc_range, dtype=torch.float32)
        pc_min = pc_range_tensor[:3]
        pc_max = pc_range_tensor[3:]
        pc_extents = pc_max - pc_min
        if torch.any(pc_extents <= 0):
            raise ValueError(
                "Each pc_range maximum must be larger than its minimum, "
                f"got {pc_range}"
            )

        grid_shape, grid_resolution = self._resolve_grid(
            pc_extents=pc_extents,
            scene_size=scene_size,
            reso=reso,
        )

        if scale_range is None:
            resolved_scale_max_ratio = (
                float(overlap_ratio)
                if scale_max_ratio is None
                else float(scale_max_ratio)
            )
            scale_min = float(grid_resolution.min()) * float(scale_min_ratio)
            scale_max = float(grid_resolution.max()) * resolved_scale_max_ratio
        else:
            scale_min = float(scale_range[0])
            scale_max = float(scale_range[1])
            resolved_scale_max_ratio = None
        if scale_min <= 0 or scale_max <= scale_min:
            raise ValueError(
                "scale_range must satisfy 0 < scale_min < scale_max, "
                f"got {[scale_min, scale_max]}"
            )

        target_scale = 0.5 * grid_resolution * float(overlap_ratio)
        if torch.any(target_scale < scale_min) or torch.any(target_scale > scale_max):
            raise ValueError(
                "The initial physical scale must lie inside scale_range. "
                f"grid_resolution={grid_resolution.tolist()}, "
                f"overlap_ratio={overlap_ratio}, "
                f"target_scale={target_scale.tolist()}, "
                f"scale_range={[scale_min, scale_max]}"
            )

        if semantics and (semantic_dim is None or semantic_dim <= 0):
            raise ValueError(
                f"semantic_dim must be positive when semantics=True, got {semantic_dim}"
            )
        if empty_label is not None:
            if not semantics:
                raise ValueError(
                    "empty_label requires semantics=True"
                )
            if not 0 <= int(empty_label) < int(semantic_dim):
                raise ValueError(
                    "empty_label must index the semantic vector, got "
                    f"empty_label={empty_label}, semantic_dim={semantic_dim}"
                )

        self.embed_dims = int(embed_dims)
        self.scene_size = grid_shape
        self.num_anchor = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.include_opa = bool(include_opa)
        self.semantic_dim = int(semantic_dim) if semantics else 0
        self.empty_label = (
            int(empty_label) if empty_label is not None else None
        )
        # self.empty_semantic_init = float(empty_semantic_init)
        self.init_opacity = float(init_opacity)
        self.scale_range = (scale_min, scale_max)
        self.scale_min_ratio = float(scale_min_ratio)
        self.scale_max_ratio = resolved_scale_max_ratio

        self.register_buffer("pc_range", pc_range_tensor, persistent=False)
        self.register_buffer(
            "grid_resolution", grid_resolution.clone(), persistent=False
        )
        self.register_buffer(
            "initial_scale", target_scale.clone(), persistent=False
        )

        xyz = self.get_meshgrid(pc_range_tensor, grid_shape, grid_resolution)
        xyz_unit = (xyz - pc_min) / pc_extents
        xyz_anchor = (
            safe_inverse_sigmoid(xyz_unit)
            if xyz_activation == "sigmoid"
            else xyz_unit
        )

        scale_unit = (target_scale - scale_min) / (scale_max - scale_min)
        scale_unit = scale_unit.unsqueeze(0).expand(self.num_anchor, -1).clone()
        scale_anchor = (
            safe_inverse_sigmoid(scale_unit)
            if scale_activation == "sigmoid"
            else scale_unit
        )

        rotations = torch.zeros(self.num_anchor, 4, dtype=torch.float32)
        rotations[:, 0] = 1.0

        if include_opa:
            opacity_prob = torch.full(
                (self.num_anchor, 1), self.init_opacity, dtype=torch.float32
            )
            opacity = safe_inverse_sigmoid(opacity_prob)
        else:
            opacity = torch.empty(self.num_anchor, 0, dtype=torch.float32)

        # These are raw semantic logits.  With an explicit empty class, a weak
        # positive empty logit makes every coarse grid cell an empty candidate
        # instead of pre-declaring it as one of the foreground classes.
        semantic = torch.zeros(
            self.num_anchor, self.semantic_dim, dtype=torch.float32
        )
        if self.empty_label is not None:
            semantic[:, self.empty_label] = torch.log(torch.tensor(2.0))

        anchor = torch.cat(
            [xyz_anchor, scale_anchor, rotations, opacity, semantic], dim=-1
        )
        self.anchor = nn.Parameter(anchor.clone(), requires_grad=anchor_grad)
        self.instance_feature = nn.Parameter(
            torch.zeros(self.num_anchor, self.embed_dims, dtype=torch.float32),
            requires_grad=feat_grad,
        )
        self.register_buffer("_anchor_init", anchor.clone(), persistent=False)

    @property
    def derived_grid_params(self):
        """Geometry values that should be shared with refinement modules."""
        return {
            "scene_size": list(self.scene_size),
            "grid_resolution": self.grid_resolution.tolist(),
            "initial_scale": self.initial_scale.tolist(),
            "scale_range": list(self.scale_range),
            "max_xyz_offset": (0.5 * self.grid_resolution).tolist(),
        }

    @staticmethod
    def _to_xyz_tensor(
        value: Union[Number, Sequence[Number]], name: str
    ) -> torch.Tensor:
        if isinstance(value, (int, float)):
            result = torch.full((3,), float(value), dtype=torch.float32)
        else:
            if len(value) != 3:
                raise ValueError(f"{name} must be a scalar or 3 values, got {value}")
            result = torch.as_tensor(value, dtype=torch.float32)
        if torch.any(result <= 0):
            raise ValueError(f"{name} must be positive, got {value}")
        return result

    @classmethod
    def _resolve_grid(
        cls,
        pc_extents: torch.Tensor,
        scene_size: Optional[Sequence[int]],
        reso: Optional[Union[Number, Sequence[Number]]],
    ) -> Tuple[Tuple[int, int, int], torch.Tensor]:
        if scene_size is None and reso is None:
            raise ValueError("Either scene_size or reso must be provided")

        if scene_size is not None:
            if len(scene_size) != 3:
                raise ValueError(
                    f"scene_size must contain 3 integers, got {scene_size}"
                )
            grid_shape = tuple(int(v) for v in scene_size)
            if any(v <= 0 for v in grid_shape):
                raise ValueError(f"scene_size must be positive, got {scene_size}")
            if any(float(v) != float(int(v)) for v in scene_size):
                raise ValueError(f"scene_size must contain integers, got {scene_size}")

            grid_resolution = pc_extents / torch.tensor(
                grid_shape, dtype=torch.float32
            )
            if reso is not None:
                requested_resolution = cls._to_xyz_tensor(reso, "reso")
                if not torch.allclose(
                    requested_resolution,
                    grid_resolution,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "scene_size and reso describe different Gaussian grids: "
                        f"scene_size={grid_shape} gives "
                        f"{grid_resolution.tolist()} m, but reso="
                        f"{requested_resolution.tolist()} m. The final voxel "
                        "size must be configured in G2V, not in this lifter."
                    )
            return grid_shape, grid_resolution

        requested_resolution = cls._to_xyz_tensor(reso, "reso")
        grid_shape_float = pc_extents / requested_resolution
        grid_shape_tensor = grid_shape_float.round().to(torch.long)
        if not torch.allclose(
            grid_shape_float,
            grid_shape_tensor.to(torch.float32),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                "pc_range extents must be exactly divisible by reso, got "
                f"extents={pc_extents.tolist()}, reso="
                f"{requested_resolution.tolist()}"
            )
        grid_shape = tuple(int(v) for v in grid_shape_tensor.tolist())
        return grid_shape, requested_resolution

    @staticmethod
    def get_meshgrid(
        ranges: Sequence[Number],
        grid: Sequence[int],
        reso: Union[Number, Sequence[Number], torch.Tensor],
    ) -> torch.Tensor:
        """Return physical centers of all coarse Gaussian grid cells."""
        ranges_tensor = torch.as_tensor(ranges, dtype=torch.float32)
        if ranges_tensor.numel() != 6:
            raise ValueError(f"ranges must contain 6 values, got {ranges}")
        resolution = GridGaussianLifter._to_xyz_tensor(reso, "reso")
        nx, ny, nz = (int(v) for v in grid)
        pc_min = ranges_tensor[:3]

        x = pc_min[0] + (torch.arange(nx, dtype=torch.float32) + 0.5) * resolution[0]
        y = pc_min[1] + (torch.arange(ny, dtype=torch.float32) + 0.5) * resolution[1]
        z = pc_min[2] + (torch.arange(nz, dtype=torch.float32) + 0.5) * resolution[2]
        gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    def init_weights(self) -> None:
        with torch.no_grad():
            self.anchor.copy_(self._anchor_init)
            self.instance_feature.zero_()
            if self.instance_feature.requires_grad:
                nn.init.xavier_uniform_(self.instance_feature, gain=1.0)

    def forward(self, ms_img_feats, metas=None, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        representation = (
            self.anchor.unsqueeze(0).expand(batch_size, -1, -1).clone()
        )
        rep_features = (
            self.instance_feature.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )

        return {
            "rep_features": rep_features,
            "representation": representation,
            "anchor_init": self.anchor.clone(),
        }



# from typing import Optional, Sequence, Tuple, Union

# import torch
# import torch.nn as nn
# from mmseg.registry import MODELS

# from .base_lifter import BaseLifter
# from ..utils.safe_ops import safe_inverse_sigmoid

# '''
#     lifter=dict(
#         type="GridGaussianLifter",
#         pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
#         embed_dims=128,
#         scene_size=[25, 25, 2],
    
#         semantics=True,
#         semantic_dim=17,
#         include_opa=True,
#         init_opacity=0.01,
#         overlap_ratio=1.25,
    
#         anchor_grad=True,
#         feat_grad=True,
#     )
# '''


# Number = Union[int, float]


# @MODELS.register_module()
# class GridGaussianLifter(BaseLifter):
#     """Initialize Gaussian candidates on a regular 3D grid.

#     ``scene_size`` is the number of Gaussian cells along x/y/z.  It is not
#     the final occupancy voxel grid.  Gaussian centers always cover the full
#     ``pc_range`` and are placed at the centers of these coarse cells.

#     Anchor layout:
#         xyz (3) + scale (3) + quaternion (4) + opacity (0/1) + semantics (C)

#     Args:
#         pc_range: Physical range ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
#         embed_dims: Dimension of each Gaussian instance feature.
#         scene_size: Number of coarse Gaussian cells along x/y/z.  If None,
#             it is derived from ``reso``.
#         reso: Optional physical stride of the Gaussian grid.  When both
#             ``scene_size`` and ``reso`` are given, they must describe the
#             same grid.  The final occupancy voxel size (for example 0.5 m)
#             should not be passed here unless the Gaussian grid is equally
#             dense.
#         scale_range: Physical min/max Gaussian standard deviation.
#             When omitted, it is derived from the coarse grid resolution.
#         scale_min_ratio: Automatic minimum scale divided by the smallest
#             coarse-grid stride.
#         scale_max_ratio: Automatic maximum scale divided by the largest
#             coarse-grid stride.  Defaults to ``overlap_ratio``, leaving the
#             initial scale near the middle rather than at the sigmoid limit.
#         overlap_ratio: Initial scale divided by half of the grid stride.
#         init_opacity: Initial occupancy strength of every Gaussian candidate.
#             A low value is important because grid anchors are candidates, not
#             pre-declared occupied cells.
#     """

#     def __init__(
#         self,
#         pc_range: Sequence[Number],
#         embed_dims: int,
#         scene_size: Optional[Sequence[int]] = (25, 25, 2),
#         reso: Optional[Union[Number, Sequence[Number]]] = None,
#         scale_range: Optional[Sequence[Number]] = None,
#         scale_min_ratio: float = 0.01,
#         scale_max_ratio: Optional[float] = None,
#         anchor_grad: bool = True,
#         feat_grad: bool = True,
#         semantics: bool = True,
#         semantic_dim: Optional[int] = 17,
#         overlap_ratio: float = 1.25,
#         include_opa: bool = True,
#         init_opacity: float = 0.01,
#         xyz_activation: str = "sigmoid",
#         scale_activation: str = "sigmoid",
#         init_cfg=None,
#         **kwargs,
#     ) -> None:
#         super().__init__(init_cfg=init_cfg)

#         if len(pc_range) != 6:
#             raise ValueError(f"pc_range must contain 6 values, got {pc_range}")
#         if scale_range is not None and len(scale_range) != 2:
#             raise ValueError("scale_range must be [scale_min, scale_max]")
#         if xyz_activation not in ("sigmoid", "identity"):
#             raise ValueError(
#                 f"Unsupported xyz_activation={xyz_activation!r}; "
#                 "expected 'sigmoid' or 'identity'"
#             )
#         if scale_activation not in ("sigmoid", "identity"):
#             raise ValueError(
#                 f"Unsupported scale_activation={scale_activation!r}; "
#                 "expected 'sigmoid' or 'identity'"
#             )
#         if embed_dims <= 0:
#             raise ValueError(f"embed_dims must be positive, got {embed_dims}")
#         if overlap_ratio <= 0:
#             raise ValueError(
#                 f"overlap_ratio must be positive, got {overlap_ratio}"
#             )
#         if scale_min_ratio <= 0:
#             raise ValueError(
#                 f"scale_min_ratio must be positive, got {scale_min_ratio}"
#             )
#         if scale_max_ratio is not None and scale_max_ratio <= 0:
#             raise ValueError(
#                 f"scale_max_ratio must be positive, got {scale_max_ratio}"
#             )
#         if include_opa and not 0.0 < init_opacity < 1.0:
#             raise ValueError(
#                 f"init_opacity must be in (0, 1), got {init_opacity}"
#             )

#         pc_range_tensor = torch.as_tensor(pc_range, dtype=torch.float32)
#         pc_min = pc_range_tensor[:3]
#         pc_max = pc_range_tensor[3:]
#         pc_extents = pc_max - pc_min
#         if torch.any(pc_extents <= 0):
#             raise ValueError(
#                 "Each pc_range maximum must be larger than its minimum, "
#                 f"got {pc_range}"
#             )

#         grid_shape, grid_resolution = self._resolve_grid(
#             pc_extents=pc_extents,
#             scene_size=scene_size,
#             reso=reso,
#         )

#         if scale_range is None:
#             resolved_scale_max_ratio = (
#                 float(overlap_ratio)
#                 if scale_max_ratio is None
#                 else float(scale_max_ratio)
#             )
#             scale_min = float(grid_resolution.min()) * float(scale_min_ratio)
#             scale_max = float(grid_resolution.max()) * resolved_scale_max_ratio
#         else:
#             scale_min = float(scale_range[0])
#             scale_max = float(scale_range[1])
#             resolved_scale_max_ratio = None
#         if scale_min <= 0 or scale_max <= scale_min:
#             raise ValueError(
#                 "scale_range must satisfy 0 < scale_min < scale_max, "
#                 f"got {[scale_min, scale_max]}"
#             )

#         target_scale = 0.5 * grid_resolution * float(overlap_ratio)
#         if torch.any(target_scale < scale_min) or torch.any(target_scale > scale_max):
#             raise ValueError(
#                 "The initial physical scale must lie inside scale_range. "
#                 f"grid_resolution={grid_resolution.tolist()}, "
#                 f"overlap_ratio={overlap_ratio}, "
#                 f"target_scale={target_scale.tolist()}, "
#                 f"scale_range={[scale_min, scale_max]}"
#             )

#         if semantics and (semantic_dim is None or semantic_dim <= 0):
#             raise ValueError(
#                 f"semantic_dim must be positive when semantics=True, got {semantic_dim}"
#             )

#         self.embed_dims = int(embed_dims)
#         self.scene_size = grid_shape
#         self.num_anchor = grid_shape[0] * grid_shape[1] * grid_shape[2]
#         self.xyz_act = xyz_activation
#         self.scale_act = scale_activation
#         self.include_opa = bool(include_opa)
#         self.semantic_dim = int(semantic_dim) if semantics else 0
#         self.init_opacity = float(init_opacity)
#         self.scale_range = (scale_min, scale_max)
#         self.scale_min_ratio = float(scale_min_ratio)
#         self.scale_max_ratio = resolved_scale_max_ratio

#         self.register_buffer("pc_range", pc_range_tensor, persistent=False)
#         self.register_buffer(
#             "grid_resolution", grid_resolution.clone(), persistent=False
#         )
#         self.register_buffer(
#             "initial_scale", target_scale.clone(), persistent=False
#         )

#         xyz = self.get_meshgrid(pc_range_tensor, grid_shape, grid_resolution)
#         xyz_unit = (xyz - pc_min) / pc_extents
#         xyz_anchor = (
#             safe_inverse_sigmoid(xyz_unit)
#             if xyz_activation == "sigmoid"
#             else xyz_unit
#         )

#         scale_unit = (target_scale - scale_min) / (scale_max - scale_min)
#         scale_unit = scale_unit.unsqueeze(0).expand(self.num_anchor, -1).clone()
#         scale_anchor = (
#             safe_inverse_sigmoid(scale_unit)
#             if scale_activation == "sigmoid"
#             else scale_unit
#         )

#         rotations = torch.zeros(self.num_anchor, 4, dtype=torch.float32)
#         rotations[:, 0] = 1.0

#         if include_opa:
#             opacity_prob = torch.full(
#                 (self.num_anchor, 1), self.init_opacity, dtype=torch.float32
#             )
#             opacity = safe_inverse_sigmoid(opacity_prob)
#         else:
#             opacity = torch.empty(self.num_anchor, 0, dtype=torch.float32)

#         # Zero logits mean a uniform conditional distribution over occupied
#         # classes without injecting an arbitrary spatial semantic prior.
#         semantic = torch.zeros(
#             self.num_anchor, self.semantic_dim, dtype=torch.float32
#         )

#         anchor = torch.cat(
#             [xyz_anchor, scale_anchor, rotations, opacity, semantic], dim=-1
#         )
#         self.anchor = nn.Parameter(anchor.clone(), requires_grad=anchor_grad)
#         self.instance_feature = nn.Parameter(
#             torch.zeros(self.num_anchor, self.embed_dims, dtype=torch.float32),
#             requires_grad=feat_grad,
#         )
#         self.register_buffer("_anchor_init", anchor.clone(), persistent=False)

#     @property
#     def derived_grid_params(self):
#         """Geometry values that should be shared with refinement modules."""
#         return {
#             "scene_size": list(self.scene_size),
#             "grid_resolution": self.grid_resolution.tolist(),
#             "initial_scale": self.initial_scale.tolist(),
#             "scale_range": list(self.scale_range),
#             "max_xyz_offset": (0.5 * self.grid_resolution).tolist(),
#         }

#     @staticmethod
#     def _to_xyz_tensor(
#         value: Union[Number, Sequence[Number]], name: str
#     ) -> torch.Tensor:
#         if isinstance(value, (int, float)):
#             result = torch.full((3,), float(value), dtype=torch.float32)
#         else:
#             if len(value) != 3:
#                 raise ValueError(f"{name} must be a scalar or 3 values, got {value}")
#             result = torch.as_tensor(value, dtype=torch.float32)
#         if torch.any(result <= 0):
#             raise ValueError(f"{name} must be positive, got {value}")
#         return result

#     @classmethod
#     def _resolve_grid(
#         cls,
#         pc_extents: torch.Tensor,
#         scene_size: Optional[Sequence[int]],
#         reso: Optional[Union[Number, Sequence[Number]]],
#     ) -> Tuple[Tuple[int, int, int], torch.Tensor]:
#         if scene_size is None and reso is None:
#             raise ValueError("Either scene_size or reso must be provided")

#         if scene_size is not None:
#             if len(scene_size) != 3:
#                 raise ValueError(
#                     f"scene_size must contain 3 integers, got {scene_size}"
#                 )
#             grid_shape = tuple(int(v) for v in scene_size)
#             if any(v <= 0 for v in grid_shape):
#                 raise ValueError(f"scene_size must be positive, got {scene_size}")
#             if any(float(v) != float(int(v)) for v in scene_size):
#                 raise ValueError(f"scene_size must contain integers, got {scene_size}")

#             grid_resolution = pc_extents / torch.tensor(
#                 grid_shape, dtype=torch.float32
#             )
#             if reso is not None:
#                 requested_resolution = cls._to_xyz_tensor(reso, "reso")
#                 if not torch.allclose(
#                     requested_resolution,
#                     grid_resolution,
#                     rtol=1e-5,
#                     atol=1e-6,
#                 ):
#                     raise ValueError(
#                         "scene_size and reso describe different Gaussian grids: "
#                         f"scene_size={grid_shape} gives "
#                         f"{grid_resolution.tolist()} m, but reso="
#                         f"{requested_resolution.tolist()} m. The final voxel "
#                         "size must be configured in G2V, not in this lifter."
#                     )
#             return grid_shape, grid_resolution

#         requested_resolution = cls._to_xyz_tensor(reso, "reso")
#         grid_shape_float = pc_extents / requested_resolution
#         grid_shape_tensor = grid_shape_float.round().to(torch.long)
#         if not torch.allclose(
#             grid_shape_float,
#             grid_shape_tensor.to(torch.float32),
#             rtol=1e-5,
#             atol=1e-5,
#         ):
#             raise ValueError(
#                 "pc_range extents must be exactly divisible by reso, got "
#                 f"extents={pc_extents.tolist()}, reso="
#                 f"{requested_resolution.tolist()}"
#             )
#         grid_shape = tuple(int(v) for v in grid_shape_tensor.tolist())
#         return grid_shape, requested_resolution

#     @staticmethod
#     def get_meshgrid(
#         ranges: Sequence[Number],
#         grid: Sequence[int],
#         reso: Union[Number, Sequence[Number], torch.Tensor],
#     ) -> torch.Tensor:
#         """Return physical centers of all coarse Gaussian grid cells."""
#         ranges_tensor = torch.as_tensor(ranges, dtype=torch.float32)
#         if ranges_tensor.numel() != 6:
#             raise ValueError(f"ranges must contain 6 values, got {ranges}")
#         resolution = GridGaussianLifter._to_xyz_tensor(reso, "reso")
#         nx, ny, nz = (int(v) for v in grid)
#         pc_min = ranges_tensor[:3]

#         x = pc_min[0] + (torch.arange(nx, dtype=torch.float32) + 0.5) * resolution[0]
#         y = pc_min[1] + (torch.arange(ny, dtype=torch.float32) + 0.5) * resolution[1]
#         z = pc_min[2] + (torch.arange(nz, dtype=torch.float32) + 0.5) * resolution[2]
#         gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
#         return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

#     def init_weights(self) -> None:
#         with torch.no_grad():
#             self.anchor.copy_(self._anchor_init)
#             self.instance_feature.zero_()
#             if self.instance_feature.requires_grad:
#                 nn.init.xavier_uniform_(self.instance_feature, gain=1.0)

#     def forward(self, ms_img_feats, metas=None, **kwargs):
#         batch_size = ms_img_feats[0].shape[0]
#         representation = (
#             self.anchor.unsqueeze(0).expand(batch_size, -1, -1).clone()
#         )
#         rep_features = (
#             self.instance_feature.unsqueeze(0)
#             .expand(batch_size, -1, -1)
#             .clone()
#         )

#         return {
#             "rep_features": rep_features,
#             "representation": representation,
#             "anchor_init": self.anchor.clone(),
#         }
