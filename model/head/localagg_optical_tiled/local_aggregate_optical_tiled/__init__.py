#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
from . import _C

AGGREGATION_MODE = getattr(_C, "aggregation_mode", "stale_or_unknown")


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opas,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke the independent optical-density C++/CUDA extension.
        (num_rendered, logits, bin_logits, density,
         geomBuffer, binningBuffer, imgBuffer) = _C.local_aggregate(*args)
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer,
            binningBuffer,
            imgBuffer,
            means3D,
            means3D_int,        # Phase 4e: needed for backward voxel-AABB filter
            radii,              # Phase 4e: needed for backward voxel-AABB filter
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density
        )
        return logits, bin_logits, density

    @staticmethod
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        (geomBuffer, binningBuffer, imgBuffer,
         means3D, means3D_int, radii,
         pts, points_int, cov3D, opas, semantics,
         logits, bin_logits, density) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            means3D_int,
            radii,
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density,
            logits_grad,
            bin_logits_grad,
            density_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opas_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opas_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1):
        super().__init__()
        if min(H, W, D) <= 0:
            raise ValueError(f"H, W and D must be positive, got {(H, W, D)}")
        if scale_multiplier <= 0:
            raise ValueError("scale_multiplier must be positive")
        if radii_min < 1:
            raise ValueError("radii_min must be at least 1")
        grid_size_tensor = torch.as_tensor(grid_size, dtype=torch.float).reshape(-1)
        if grid_size_tensor.numel() not in (1, 3) or torch.any(grid_size_tensor <= 0):
            raise ValueError("grid_size must be a positive scalar or length-3 value")
        pc_min_tensor = torch.as_tensor(pc_min, dtype=torch.float).reshape(-1)
        if pc_min_tensor.numel() != 3:
            raise ValueError("pc_min must contain exactly three coordinates")
        self.scale_multiplier = float(scale_multiplier)
        self.H = int(H)
        self.W = int(W)
        self.D = int(D)
        self.register_buffer('pc_min', pc_min_tensor.unsqueeze(0))
        self.register_buffer('grid_size', grid_size_tensor)
        self.radii_min = int(radii_min)

    def forward(
        self, 
        pts,
        means3D, 
        opas,
        semantics, 
        scales, 
        cov3D): 

        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(f"pts must be [1, H*W*D, 3], got {tuple(pts.shape)}")
        if means3D.ndim != 3 or means3D.shape[-1] != 3:
            raise ValueError(f"means3D must be [1, N, 3], got {tuple(means3D.shape)}")
        if opas.shape != means3D.shape[:2]:
            raise ValueError(f"opas must be [1, N], got {tuple(opas.shape)}")
        if semantics.ndim != 3 or semantics.shape[:2] != means3D.shape[:2]:
            raise ValueError(f"semantics must be [1, N, 18], got {tuple(semantics.shape)}")
        if semantics.shape[-1] != 18:
            raise ValueError(
                "This extension was compiled with NUM_CHANNELS=18, got "
                f"{semantics.shape[-1]} channels."
            )
        if scales.shape != means3D.shape:
            raise ValueError(f"scales must be [1, N, 3], got {tuple(scales.shape)}")
        if cov3D.shape != (*means3D.shape[:2], 3, 3):
            raise ValueError(f"cov3D must be [1, N, 3, 3], got {tuple(cov3D.shape)}")
        if pts.shape[0] != 1 or means3D.shape[0] != 1:
            raise ValueError("The tiled CUDA aggregator currently supports batch size 1.")
        if pts.shape[1] != self.H * self.W * self.D:
            raise ValueError(
                "The tiled CUDA aggregator requires a complete dense voxel grid "
                f"with H*W*D={self.H * self.W * self.D} points, got {pts.shape[1]}."
            )
        if means3D.shape[1] == 0:
            raise ValueError("At least one Gaussian is required by the CUDA binning kernel.")
        if not all(t.is_cuda for t in (pts, means3D, opas, semantics, scales, cov3D)):
            raise ValueError("All tiled-aggregator inputs must be CUDA tensors.")

        # The CUDA ABI is float32. Explicit casts also make autocast training
        # safe; autograd casts returned gradients back through these ops.
        pts = pts.float().squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.float().squeeze(0)
        opas = opas.float().squeeze(0)
        semantics = semantics.float().squeeze(0)
        scales = scales.detach().float().squeeze(0)
        cov3D = cov3D.float().squeeze(0)

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # Invoke C++/CUDA rasterization routine
        logits, bin_logits, density = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        return logits, bin_logits, density # n, c; n, c; n
