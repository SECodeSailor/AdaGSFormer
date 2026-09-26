from typing import Tuple

import torch
import torch.nn as nn


def optical_density_g2v_reference(
    points: torch.Tensor,
    means: torch.Tensor,
    opacities: torch.Tensor,
    semantics: torch.Tensor,
    covariance_inverse: torch.Tensor,
    point_chunk_size: int = 4096,
    opacity_eps: float = 1e-6,
    density_eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable PyTorch reference for optical-density G2V.

    Args:
        points: Voxel centers with shape ``[B, V, 3]``.
        means: Gaussian centers with shape ``[B, N, 3]``.
        opacities: Post-sigmoid alpha values, ``[B, N]`` or ``[B, N, 1]``.
        semantics: Per-Gaussian semantic features, ``[B, N,C]``.  The legacy
            path passes conditional class probabilities; the explicit-empty
            path passes 18 raw class logits.  Optical G2V applies the same
            density-weighted mean to either representation.
        covariance_inverse: Inverse physical covariance, ``[B, N, 3, 3]``.

    Returns:
        Aggregated semantic features ``[B,V,C]``, occupancy ``[B,V]`` and
        optical density ``[B,V]``.

    This implementation chunks the voxel dimension to bound memory, but is
    still O(V*N).  It is intended as a correctness oracle and CPU fallback for
    small tests; full-grid training should use the CUDA implementation.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B, V, 3], got {points.shape}")
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError(f"means must be [B, N, 3], got {means.shape}")
    if opacities.ndim == 3 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    if opacities.shape != means.shape[:2]:
        raise ValueError(
            f"opacities must be [B, N], got {opacities.shape}"
        )
    if semantics.shape[:2] != means.shape[:2] or semantics.ndim != 3:
        raise ValueError(
            f"semantics must be [B, N, C], got {semantics.shape}"
        )
    if covariance_inverse.shape != (*means.shape[:2], 3, 3):
        raise ValueError(
            "covariance_inverse must be [B, N, 3, 3], got "
            f"{covariance_inverse.shape}"
        )
    if points.shape[0] != means.shape[0]:
        raise ValueError("points and Gaussians must have the same batch size")
    if means.shape[1] == 0:
        raise ValueError("G2V requires at least one Gaussian")
    if point_chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive")

    # rho=-log(1-alpha) is non-negative and exactly recovers alpha at G=1:
    # 1-exp(-rho) = alpha.
    alpha = opacities.float().clamp(min=0.0, max=1.0 - opacity_eps)
    optical_strength = -torch.log1p(-alpha)
    means_f = means.float()
    semantics_f = semantics.float()
    covariance_inverse_f = covariance_inverse.float()
    points_f = points.float()

    semantic_chunks = []
    occupancy_chunks = []
    density_chunks = []
    num_classes = semantics.shape[-1]

    for start in range(0, points.shape[1], point_chunk_size):
        end = min(start + point_chunk_size, points.shape[1])
        delta = means_f.unsqueeze(1) - points_f[:, start:end].unsqueeze(2)
        quadratic = torch.einsum(
            "bvni,bnij,bvnj->bvn",
            delta,
            covariance_inverse_f,
            delta,
        ).clamp_min(0.0)
        gaussian = torch.exp(-0.5 * quadratic)
        weights = optical_strength.unsqueeze(1) * gaussian
        density = weights.sum(dim=-1)
        semantic_numerator = torch.einsum(
            "bvn,bnc->bvc", weights, semantics_f
        )
        conditional = semantic_numerator / density.unsqueeze(-1).clamp_min(
            density_eps
        )

        # Preserve the established optical-G2V zero-density fallback.  In the
        # explicit-empty path these voxels are excluded from semantic loss and
        # occ_prob=0 forces the final prediction to empty.
        empty_density = density <= density_eps
        if num_classes > 0 and torch.any(empty_density):
            fallback = torch.zeros_like(conditional)
            if num_classes > 1:
                fallback[..., :-1] = 1.0 / float(num_classes - 1)
            conditional = torch.where(
                empty_density.unsqueeze(-1), fallback, conditional
            )

        occupancy = -torch.expm1(-density)
        semantic_chunks.append(conditional)
        occupancy_chunks.append(occupancy)
        density_chunks.append(density)

    output_dtype = means.dtype
    return (
        torch.cat(semantic_chunks, dim=1).to(output_dtype),
        torch.cat(occupancy_chunks, dim=1).to(output_dtype),
        torch.cat(density_chunks, dim=1).to(output_dtype),
    )


class OpticalDensityAggregatorReference(nn.Module):
    """LocalAggregator-compatible wrapper around the PyTorch reference."""

    def __init__(self, point_chunk_size: int = 4096, **kwargs) -> None:
        super().__init__()
        self.point_chunk_size = int(point_chunk_size)

    def forward(
        self,
        pts: torch.Tensor,
        means3D: torch.Tensor,
        opas: torch.Tensor,
        semantics: torch.Tensor,
        scales: torch.Tensor,
        cov3D: torch.Tensor,
    ):
        del scales
        conditional, occupancy, density = optical_density_g2v_reference(
            points=pts,
            means=means3D,
            opacities=opas,
            semantics=semantics,
            covariance_inverse=cov3D,
            point_chunk_size=self.point_chunk_size,
        )
        if conditional.shape[0] != 1:
            raise ValueError(
                "GaussianFormer G2V currently supports batch_size=1"
            )
        return conditional[0], occupancy[0], density[0]
