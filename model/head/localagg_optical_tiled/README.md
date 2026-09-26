# Tiled optical-density Gaussian-to-voxel CUDA operator

This is an independent CUDA extension. It does not replace or modify
`localagg_prob_fast`.

The execution layout is ported from S2GO's complete tiled implementation:

- `4 x 4 x 4` voxels per CUDA block (`64` threads);
- Gaussian-to-tile AABB binning followed by radix sorting;
- cooperative shared-memory loading in chunks of `64` Gaussians;
- a tile-blocked backward pass in which one thread accumulates one Gaussian's
  gradients over the tile's 64 voxels before issuing global `atomicAdd`s.

The aggregation math is optical density:

```text
rho_i       = -log(1 - alpha_i)
G_i(v)      = exp(-0.5 * (mu_i-v)^T Sigma_i^-1 (mu_i-v))
w_i(v)      = rho_i * G_i(v)
D(v)        = sum_i w_i(v)
occupancy   = 1 - exp(-D(v))
semantic_c  = sum_i w_i(v) * semantic_i,c / max(D(v), eps)
```

The operator is compiled for 18 semantic channels. Its point tensor must be a
complete dense voxel grid in scanline order
`index = x * W * D + y * D + z`. Grid dimensions do not have to be divisible
by four; edge threads are masked.

Build it inside the GaussianFormer CUDA environment:

```bash
cd model/head/localagg_optical_tiled
pip install -v .
```

Select it in `GaussianHead` with `use_optical_density=True` and
`optical_backend='cuda'`. Use `optical_backend='torch'` for the small,
differentiable correctness reference.
