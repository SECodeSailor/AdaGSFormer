/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * S2GO Phase 4c modifications: tile-blocked forward render signature.
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace FORWARD
{
	// Per-Gaussian preprocess: count overlapped tiles in tile_grid coords.
	void preprocess(
		const int P,
		const int* points_xyz,
		const int* radii,
		const dim3 tile_grid,
		uint32_t* tiles_touched);

	// Tile-blocked render. tile_grid = (Tx,Ty,Tz); voxel_grid = (H,W,D).
	// pts must be in dense voxel-grid scanline order (see forward.cu comment).
	// means3D_int + radii enable an inner-loop voxel-AABB check so the kernel
	// evaluates exactly the same (voxel, Gaussian) pair set as the per-voxel oracle.
	void render(
		const dim3 tile_grid,
		const dim3 voxel_grid,
		const float* pts,
		const uint2* ranges,
		const uint32_t* point_list,
		const float* means3D,
		const int* means3D_int,
		const int* radii,
		const float* cov3D,
		const float* opas,
		const float* semantic,
		float* out_logits,
		float* out_bin_logits,
		float* out_density);
}

#endif
