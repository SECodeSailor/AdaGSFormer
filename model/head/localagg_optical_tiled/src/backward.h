/*
 * Phase 4e: tile-blocked backward kernel.
 *
 * Same parallelism pattern as forward (block per tile, BLOCK_SIZE threads).
 * After cooperative load, thread `t` handles Gaussian `t` of the chunk:
 *   scans all voxels in the tile, accumulates that Gaussian's gradient
 *   contributions in per-thread registers, then atomicAdds once at the end
 *   per gradient component. Cross-tile contention on the same Gaussian is
 *   acceptable (one tile's worth, ~tens of writes per Gaussian total).
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace BACKWARD
{
	// pts must be in scanline order (same precondition as FORWARD::render).
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
		// forward outputs (needed for grad calc)
		const float* logits,
		const float* bin_logits,
		const float* density,
		// gradients flowing in
		const float* logits_grad,
		const float* bin_logits_grad,
		const float* density_grad,
		// outputs (gradients to compute)
		float* means3D_grad,
		float* opas_grad,
		float* semantics_grad,
		float* cov3D_grad);
}

#endif
