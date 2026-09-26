// Optical-density G2V using S2GO's 4x4x4 tile-blocked backward layout.
//
// Block-per-tile, BLOCK_SIZE threads per block.
// Phase 1: cooperatively load per-voxel state into __shared__ (one thread per voxel).
// Phase 2: iterate tile's Gaussian range in CHUNK_SIZE chunks. For each chunk:
//   2a. cooperatively load Gaussian state into __shared__.
//   2b. each thread t handles Gaussian t of the chunk; scan all 64 tile-voxels;
//       accumulate this Gaussian's gradient in per-thread registers.
//   2c. atomicAdd once per gradient component to global memory.
//
// The S2GO parallel layout is retained; the derivatives implement optical-density G2V.

#include "backward.h"
#include "auxiliary.h"
#include <math.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


template <uint32_t CHANNELS>
__global__ void renderBackwardCUDA(
	const dim3 tile_grid,
	const dim3 voxel_grid,
	const float* __restrict__ pts,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ means3D,
	const int*   __restrict__ means3D_int,
	const int*   __restrict__ radii,
	const float* __restrict__ cov3D,
	const float* __restrict__ opas,
	const float* __restrict__ semantic,
	const float* __restrict__ logits,
	const float* __restrict__ bin_logits,
	const float* __restrict__ density,
	const float* __restrict__ logits_grad,
	const float* __restrict__ bin_logits_grad,
	const float* __restrict__ density_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ opas_grad,
	float* __restrict__ semantics_grad,
	float* __restrict__ cov3D_grad)
{
	// ----- identify tile + thread's voxel within tile -----
	const uint tx = blockIdx.x;
	const uint ty = blockIdx.y;
	const uint tz = blockIdx.z;
	const uint tile_id = tx * tile_grid.y * tile_grid.z + ty * tile_grid.z + tz;

	const uint t = threadIdx.x;
	const uint vx_in = t / (TILE_DIM * TILE_DIM);
	const uint vy_in = (t / TILE_DIM) % TILE_DIM;
	const uint vz_in = t % TILE_DIM;
	const uint vx = tx * TILE_DIM + vx_in;
	const uint vy = ty * TILE_DIM + vy_in;
	const uint vz = tz * TILE_DIM + vz_in;
	const bool valid = (vx < voxel_grid.x) && (vy < voxel_grid.y) && (vz < voxel_grid.z);
	const uint pts_idx = vx * voxel_grid.y * voxel_grid.z + vy * voxel_grid.z + vz;

	// ----- Phase 1: cooperative load of per-voxel state into shared mem -----
	// Each thread loads ONE voxel's state (its own).
	__shared__ float3 voxel_pos_sh        [BLOCK_SIZE];
	__shared__ uint3  voxel_coords_sh     [BLOCK_SIZE];     // (vx, vy, vz) for AABB check
	__shared__ bool   voxel_valid_sh      [BLOCK_SIZE];
	__shared__ float  density_sum_sh      [BLOCK_SIZE];
	__shared__ float  occupancy_sh        [BLOCK_SIZE];
	__shared__ float  bin_logit_grad_sh   [BLOCK_SIZE];
	__shared__ float  density_grad_sh     [BLOCK_SIZE];
	__shared__ float  logits_sh           [BLOCK_SIZE * CHANNELS];
	__shared__ float  logits_grad_sh      [BLOCK_SIZE * CHANNELS];

	voxel_valid_sh[t] = valid;
	voxel_coords_sh[t] = make_uint3(vx, vy, vz);
	if (valid) {
		voxel_pos_sh[t]      = make_float3(pts[3 * pts_idx + 0],
		                                   pts[3 * pts_idx + 1],
		                                   pts[3 * pts_idx + 2]);
		density_sum_sh[t]    = density[pts_idx];
		occupancy_sh[t]      = bin_logits [pts_idx];
		bin_logit_grad_sh[t] = bin_logits_grad[pts_idx];
		density_grad_sh[t]   = density_grad[pts_idx];
		#pragma unroll
		for (int c = 0; c < CHANNELS; ++c) {
			logits_sh     [t * CHANNELS + c] = logits     [pts_idx * CHANNELS + c];
			logits_grad_sh[t * CHANNELS + c] = logits_grad[pts_idx * CHANNELS + c];
		}
	}
	__syncthreads();

	// ----- Phase 2: iterate tile's Gaussian range -----
	const uint2 range = ranges[tile_id];
	const int n_g = (int)range.y - (int)range.x;

	__shared__ float3 chunk_means    [CHUNK_SIZE];
	__shared__ int    chunk_means_int[CHUNK_SIZE * 3];
	__shared__ int    chunk_radii    [CHUNK_SIZE * 3];
	__shared__ float  chunk_cov      [CHUNK_SIZE * 6];
	__shared__ float  chunk_opa      [CHUNK_SIZE];
	__shared__ float  chunk_sem      [CHUNK_SIZE * CHANNELS];
	__shared__ uint   chunk_gauss_id [CHUNK_SIZE];

	for (int chunk_off = 0; chunk_off < n_g; chunk_off += (int)CHUNK_SIZE) {
		const int n_in_chunk = min((int)CHUNK_SIZE, n_g - chunk_off);

		// 2a. cooperative Gaussian load (1 Gaussian per thread)
		if ((int)t < n_in_chunk) {
			const uint gs_idx = (uint)point_list[range.x + chunk_off + (int)t];
			chunk_gauss_id[t] = gs_idx;
			chunk_means[t] = make_float3(means3D[3 * gs_idx + 0],
			                             means3D[3 * gs_idx + 1],
			                             means3D[3 * gs_idx + 2]);
			#pragma unroll
			for (int k = 0; k < 3; ++k) {
				chunk_means_int[t * 3 + k] = means3D_int[3 * gs_idx + k];
				chunk_radii    [t * 3 + k] = radii      [3 * gs_idx + k];
			}
			#pragma unroll
			for (int k = 0; k < 6; ++k)
				chunk_cov[t * 6 + k] = cov3D[6 * gs_idx + k];
			chunk_opa[t] = opas[gs_idx];
			#pragma unroll
			for (int c = 0; c < CHANNELS; ++c)
				chunk_sem[t * CHANNELS + c] = semantic[CHANNELS * gs_idx + c];
		}
		__syncthreads();

		// 2b. thread t handles Gaussian t in chunk: scan all tile voxels
		if ((int)t < n_in_chunk) {
			const uint gs_idx = chunk_gauss_id[t];
			const float3 mean = chunk_means[t];
			const int mxi = chunk_means_int[t * 3 + 0];
			const int myi = chunk_means_int[t * 3 + 1];
			const int mzi = chunk_means_int[t * 3 + 2];
			const int rx  = chunk_radii    [t * 3 + 0];
			const int ry  = chunk_radii    [t * 3 + 1];
			const int rz  = chunk_radii    [t * 3 + 2];
			const float c1x = chunk_cov[t * 6 + 0];
			const float c1y = chunk_cov[t * 6 + 1];
			const float c1z = chunk_cov[t * 6 + 2];
			const float c2x = chunk_cov[t * 6 + 3];
			const float c2y = chunk_cov[t * 6 + 4];
			const float c2z = chunk_cov[t * 6 + 5];
			const float raw_opa = chunk_opa[t];
			const float alpha = fminf(fmaxf(raw_opa, 0.0f), 1.0f - 1e-6f);
			const float one_minus_alpha = fmaxf(1.0f - alpha, 1e-6f);
			const float rho = -logf(one_minus_alpha);
			// Match torch.clamp outside the valid alpha interval. Normal sigmoid
			// opacities lie strictly inside this interval.
			const float alpha_grad_scale =
				(raw_opa >= 0.0f && raw_opa <= 1.0f - 1e-6f) ? 1.0f : 0.0f;
			float sem[CHANNELS];
			#pragma unroll
			for (int c = 0; c < CHANNELS; ++c) sem[c] = chunk_sem[t * CHANNELS + c];

			// Per-Gaussian gradient accumulators (registers)
			float means_grad   [3] = {0.0f, 0.0f, 0.0f};
			float cov_grad     [6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			float opa_grad     = 0.0f;
			float semantic_grad[CHANNELS];
			#pragma unroll
			for (int c = 0; c < CHANNELS; ++c) semantic_grad[c] = 0.0f;

			// Inner loop: scan all voxels in this tile
			for (int v = 0; v < (int)BLOCK_SIZE; ++v) {
				if (!voxel_valid_sh[v]) continue;

				// Voxel-AABB filter (matches Phase 3 oracle's per-voxel grouping).
				const uint3 vc = voxel_coords_sh[v];
				if ((int)vc.x < mxi - rx || (int)vc.x > mxi + rx ||
				    (int)vc.y < myi - ry || (int)vc.y > myi + ry ||
				    (int)vc.z < mzi - rz || (int)vc.z > mzi + rz) {
					continue;
				}

				const float3 voxel_pos = voxel_pos_sh[v];
				const float3 d = make_float3(mean.x - voxel_pos.x,
				                             mean.y - voxel_pos.y,
				                             mean.z - voxel_pos.z);
				float power = c1x * d.x * d.x + c1y * d.y * d.y + c1z * d.z * d.z;
				power = -0.5f * power - (c2x * d.x * d.y + c2y * d.y * d.z + c2z * d.x * d.z);
				power = expf(power);
				const float weight = rho * power;

				// dL/d(weight) combines conditional-semantics, occupancy and
				// exposed-density losses. The quotient derivative is
				// (sem[c] - output[c]) / D.
				float weight_grad = 0.0f;
				const float density_sum = density_sum_sh[v];

				if (density_sum > 1e-9f) {
					#pragma unroll
					for (int c = 0; c < CHANNELS; ++c) {
						const float lg = logits_grad_sh[v * CHANNELS + c];
						const float lc = logits_sh     [v * CHANNELS + c];
						semantic_grad[c] += lg * weight / density_sum;
						weight_grad += lg * (sem[c] - lc) / density_sum;
					}
				}

				weight_grad += (1.0f - occupancy_sh[v]) * bin_logit_grad_sh[v];
				weight_grad += density_grad_sh[v];

				const float power_grad = weight_grad * rho;
				opa_grad += weight_grad * power / one_minus_alpha * alpha_grad_scale;

				// Optical occupancy and opacity derivatives were accumulated above.

				means_grad[0] -= power_grad * power * (c1x * d.x + c2x * d.y + c2z * d.z);
				means_grad[1] -= power_grad * power * (c2x * d.x + c1y * d.y + c2y * d.z);
				means_grad[2] -= power_grad * power * (c2z * d.x + c2y * d.y + c1z * d.z);

				cov_grad[0] += power_grad * power * (-0.5f * d.x * d.x);
				cov_grad[1] += power_grad * power * (-0.5f * d.y * d.y);
				cov_grad[2] += power_grad * power * (-0.5f * d.z * d.z);
				cov_grad[3] += power_grad * power * (-d.x * d.y);
				cov_grad[4] += power_grad * power * (-d.y * d.z);
				cov_grad[5] += power_grad * power * (-d.x * d.z);
			}

			// 2c. atomicAdd Gaussian's gradient ONCE per component (cross-tile contention OK)
			atomicAdd(&means3D_grad[gs_idx * 3 + 0], means_grad[0]);
			atomicAdd(&means3D_grad[gs_idx * 3 + 1], means_grad[1]);
			atomicAdd(&means3D_grad[gs_idx * 3 + 2], means_grad[2]);
			atomicAdd(&opas_grad   [gs_idx],         opa_grad);
			#pragma unroll
			for (int c = 0; c < CHANNELS; ++c)
				atomicAdd(&semantics_grad[gs_idx * CHANNELS + c], semantic_grad[c]);
			#pragma unroll
			for (int k = 0; k < 6; ++k)
				atomicAdd(&cov3D_grad[gs_idx * 6 + k], cov_grad[k]);
		}
		__syncthreads();
	}
}


void BACKWARD::render(
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
	const float* logits,
	const float* bin_logits,
	const float* density,
	const float* logits_grad,
	const float* bin_logits_grad,
	const float* density_grad,
	float* means3D_grad,
	float* opas_grad,
	float* semantics_grad,
	float* cov3D_grad)
{
	const dim3 grid(tile_grid.x, tile_grid.y, tile_grid.z);
	const dim3 block(BLOCK_SIZE, 1, 1);
	renderBackwardCUDA<NUM_CHANNELS> <<< grid, block >>> (
		tile_grid, voxel_grid,
		pts, ranges, point_list,
		means3D, means3D_int, radii,
		cov3D, opas, semantic,
		logits, bin_logits, density,
		logits_grad, bin_logits_grad, density_grad,
		means3D_grad, opas_grad, semantics_grad, cov3D_grad);
}
