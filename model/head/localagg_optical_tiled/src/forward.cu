// Optical-density G2V using S2GO's 4x4x4 tile-blocked forward layout.
//
// PRECONDITION (caller's responsibility): pts and points_int must be laid out in
// dense voxel-grid scanline order. Specifically pts[i*3..] is the world-space
// position of the voxel at coords (vx, vy, vz) where
//      i = vx * W * D + vy * D + vz
// This holds when the caller builds pts via a 3-axis meshgrid in (H, W, D) order.
// (The canonical S2GO/SurroundOcc dataloaders satisfy this.)
//
// PARALLELISM:
//   gridDim  = (Tx, Ty, Tz) = ceil((H,W,D)/TILE_DIM)
//   blockDim = (BLOCK_SIZE,) = 64 threads, one thread = one voxel in the tile

#include "forward.h"
#include "auxiliary.h"
#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Count how many tiles each Gaussian's AABB overlaps (per-Gaussian, 1 thread each).
__global__ void preprocessCUDA(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 tile_grid,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	tiles_touched[idx] = 0;

	uint3 rect_min, rect_max;
	getTileRect(points_xyz + 3 * idx, radii + 3 * idx, rect_min, rect_max, tile_grid);
	const uint nx = rect_max.x - rect_min.x;
	const uint ny = rect_max.y - rect_min.y;
	const uint nz = rect_max.z - rect_min.z;
	if (nx * ny * nz == 0)
		return;

	tiles_touched[idx] = nx * ny * nz;
}


// Tile-blocked render: 1 block per tile, 64 threads per block (one per voxel in tile).
template <uint32_t CHANNELS>
__global__ void renderCUDA(
	const dim3 tile_grid,                                // (Tx, Ty, Tz)
	const dim3 voxel_grid,                               // (H,  W,  D)
	const float* __restrict__ pts,                       // (M, 3) voxel world positions, scanline order
	const uint2* __restrict__ ranges,                    // (Tx*Ty*Tz,) per-tile [start, end] in point_list
	const uint32_t* __restrict__ point_list,             // sorted Gaussian IDs per tile
	const float* __restrict__ means3D,                   // (P, 3)
	const int*   __restrict__ means3D_int,               // (P, 3) Gaussian centers in voxel coords
	const int*   __restrict__ radii,                     // (P, 3) per-axis radius in voxel units
	const float* __restrict__ cov3D,                     // (P, 6)
	const float* __restrict__ opas,                      // (P,)
	const float* __restrict__ semantic,                  // (P, C)
	float* __restrict__ out_logits,                      // (M, C)
	float* __restrict__ out_bin_logits,                  // (M,)
	float* __restrict__ out_density)                     // (M,)
{
	// ----- 1. identify this block's tile & this thread's voxel -----
	const uint tx = blockIdx.x;
	const uint ty = blockIdx.y;
	const uint tz = blockIdx.z;
	const uint tile_id = tx * tile_grid.y * tile_grid.z + ty * tile_grid.z + tz;

	const uint t = threadIdx.x;                          // 0..63
	const uint vx_in = t / (TILE_DIM * TILE_DIM);
	const uint vy_in = (t / TILE_DIM) % TILE_DIM;
	const uint vz_in = t % TILE_DIM;
	const uint vx = tx * TILE_DIM + vx_in;
	const uint vy = ty * TILE_DIM + vy_in;
	const uint vz = tz * TILE_DIM + vz_in;
	const bool valid = (vx < voxel_grid.x) && (vy < voxel_grid.y) && (vz < voxel_grid.z);

	// scanline-ordered index into pts/output arrays
	const uint pts_idx = vx * voxel_grid.y * voxel_grid.z + vy * voxel_grid.z + vz;

	float3 voxel_pos;
	float C[CHANNELS];
	#pragma unroll
	for (int c = 0; c < CHANNELS; ++c) C[c] = 0.0f;
	float density = 0.0f;

	if (valid) {
		voxel_pos = make_float3(pts[3 * pts_idx + 0],
		                        pts[3 * pts_idx + 1],
		                        pts[3 * pts_idx + 2]);
	}

	// ----- 2. tile's Gaussian range -----
	const uint2 range = ranges[tile_id];
	const int n_g = (int)range.y - (int)range.x;

	// ----- 3. shared memory for cooperative chunk loads -----
	__shared__ float3 chunk_means    [CHUNK_SIZE];
	__shared__ int    chunk_means_int[CHUNK_SIZE * 3];     // for voxel-AABB check
	__shared__ int    chunk_radii    [CHUNK_SIZE * 3];     // per-axis radii
	__shared__ float  chunk_cov      [CHUNK_SIZE * 6];
	__shared__ float  chunk_opa      [CHUNK_SIZE];
	__shared__ float  chunk_sem      [CHUNK_SIZE * CHANNELS];

	// ----- 4. iterate the range in CHUNK_SIZE-Gaussian chunks -----
	for (int chunk_off = 0; chunk_off < n_g; chunk_off += (int)CHUNK_SIZE) {
		const int n_in_chunk = min((int)CHUNK_SIZE, n_g - chunk_off);

		// Cooperative load: thread t loads Gaussian (chunk_off + t)
		if ((int)t < n_in_chunk) {
			const int gs_idx = (int)point_list[range.x + chunk_off + (int)t];
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

		// Each thread (=voxel) accumulates this chunk's optical density.
		if (valid) {
			for (int g = 0; g < n_in_chunk; ++g) {
				// Voxel-AABB check: skip Gaussians whose AABB doesn't include this voxel.
				// This restores exact equivalence with the per-voxel oracle (Phase 3).
				const int mxi = chunk_means_int[g * 3 + 0];
				const int myi = chunk_means_int[g * 3 + 1];
				const int mzi = chunk_means_int[g * 3 + 2];
				const int rx  = chunk_radii    [g * 3 + 0];
				const int ry  = chunk_radii    [g * 3 + 1];
				const int rz  = chunk_radii    [g * 3 + 2];
				if ((int)vx < mxi - rx || (int)vx > mxi + rx ||
				    (int)vy < myi - ry || (int)vy > myi + ry ||
				    (int)vz < mzi - rz || (int)vz > mzi + rz) {
					continue;
				}

				const float3 m = chunk_means[g];
				const float  c1x = chunk_cov[g * 6 + 0];
				const float  c1y = chunk_cov[g * 6 + 1];
				const float  c1z = chunk_cov[g * 6 + 2];
				const float  c2x = chunk_cov[g * 6 + 3];
				const float  c2y = chunk_cov[g * 6 + 4];
				const float  c2z = chunk_cov[g * 6 + 5];
				const float3 d = make_float3(m.x - voxel_pos.x,
				                             m.y - voxel_pos.y,
				                             m.z - voxel_pos.z);
				float power = c1x * d.x * d.x + c1y * d.y * d.y + c1z * d.z * d.z;
				power = -0.5f * power - (c2x * d.x * d.y + c2y * d.y * d.z + c2z * d.x * d.z);
				power = expf(power);
				// Convert alpha to an additive optical strength.  At a Gaussian's
				// center, 1-exp(-rho) exactly equals alpha.
				const float alpha = fminf(fmaxf(chunk_opa[g], 0.0f), 1.0f - 1e-6f);
				const float rho = -log1pf(-alpha);
				const float weight = rho * power;
				#pragma unroll
				for (int c = 0; c < CHANNELS; ++c)
					C[c] += chunk_sem[g * CHANNELS + c] * weight;
				density += weight;
			}
		}
		__syncthreads();                                     // before next chunk's load
	}

	// ----- 5. write outputs -----
	if (valid) {
		if (density > 1e-9f) {
			#pragma unroll
			for (int c = 0; c < CHANNELS; ++c)
				out_logits[pts_idx * CHANNELS + c] = C[c] / density;
		} else {
			// Preserve the original zero-density fallback. Explicit-empty
			// training masks these voxels out of the semantic loss and uses
			// occupancy=0 to force the final label to empty.
			#pragma unroll
			for (int c = 0; c < CHANNELS - 1; ++c)
				out_logits[pts_idx * CHANNELS + c] =
					1.0f / (float)(CHANNELS - 1);
			out_logits[pts_idx * CHANNELS + CHANNELS - 1] = 0.0f;
		}
		out_bin_logits [pts_idx] = -expm1f(-density);
		out_density    [pts_idx] = density;
	}
}


void FORWARD::render(
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
	float* out_density)
{
	const dim3 grid(tile_grid.x, tile_grid.y, tile_grid.z);
	const dim3 block(BLOCK_SIZE, 1, 1);
	renderCUDA<NUM_CHANNELS> <<< grid, block >>> (
		tile_grid, voxel_grid,
		pts, ranges, point_list,
		means3D, means3D_int, radii, cov3D, opas, semantic,
		out_logits, out_bin_logits, out_density);
}


void FORWARD::preprocess(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 tile_grid,
	uint32_t* tiles_touched)
{
	preprocessCUDA <<< (P + 255) / 256, 256 >>> (
		P, points_xyz, radii, tile_grid, tiles_touched);
}
