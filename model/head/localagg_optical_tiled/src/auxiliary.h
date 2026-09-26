#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"


// Voxel-coord rect (used by the per-voxel renderCUDA in the original GF-2 design).
__forceinline__ __device__ void getRect(const int* p, const int* radius, uint3& rect_min, uint3& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)(p[0] - radius[0]))),
		min(grid.y, max((int)0, (int)(p[1] - radius[1]))),
        min(grid.z, max((int)0, (int)(p[2] - radius[2])))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)(p[0] + radius[0] + 1))),
		min(grid.y, max((int)0, (int)(p[1] + radius[1] + 1))),
        min(grid.z, max((int)0, (int)(p[2] + radius[2] + 1)))
	};
}

// Tile-coordinate rectangle for the S2GO-derived optical tiled design. Inputs are
// voxel coords (as produced by the Python wrapper); output is the inclusive-
// exclusive range of TILE_DIM^3 tiles the Gaussian's AABB overlaps, clamped
// to the ceiling-divided tile grid.
__forceinline__ __device__ void getTileRect(const int* p, const int* radius, uint3& rect_min, uint3& rect_max, dim3 tile_grid)
{
	// floor( (p - r) / TILE_DIM ), clamp to [0, tile_grid)
	rect_min = {
		min(tile_grid.x, (uint32_t)max((int)0, (p[0] - radius[0]) / (int)TILE_DIM)),
		min(tile_grid.y, (uint32_t)max((int)0, (p[1] - radius[1]) / (int)TILE_DIM)),
		min(tile_grid.z, (uint32_t)max((int)0, (p[2] - radius[2]) / (int)TILE_DIM))
	};
	// floor( (p + r) / TILE_DIM ) + 1, clamp to [0, tile_grid]
	rect_max = {
		min(tile_grid.x, (uint32_t)max((int)0, (p[0] + radius[0]) / (int)TILE_DIM + 1)),
		min(tile_grid.y, (uint32_t)max((int)0, (p[1] + radius[1]) / (int)TILE_DIM + 1)),
		min(tile_grid.z, (uint32_t)max((int)0, (p[2] + radius[2]) / (int)TILE_DIM + 1))
	};
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
