/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 18 // Default 3, RGB

// S2GO-derived tile-blocked execution layout for optical-density G2V.
//   TILE_DIM^3 voxels per tile; 1 CUDA block per tile; 1 thread per voxel-in-tile.
//   Cooperative shared-mem load of CHUNK_SIZE Gaussians at a time.
#define TILE_DIM    4
#define BLOCK_SIZE  (TILE_DIM * TILE_DIM * TILE_DIM)   // 64
#define CHUNK_SIZE  BLOCK_SIZE                          // 64 Gaussians per shared chunk

#endif
