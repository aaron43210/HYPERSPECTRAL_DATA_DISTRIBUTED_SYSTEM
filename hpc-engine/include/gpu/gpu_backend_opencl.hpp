#pragma once
#include <string>
#include "RasterReader.hpp"  // resolved via the hpc_core `include/` search path

namespace hsi {
namespace gpu {

// Represents a 2D tile chunk in memory
struct RasterChunk {
    int w = 0;
    int h = 0;
    float* red = nullptr;
    float* nir = nullptr;
    float* green = nullptr;
    float* swir = nullptr;
    float* blue = nullptr;
};

// Initializes the OpenCL context globally. Should be called ONCE before processing starts.
bool init_opencl();

// Tears down the OpenCL context.
void teardown_opencl();

// Submits a chunk of raster data to the GPU to calculate spectral indices.
// 'out_index' should be pre-allocated to w * h floats.
// index_code: 0=NDVI, 1=NDWI, 2=NDBI, 3=SAVI, 4=EVI, 5=NDMI, 6=MSAVI, 7=NORMALIZE
bool compute_index_opencl(const RasterChunk& chunk, float* out_index, int index_code);

// The exact wrapper function that main.cpp expects
bool compute_spectral_features_gpu(const HyperspectralTile& tile, const BandConfig& config, SpectralFeatureSet& sf);

} // namespace gpu
} // namespace hsi
