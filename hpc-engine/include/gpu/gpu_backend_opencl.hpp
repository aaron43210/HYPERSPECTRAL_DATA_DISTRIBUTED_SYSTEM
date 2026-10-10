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

// ---------------------------------------------------------------------------
// Whole-scene GPU operations (SAM, PCA). Each initialises its extra kernels
// lazily on first use (reusing the context/queue created by init_opencl) and
// returns false on ANY failure so the caller runs the documented CPU fallback
// for that tile. All kernels are single-precision float so they build on GPUs
// without cl_khr_fp64 (e.g. Apple Silicon); float64 accumulation happens on the
// host.
// ---------------------------------------------------------------------------

// SAM: one work-item per pixel, single pass. The tile's bands are uploaded as
// ONE packed band-major buffer (band b of pixel i at b*n + i), the reference in
// __constant memory, and ||ref|| as a scalar. Mirrors sam::computeTileCPU.
//   outAngle/outSim : resized to n; radians / cosine-similarity[0,1]; -1 nodata.
bool compute_sam_gpu(const HyperspectralTile& tile,
                     const std::vector<float>& ref, float refNorm,
                     std::vector<float>& outAngle, std::vector<float>& outSim);

// PCA covariance partial for ONE tile (map step). The kernel computes, for every
// (j,k) entry, Σ_valid (x_j-μ_j)(x_k-μ_k) over the tile's valid pixels using
// Kahan-compensated float accumulation; the Bk×Bk float partial is summed into
// the caller's double accumulator (host reduce step). keptBands are 1-based.
//   covPartialOut : resized to Bk*Bk (row-major), this tile's contribution.
bool compute_pca_cov_gpu(const HyperspectralTile& tile,
                         const std::vector<int>& keptBands,
                         const std::vector<float>& mean,
                         const std::vector<unsigned char>& valid,
                         std::vector<double>& covPartialOut);

// PCA projection for ONE tile: y_c = Σ_b Vk[b,c] * (x_b - μ_b) * invStd_b, one
// work-item per pixel. Vk is Bk×k column-major (Vk[c*Bk + b]). Nodata pixels get
// `nodataVal`. outComponents is resized to k vectors of n.
bool compute_pca_project_gpu(const HyperspectralTile& tile,
                             const std::vector<int>& keptBands,
                             const std::vector<float>& mean,
                             const std::vector<float>& invStd,
                             const std::vector<float>& Vk, int k,
                             const std::vector<unsigned char>& valid,
                             float nodataVal,
                             std::vector<std::vector<float>>& outComponents);

} // namespace gpu
} // namespace hsi
