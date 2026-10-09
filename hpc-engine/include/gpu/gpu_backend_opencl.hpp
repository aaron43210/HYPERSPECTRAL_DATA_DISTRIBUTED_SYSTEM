#pragma once
// =============================================================================
// gpu_backend_opencl.hpp — Cross-vendor GPU backend (OpenCL)
//
// OpenCL is the single portable API that targets every GPU this project cares
// about through CL_DEVICE_TYPE_GPU:
//   * Apple  (Apple Silicon / AMD on macOS, via the OpenCL.framework)
//   * NVIDIA (CUDA driver's OpenCL ICD)
//   * Intel  (Iris / UHD integrated GPUs, via the Intel OpenCL runtime)
//   * AMD    (ROCm / Adrenalin OpenCL ICD)
// so one backend covers all four vendors — there is no need for separate
// CUDA/Metal/oneAPI code paths.
//
// The spectral-index math here MUST match the CPU reference in
// SpectralFeatures.hpp exactly (same formulas, same 10000 reflectance scaling,
// same "0.0 == nodata" convention, same [-1,1] clamp) so results are identical
// regardless of which backend the engine chooses.
// =============================================================================
#include <string>
#include "RasterReader.hpp"   // HyperspectralTile, BandConfig, SpectralFeatureSet

namespace hsi {
namespace gpu {

// Index codes — MUST match the fixed engine order used by SpectralFeatureSet
// and TileWriter (NDVI, NDWI, NDBI, SAVI, EVI, NDMI, MSAVI).
enum IndexCode {
    IDX_NDVI  = 0,
    IDX_NDWI  = 1,
    IDX_NDBI  = 2,
    IDX_SAVI  = 3,
    IDX_EVI   = 4,
    IDX_NDMI  = 5,
    IDX_MSAVI = 6,
};

// A chunk of band data for one tile. Pointers are to host memory of length w*h
// (row-major). A null pointer means "band unavailable" and is treated as all
// zeros (i.e. nodata) by the kernel, so any index needing it yields 0.0.
struct RasterChunk {
    int w = 0;
    int h = 0;
    const float* red   = nullptr;
    const float* nir   = nullptr;
    const float* green = nullptr;
    const float* swir  = nullptr;
    const float* blue  = nullptr;   // required by EVI
    float scale = 10000.0f;         // reflectance scale (EnMAP L2A). Must match CPU.
};

// Initialize the OpenCL context/queue/kernel once, choosing the first available
// GPU device across all platforms. Returns false (no throw) if no GPU/OpenCL is
// present, so the caller can fall back to the CPU path.
bool init_opencl();

// Release all OpenCL objects. Safe to call if init was never run / already torn down.
void teardown_opencl();

// Compute ONE index for a chunk into out_index (pre-allocated to w*h floats).
// Returns false on failure. NOT thread-safe on its own — serialize calls, or
// use compute_spectral_features_gpu() which locks internally.
bool compute_index_opencl(const RasterChunk& chunk, float* out_index, int index_code);

// High-level bridge: fill a SpectralFeatureSet (all 7 indices) for a tile using
// the GPU, matching SpectralFeatures::compute(). Looks up the nir/red/green/
// swir/blue bands by their 1-based GDAL index from `cfg`. Thread-safe (locks a
// shared mutex internally, so it may be called from the OpenMP tile loop).
// Returns false if the GPU path failed for any index (caller should fall back
// to the CPU compute()); on success `out.computed` is set true.
bool compute_spectral_features_gpu(const HyperspectralTile& tile,
                                   const BandConfig& cfg,
                                   SpectralFeatureSet& out);

} // namespace gpu
} // namespace hsi
