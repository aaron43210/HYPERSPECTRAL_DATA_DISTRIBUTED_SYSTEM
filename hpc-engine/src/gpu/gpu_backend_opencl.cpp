// =============================================================================
// gpu_backend_opencl.cpp — OpenCL implementation of the cross-vendor GPU backend
// =============================================================================
#include "gpu/gpu_backend_opencl.hpp"

// Pin the OpenCL API to 1.2 BEFORE including the headers. 1.2 is the lowest
// common denominator supported by every target (Apple, NVIDIA, Intel, AMD) and
// keeps clCreateCommandQueue (removed from the default surface in 2.0+ headers)
// available and non-deprecated.
#define CL_TARGET_OPENCL_VERSION 120

#include <algorithm>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

namespace hsi {
namespace gpu {

// -----------------------------------------------------------------------------
// Kernel. Computes ONE spectral index per work-item. The math is a line-for-line
// port of SpectralFeatures.hpp: reflectance is de-scaled by `scale` (10000 for
// EnMAP L2A), an index is produced ONLY where all of its operand bands are
// non-zero (0.0 == nodata), the result is clamped to [-1, 1], and nodata pixels
// are written as 0.0 (NOT NaN) to match the engine's nodata convention.
// -----------------------------------------------------------------------------
static const char* KERNEL_SRC = R"CLC(
__kernel void calc_index(__global const float* red,
                         __global const float* nir,
                         __global const float* green,
                         __global const float* swir,
                         __global const float* blue,
                         __global float*       out,
                         const int             n,
                         const int             index_code,
                         const float           scale)
{
    const int i = get_global_id(0);
    if (i >= n) return;

    // Raw (pre-scale) values; raw == 0 marks nodata / masked bands.
    const float rr = red[i], rv = nir[i], rg = green[i], rs = swir[i], rb = blue[i];
    const float inv = 1.0f / scale;
    const float r = rr * inv, v = rv * inv, g = rg * inv, s = rs * inv, b = rb * inv;
    const float L = 0.5f;

    float res = 0.0f; // nodata default

    if (index_code == 0) {               // NDVI
        if (rv != 0.0f && rr != 0.0f) {
            float d = v + r;
            if (d != 0.0f) res = clamp((v - r) / d, -1.0f, 1.0f);
        }
    } else if (index_code == 1) {        // NDWI
        if (rg != 0.0f && rv != 0.0f) {
            float d = g + v;
            if (d != 0.0f) res = clamp((g - v) / d, -1.0f, 1.0f);
        }
    } else if (index_code == 2) {        // NDBI
        if (rs != 0.0f && rv != 0.0f) {
            float d = s + v;
            if (d != 0.0f) res = clamp((s - v) / d, -1.0f, 1.0f);
        }
    } else if (index_code == 3) {        // SAVI
        if (rv != 0.0f && rr != 0.0f) {
            float d = v + r + L;
            if (d != 0.0f) res = clamp(((v - r) / d) * (1.0f + L), -1.0f, 1.0f);
        }
    } else if (index_code == 4) {        // EVI
        if (rv != 0.0f && rr != 0.0f && rb != 0.0f) {
            float d = v + 6.0f * r - 7.5f * b + 1.0f;
            if (d != 0.0f) res = clamp(2.5f * (v - r) / d, -1.0f, 1.0f);
        }
    } else if (index_code == 5) {        // NDMI
        if (rs != 0.0f && rv != 0.0f) {
            float d = s + v;
            if (d != 0.0f) res = clamp((v - s) / d, -1.0f, 1.0f);
        }
    } else if (index_code == 6) {        // MSAVI
        if (rv != 0.0f && rr != 0.0f) {
            float term = 2.0f * v + 1.0f;
            float disc = term * term - 8.0f * (v - r);
            if (disc >= 0.0f) res = clamp((term - sqrt(disc)) * 0.5f, -1.0f, 1.0f);
        }
    }
    out[i] = res;
}
)CLC";

#define HSI_CL_CHECK(expr)                                                     \
    do {                                                                       \
        cl_int _rc = (expr);                                                   \
        if (_rc != CL_SUCCESS) {                                               \
            throw std::runtime_error(                                          \
                std::string("OpenCL error ") + std::to_string(_rc) +           \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));           \
        }                                                                      \
    } while (0)

// Global OpenCL state. The context/queue/kernel are shared, so GPU dispatch is
// serialized by g_dispatch_mutex (see compute_spectral_features_gpu).
static cl_context       g_ctx    = nullptr;
static cl_command_queue g_queue  = nullptr;
static cl_program       g_prog   = nullptr;
static cl_kernel        g_kernel = nullptr;
static size_t           g_local_size = 256;
static std::mutex       g_dispatch_mutex;

bool init_opencl() {
    if (g_ctx) return true; // Already initialized

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0)
        return false;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_device_id device = nullptr;

    // Prefer the first GPU device across all platforms (Apple/NVIDIA/Intel/AMD).
    for (auto p : platforms) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd > 0) {
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) == CL_SUCCESS)
                break;
            device = nullptr;
        }
    }
    if (!device) return false; // no GPU → caller falls back to CPU

    char name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);

    size_t max_wg = 256;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
    g_local_size = std::min<size_t>(256, max_wg ? max_wg : 1);

    cl_int err = CL_SUCCESS;
    g_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !g_ctx) { g_ctx = nullptr; return false; }

    g_queue = clCreateCommandQueue(g_ctx, device, 0, &err);
    if (err != CL_SUCCESS || !g_queue) { teardown_opencl(); return false; }

    g_prog = clCreateProgramWithSource(g_ctx, 1, &KERNEL_SRC, nullptr, &err);
    if (err != CL_SUCCESS || !g_prog) { teardown_opencl(); return false; }

    err = clBuildProgram(g_prog, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(g_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0)
            clGetProgramBuildInfo(g_prog, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        std::cerr << "[opencl] Kernel build failed:\n" << log << "\n";
        teardown_opencl();
        return false;
    }

    g_kernel = clCreateKernel(g_prog, "calc_index", &err);
    if (err != CL_SUCCESS || !g_kernel) { teardown_opencl(); return false; }

    std::cout << "[opencl] GPU backend ready on device: " << name << "\n";
    return true;
}

void teardown_opencl() {
    if (g_kernel) { clReleaseKernel(g_kernel);        g_kernel = nullptr; }
    if (g_prog)   { clReleaseProgram(g_prog);         g_prog   = nullptr; }
    if (g_queue)  { clReleaseCommandQueue(g_queue);   g_queue  = nullptr; }
    if (g_ctx)    { clReleaseContext(g_ctx);          g_ctx    = nullptr; }
}

bool compute_index_opencl(const RasterChunk& chunk, float* out_index, int index_code) {
    if (!g_ctx || !g_kernel || !g_queue) return false;

    const size_t n = static_cast<size_t>(chunk.w) * static_cast<size_t>(chunk.h);
    if (n == 0) return true;
    const size_t bytes = n * sizeof(float);

    cl_int err = CL_SUCCESS;
    // Upload a band, or a zero-filled buffer when the band is unavailable (so a
    // missing operand reads as nodata and the index collapses to 0.0).
    auto ensure_buf = [&](const float* ptr) -> cl_mem {
        if (ptr) {
            cl_mem m = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      bytes, const_cast<float*>(ptr), &err);
            HSI_CL_CHECK(err);
            return m;
        }
        cl_mem m = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        HSI_CL_CHECK(err);
        std::vector<float> zeros(n, 0.0f);
        HSI_CL_CHECK(clEnqueueWriteBuffer(g_queue, m, CL_TRUE, 0, bytes, zeros.data(),
                                          0, nullptr, nullptr));
        return m;
    };

    cl_mem buf_red   = ensure_buf(chunk.red);
    cl_mem buf_nir   = ensure_buf(chunk.nir);
    cl_mem buf_green = ensure_buf(chunk.green);
    cl_mem buf_swir  = ensure_buf(chunk.swir);
    cl_mem buf_blue  = ensure_buf(chunk.blue);
    cl_mem buf_out   = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    HSI_CL_CHECK(err);

    const int   n_int = static_cast<int>(n);
    const float scale = (chunk.scale != 0.0f) ? chunk.scale : 1.0f;
    cl_uint a = 0;
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_mem), &buf_red));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_mem), &buf_nir));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_mem), &buf_green));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_mem), &buf_swir));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_mem), &buf_blue));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_mem), &buf_out));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_int), &n_int));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_int), &index_code));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, a++, sizeof(cl_float), &scale));

    const size_t global_size = ((n + g_local_size - 1) / g_local_size) * g_local_size;
    HSI_CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr,
                                        &global_size, &g_local_size, 0, nullptr, nullptr));
    HSI_CL_CHECK(clEnqueueReadBuffer(g_queue, buf_out, CL_TRUE, 0, bytes, out_index,
                                     0, nullptr, nullptr));

    clReleaseMemObject(buf_red);
    clReleaseMemObject(buf_nir);
    clReleaseMemObject(buf_green);
    clReleaseMemObject(buf_swir);
    clReleaseMemObject(buf_blue);
    clReleaseMemObject(buf_out);
    return true;
}

// -----------------------------------------------------------------------------
// High-level bridge: fill all 7 indices for a tile, matching the CPU reference.
// -----------------------------------------------------------------------------
bool compute_spectral_features_gpu(const HyperspectralTile& tile,
                                   const BandConfig& cfg,
                                   SpectralFeatureSet& out) {
    if (!g_ctx) return false;
    if (tile.bands.empty()) { out.computed = false; return true; }

    const size_t numPixels = tile.bands[0].values.size();

    // Resolve band pointers by 1-based GDAL index, exactly like the CPU path.
    const float* nir = nullptr; const float* red  = nullptr;
    const float* grn = nullptr; const float* swir = nullptr; const float* blue = nullptr;
    for (const auto& bnd : tile.bands) {
        if (bnd.index == cfg.nir)   nir  = bnd.values.data();
        if (bnd.index == cfg.red)   red  = bnd.values.data();
        if (bnd.index == cfg.green) grn  = bnd.values.data();
        if (bnd.index == cfg.swir)  swir = bnd.values.data();
        if (bnd.index == cfg.blue)  blue = bnd.values.data();
    }

    RasterChunk chunk;
    chunk.w = static_cast<int>(numPixels);   // dispatched as a 1-D range of w*h
    chunk.h = 1;
    chunk.red = red; chunk.nir = nir; chunk.green = grn; chunk.swir = swir; chunk.blue = blue;
    chunk.scale = 10000.0f;

    out.ndvi.assign(numPixels, 0.0f);
    out.ndwi.assign(numPixels, 0.0f);
    out.ndbi.assign(numPixels, 0.0f);
    out.savi.assign(numPixels, 0.0f);
    out.evi.assign(numPixels, 0.0f);
    out.ndmi.assign(numPixels, 0.0f);
    out.msavi.assign(numPixels, 0.0f);

    // Serialize access to the shared kernel/queue so this is safe to call from
    // the engine's OpenMP tile loop.
    std::lock_guard<std::mutex> lock(g_dispatch_mutex);
    try {
        if (!compute_index_opencl(chunk, out.ndvi.data(),  IDX_NDVI))  return false;
        if (!compute_index_opencl(chunk, out.ndwi.data(),  IDX_NDWI))  return false;
        if (!compute_index_opencl(chunk, out.ndbi.data(),  IDX_NDBI))  return false;
        if (!compute_index_opencl(chunk, out.savi.data(),  IDX_SAVI))  return false;
        if (!compute_index_opencl(chunk, out.evi.data(),   IDX_EVI))   return false;
        if (!compute_index_opencl(chunk, out.ndmi.data(),  IDX_NDMI))  return false;
        if (!compute_index_opencl(chunk, out.msavi.data(), IDX_MSAVI)) return false;
    } catch (const std::exception& e) {
        std::cerr << "[opencl] dispatch failed, falling back to CPU: " << e.what() << "\n";
        return false;
    }

    out.computed = true;
    return true;
}

} // namespace gpu
} // namespace hsi
