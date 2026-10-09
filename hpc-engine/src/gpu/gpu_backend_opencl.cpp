#include "gpu/gpu_backend_opencl.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <mutex>

// Pin the OpenCL API version BEFORE the CL headers are pulled in. 1.2 is the
// common denominator across Apple, NVIDIA, Intel and AMD. Crucially, the
// Khronos headers used on Linux and Windows *remove* the 1.x entry points by
// default (or mark them deprecated) unless this macro is set, which would hide
// the clCreateCommandQueue call below. Apple's own <OpenCL/cl.h> ignores the
// macro, so defining it everywhere is safe and keeps the three platforms in
// sync. The guard lets the build system override it via -D if ever needed.
#ifndef CL_TARGET_OPENCL_VERSION
#  define CL_TARGET_OPENCL_VERSION 120
#endif

#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

namespace hsi {
namespace gpu {

static const char* KERNEL_SRC = R"CLC(
#define HSI_NAN (as_float(0x7fc00000u))

__kernel void calc_index(__global const float* red,
                         __global const float* nir,
                         __global const float* green,
                         __global const float* swir,
                         __global const float* blue,
                         __global float*       out,
                         int                   n,
                         int                   index_code)
{
    const int i = get_global_id(0);
    if (i >= n) return;

    // EnMAP L2A data is scaled by 10,000. Convert to 0.0 - 1.0 reflectance.
    const float r = red ? (red[i] / 10000.0f) : 0.0f;
    const float v = nir ? (nir[i] / 10000.0f) : 0.0f;
    const float g = green ? (green[i] / 10000.0f) : 0.0f;
    const float s = swir ? (swir[i] / 10000.0f) : 0.0f;
    const float b = blue ? (blue[i] / 10000.0f) : 0.0f;
    const float eps = 1e-6f;
    const float L = 0.5f;

    // A value of 0.0 marks nodata or masked out pixels.
    if (index_code == 0 || index_code == 3 || index_code == 6) { // NDVI, SAVI, MSAVI (needs nir, red)
        if (v == 0.0f || r == 0.0f) { out[i] = 0.0f; return; }
    } else if (index_code == 1) { // NDWI (needs green, nir)
        if (g == 0.0f || v == 0.0f) { out[i] = 0.0f; return; }
    } else if (index_code == 2 || index_code == 5) { // NDBI, NDMI (needs swir, nir)
        if (s == 0.0f || v == 0.0f) { out[i] = 0.0f; return; }
    } else if (index_code == 4) { // EVI (needs nir, red, blue)
        if (v == 0.0f || r == 0.0f || b == 0.0f) { out[i] = 0.0f; return; }
    }

    float result = 0.0f;
    if (index_code == 0) { // NDVI
        const float d = v + r;
        result = (fabs(d) < eps) ? HSI_NAN : (v - r) / d;
    } else if (index_code == 1) { // NDWI
        const float d = g + v;
        result = (fabs(d) < eps) ? HSI_NAN : (g - v) / d;
    } else if (index_code == 2) { // NDBI
        const float d = s + v;
        result = (fabs(d) < eps) ? HSI_NAN : (s - v) / d;
    } else if (index_code == 3) { // SAVI
        const float d = v + r + L;
        result = (fabs(d) < eps) ? HSI_NAN : ((v - r) / d) * (1.0f + L);
    } else if (index_code == 4) { // EVI
        const float d = v + 6.0f * r - 7.5f * b + 1.0f;
        result = (fabs(d) < eps) ? HSI_NAN : 2.5f * (v - r) / d;
    } else if (index_code == 5) { // NDMI
        const float d = v + s;
        result = (fabs(d) < eps) ? HSI_NAN : (v - s) / d;
    } else if (index_code == 6) { // MSAVI
        const float term = 2.0f * v + 1.0f;
        const float disc = term * term - 8.0f * (v - r);
        result = (disc >= 0.0f) ? (term - sqrt(disc)) / 2.0f : HSI_NAN;
    } else { // NORMALIZE
        result = fmin(fmax(r, 0.0f), 1.0f);
    }
    
    // Clamp to -1..1
    out[i] = fmax(-1.0f, fmin(1.0f, result));
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

static cl_context g_ctx = nullptr;
static cl_command_queue g_queue = nullptr;
static cl_program g_prog = nullptr;
static cl_kernel g_kernel = nullptr;
static size_t g_local_size = 256;

bool init_opencl() {
    if (g_ctx) return true;

    // Initialisation must never throw out of here: `--mode gpu` is documented to
    // fall back to the CPU transparently, and both call sites invoke this outside
    // any try/catch. A present-but-broken GPU/ICD (context/queue/program/kernel
    // creation failing) must therefore yield `false`, not abort the process.
    try {

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0)
        return false;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;

    for (auto p : platforms) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd > 0) {
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
            platform = p;
            break;
        }
    }

    if (!device) return false;

    char name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "[opencl] Initializing Device: " << name << "\n";

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(g_local_size), &g_local_size, nullptr);
    g_local_size = std::min<size_t>(256, g_local_size);

    cl_int err;
    g_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    HSI_CL_CHECK(err);

    g_queue = clCreateCommandQueue(g_ctx, device, 0, &err);
    HSI_CL_CHECK(err);

    g_prog = clCreateProgramWithSource(g_ctx, 1, &KERNEL_SRC, nullptr, &err);
    HSI_CL_CHECK(err);

    err = clBuildProgram(g_prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(g_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0)
            clGetProgramBuildInfo(g_prog, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        std::cerr << "[opencl] Build log:\n" << log << "\n";
        return false;
    }

    g_kernel = clCreateKernel(g_prog, "calc_index", &err);
    HSI_CL_CHECK(err);

    return true;

    } catch (const std::exception& e) {
        std::cerr << "[opencl] initialisation failed, using CPU: " << e.what() << "\n";
        teardown_opencl();   // release any partially-created objects
        return false;
    }
}

void teardown_opencl() {
    if (g_kernel) { clReleaseKernel(g_kernel); g_kernel = nullptr; }
    if (g_prog)   { clReleaseProgram(g_prog); g_prog = nullptr; }
    if (g_queue)  { clReleaseCommandQueue(g_queue); g_queue = nullptr; }
    if (g_ctx)    { clReleaseContext(g_ctx); g_ctx = nullptr; }
}

// Serialises GPU dispatch. The kernel and command queue are process-global
// singletons and clSetKernelArg on a shared cl_kernel is NOT thread-safe
// (OpenCL 1.2 §5.9). The engine calls compute_index_opencl from inside the
// OpenMP tile loop (--mode gpu), so without this lock concurrent tiles would
// race on the kernel's argument state and corrupt the index bands.
static std::mutex g_dispatch_mutex;

namespace {
// Minimal RAII guard so a mid-dispatch failure (or thrown HSI_CL_CHECK) releases
// every GPU buffer during stack unwinding instead of leaking it in the
// long-lived context across repeated failing tiles.
struct ClMem {
    cl_mem m = nullptr;
    ClMem() = default;
    explicit ClMem(cl_mem h) : m(h) {}
    ~ClMem() { if (m) clReleaseMemObject(m); }
    ClMem(const ClMem&) = delete;
    ClMem& operator=(const ClMem&) = delete;
    ClMem(ClMem&& o) noexcept : m(o.m) { o.m = nullptr; }
    ClMem& operator=(ClMem&& o) noexcept { if (this != &o) { if (m) clReleaseMemObject(m); m = o.m; o.m = nullptr; } return *this; }
    operator cl_mem() const { return m; }
};
} // namespace

bool compute_index_opencl(const RasterChunk& chunk, float* out_index, int index_code) {
    if (!g_ctx) return false;

    const size_t n = static_cast<size_t>(chunk.w) * chunk.h;
    if (n == 0) return true;
    const size_t bytes = n * sizeof(float);

    // Hold the dispatch lock for the whole GPU interaction (buffer setup through
    // readback) so concurrent OpenMP tiles never share the kernel's arg state.
    std::lock_guard<std::mutex> lock(g_dispatch_mutex);

    // Any CL failure returns false (not throws) so the caller runs the CPU path
    // for this tile — the documented per-tile fallback — and RAII frees buffers.
    try {
    cl_int err;
    auto ensure_buf = [&](const float* ptr) -> ClMem {
        if (ptr) {
            cl_mem m = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)ptr, &err);
            HSI_CL_CHECK(err);
            return ClMem(m);
        } else {
            cl_mem m = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
            HSI_CL_CHECK(err);
            ClMem guard(m);
            std::vector<float> zeros(n, 0.0f);
            HSI_CL_CHECK(clEnqueueWriteBuffer(g_queue, m, CL_TRUE, 0, bytes, zeros.data(), 0, nullptr, nullptr));
            return guard;
        }
    };

    ClMem buf_red   = ensure_buf(chunk.red);
    ClMem buf_nir   = ensure_buf(chunk.nir);
    ClMem buf_green = ensure_buf(chunk.green);
    ClMem buf_swir  = ensure_buf(chunk.swir);
    ClMem buf_blue  = ensure_buf(chunk.blue);

    cl_mem out_raw = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    HSI_CL_CHECK(err);
    ClMem buf_out(out_raw);

    // clSetKernelArg needs the address of a cl_mem value; take plain handles
    // from the RAII guards (ownership stays with the guards).
    cl_mem a_red = buf_red, a_nir = buf_nir, a_green = buf_green,
           a_swir = buf_swir, a_blue = buf_blue, a_out = buf_out;
    int arg = 0;
    int n_int = static_cast<int>(n);
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_mem), &a_red));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_mem), &a_nir));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_mem), &a_green));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_mem), &a_swir));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_mem), &a_blue));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_mem), &a_out));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_int), &n_int));
    HSI_CL_CHECK(clSetKernelArg(g_kernel, arg++, sizeof(cl_int), &index_code));

    const size_t global_size = ((n + g_local_size - 1) / g_local_size) * g_local_size;
    HSI_CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global_size, &g_local_size, 0, nullptr, nullptr));

    HSI_CL_CHECK(clEnqueueReadBuffer(g_queue, a_out, CL_TRUE, 0, bytes, out_index, 0, nullptr, nullptr));

    return true;  // buffers released by ClMem destructors on scope exit
    } catch (const std::exception& e) {
        std::cerr << "[opencl] dispatch failed, using CPU for this tile: " << e.what() << "\n";
        return false;
    }
}

bool compute_spectral_features_gpu(const HyperspectralTile& tile, const BandConfig& config, SpectralFeatureSet& sf) {
    if (tile.bands.empty() || !g_ctx) return false;
    
    const size_t numPixels = tile.bands[0].values.size();
    
    // 1. Resize output arrays to match the tile size (initialized to 0.0)
    sf.ndvi.resize(numPixels, 0.0f);
    sf.ndwi.resize(numPixels, 0.0f);
    sf.ndbi.resize(numPixels, 0.0f);
    sf.savi.resize(numPixels, 0.0f);
    sf.evi.resize(numPixels, 0.0f);
    sf.ndmi.resize(numPixels, 0.0f);
    sf.msavi.resize(numPixels, 0.0f);

    // 2. Map the pointers from the Tile bands to our GPU struct
    RasterChunk chunk;
    chunk.w = tile.width();
    chunk.h = tile.height();
    
    for (const auto& b : tile.bands) {
        if (b.index == config.nir)   chunk.nir   = (float*)b.values.data();
        if (b.index == config.red)   chunk.red   = (float*)b.values.data();
        if (b.index == config.green) chunk.green = (float*)b.values.data();
        if (b.index == config.swir)  chunk.swir  = (float*)b.values.data();
        if (b.index == config.blue)  chunk.blue  = (float*)b.values.data();
    }

    // 3. Dispatch to the GPU for each of the 7 indices
    bool ok = true;
    ok &= compute_index_opencl(chunk, sf.ndvi.data(), 0);  // NDVI
    ok &= compute_index_opencl(chunk, sf.ndwi.data(), 1);  // NDWI
    ok &= compute_index_opencl(chunk, sf.ndbi.data(), 2);  // NDBI
    ok &= compute_index_opencl(chunk, sf.savi.data(), 3);  // SAVI
    ok &= compute_index_opencl(chunk, sf.evi.data(), 4);   // EVI
    ok &= compute_index_opencl(chunk, sf.ndmi.data(), 5);  // NDMI
    ok &= compute_index_opencl(chunk, sf.msavi.data(), 6); // MSAVI
    
    sf.computed = ok;
    return ok;
}

} // namespace gpu
} // namespace hsi
