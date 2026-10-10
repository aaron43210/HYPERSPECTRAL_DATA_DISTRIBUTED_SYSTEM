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
static cl_device_id g_device = nullptr;
static size_t g_local_size = 256;

// Lazily-built second program holding the whole-scene op kernels (SAM, PCA).
// Kept separate from the indices program so a build failure here can never
// disable the (already working) indices GPU path — it only forces the CPU
// fallback for SAM/PCA.
static cl_program g_ops_prog     = nullptr;
static cl_kernel  g_k_sam        = nullptr;
static cl_kernel  g_k_pca_cov    = nullptr;
static cl_kernel  g_k_pca_proj   = nullptr;
static bool       g_ops_ready    = false;
static bool       g_ops_tried    = false;

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

    g_device = device;

    char name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    cl_uint compute_units = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
    std::cout << "[opencl] Initializing Device: " << name << " (" << compute_units << " Compute Units)\n";

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
    if (g_k_sam)      { clReleaseKernel(g_k_sam);      g_k_sam = nullptr; }
    if (g_k_pca_cov)  { clReleaseKernel(g_k_pca_cov);  g_k_pca_cov = nullptr; }
    if (g_k_pca_proj) { clReleaseKernel(g_k_pca_proj); g_k_pca_proj = nullptr; }
    if (g_ops_prog)   { clReleaseProgram(g_ops_prog);  g_ops_prog = nullptr; }
    g_ops_ready = false;
    g_ops_tried = false;
    if (g_kernel) { clReleaseKernel(g_kernel); g_kernel = nullptr; }
    if (g_prog)   { clReleaseProgram(g_prog); g_prog = nullptr; }
    if (g_queue)  { clReleaseCommandQueue(g_queue); g_queue = nullptr; }
    if (g_ctx)    { clReleaseContext(g_ctx); g_ctx = nullptr; }
    g_device = nullptr;
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

// ===========================================================================
// Whole-scene op kernels (SAM, PCA). Single-precision float throughout so they
// build on GPUs without cl_khr_fp64; float64 reductions happen on the host.
// ===========================================================================
static const char* OPS_KERNEL_SRC = R"CLC(
// --- SAM: one work-item per pixel ---------------------------------------
__kernel void sam_angle(__global const float* data,   // B*n band-major
                        __constant float*     ref,     // B
                        int n, int B, float refNorm,
                        __global float* outAngle,
                        __global float* outSim)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    float dot = 0.0f, nt = 0.0f;
    for (int b = 0; b < B; ++b) {
        const float t  = data[(long)b * n + i];
        const float rv = ref[b];
        dot += t * rv;
        nt  += t * t;
    }
    if (nt <= 0.0f || refNorm <= 0.0f) { outAngle[i] = -1.0f; outSim[i] = -1.0f; return; }
    float cosv = dot / (sqrt(nt) * refNorm);
    cosv = fmax(-1.0f, fmin(1.0f, cosv));   // clamp BEFORE acos (fast-math safe)
    outAngle[i] = acos(cosv);
    outSim[i]   = cosv < 0.0f ? 0.0f : cosv;
}

// --- PCA covariance partial: one work-item per (j,k) matrix entry --------
__kernel void pca_cov(__global const float* data,   // Bk*n band-major
                      __constant float*     mean,    // Bk
                      __global const uchar* valid,   // n (1=valid, 0=nodata)
                      int n, int Bk,
                      __global float* cov)           // Bk*Bk row-major partial
{
    const int id = get_global_id(0);
    if (id >= Bk * Bk) return;
    const int j = id / Bk;
    const int k = id % Bk;
    const float mj = mean[j];
    const float mk = mean[k];
    float sum = 0.0f, comp = 0.0f;            // Kahan-compensated float sum
    const long baseJ = (long)j * n;
    const long baseK = (long)k * n;
    for (int i = 0; i < n; ++i) {
        if (valid[i] == 0) continue;
        const float prod = (data[baseJ + i] - mj) * (data[baseK + i] - mk);
        const float y = prod - comp;
        const float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    cov[id] = sum;
}

// --- PCA projection: one work-item per pixel -----------------------------
__kernel void pca_project(__global const float* data,  // Bk*n band-major
                          __constant float*     mean,   // Bk
                          __constant float*     invStd, // Bk (1.0 for covariance)
                          __global const float* Vk,      // Bk*k column-major
                          __global const uchar* valid,   // n
                          int n, int Bk, int k, float nodataVal,
                          __global float* out)           // k*n band-major
{
    const int i = get_global_id(0);
    if (i >= n) return;
    if (valid[i] == 0) {
        for (int c = 0; c < k; ++c) out[(long)c * n + i] = nodataVal;
        return;
    }
    for (int c = 0; c < k; ++c) {
        float acc = 0.0f;
        const long vbase = (long)c * Bk;
        for (int b = 0; b < Bk; ++b)
            acc += Vk[vbase + b] * ((data[(long)b * n + i] - mean[b]) * invStd[b]);
        out[(long)c * n + i] = acc;
    }
}
)CLC";

// Build the ops program once (lazy). Returns false (and leaves CPU fallback in
// charge) if the context is absent or the program/kernels cannot be built.
static bool ensure_ops_kernels() {
    std::lock_guard<std::mutex> lock(g_dispatch_mutex);
    if (g_ops_ready) return true;
    if (g_ops_tried) return false;      // already failed once; don't re-spam
    g_ops_tried = true;
    if (!g_ctx || !g_device) return false;
    try {
        cl_int err;
        g_ops_prog = clCreateProgramWithSource(g_ctx, 1, &OPS_KERNEL_SRC, nullptr, &err);
        HSI_CL_CHECK(err);
        err = clBuildProgram(g_ops_prog, 1, &g_device, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(g_ops_prog, g_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            if (log_size > 0)
                clGetProgramBuildInfo(g_ops_prog, g_device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::cerr << "[opencl] ops build log:\n" << log << "\n";
            return false;
        }
        g_k_sam      = clCreateKernel(g_ops_prog, "sam_angle",   &err); HSI_CL_CHECK(err);
        g_k_pca_cov  = clCreateKernel(g_ops_prog, "pca_cov",     &err); HSI_CL_CHECK(err);
        g_k_pca_proj = clCreateKernel(g_ops_prog, "pca_project", &err); HSI_CL_CHECK(err);
        g_ops_ready = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[opencl] ops kernels unavailable, using CPU: " << e.what() << "\n";
        return false;
    }
}

// Upload the kept bands of a tile as one packed band-major device buffer:
// packed[row*n + i] = tile.bands[keptBands[row]-1].values[i]. One write per
// band (no host-side packed copy). Caller holds g_dispatch_mutex.
static ClMem upload_band_major(const HyperspectralTile& tile,
                               const std::vector<int>& keptBands,
                               size_t n, cl_int& err) {
    const size_t rows  = keptBands.size();
    const size_t bytes = rows * n * sizeof(float);
    cl_mem raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    HSI_CL_CHECK(err);
    ClMem buf(raw);
    for (size_t row = 0; row < rows; ++row) {
        const int b0 = keptBands[row] - 1;    // 1-based -> 0-based position
        HSI_CL_CHECK(clEnqueueWriteBuffer(g_queue, raw, CL_TRUE,
            row * n * sizeof(float), n * sizeof(float),
            tile.bands[b0].values.data(), 0, nullptr, nullptr));
    }
    HSI_CL_CHECK(clFinish(g_queue));
    return buf;
}

bool compute_sam_gpu(const HyperspectralTile& tile,
                     const std::vector<float>& ref, float refNorm,
                     std::vector<float>& outAngle, std::vector<float>& outSim) {
    if (!g_ctx || tile.bands.empty()) return false;
    if (!ensure_ops_kernels()) return false;

    const int    B = static_cast<int>(tile.bands.size());
    if (static_cast<int>(ref.size()) != B) return false;
    const size_t n = tile.bands[0].values.size();
    if (n == 0) { outAngle.clear(); outSim.clear(); return true; }

    outAngle.assign(n, -1.0f);
    outSim.assign(n, -1.0f);

    std::lock_guard<std::mutex> lock(g_dispatch_mutex);
    try {
        cl_int err;
        std::vector<int> allBands(B);
        for (int b = 0; b < B; ++b) allBands[b] = b + 1;
        ClMem buf_data = upload_band_major(tile, allBands, n, err);

        cl_mem ref_raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        B * sizeof(float), (void*)ref.data(), &err);
        HSI_CL_CHECK(err); ClMem buf_ref(ref_raw);

        const size_t bytes = n * sizeof(float);
        cl_mem a_raw = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        HSI_CL_CHECK(err); ClMem buf_angle(a_raw);
        cl_mem s_raw = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        HSI_CL_CHECK(err); ClMem buf_sim(s_raw);

        cl_mem d = buf_data, r = buf_ref, oa = buf_angle, os = buf_sim;
        int n_int = static_cast<int>(n), arg = 0;
        HSI_CL_CHECK(clSetKernelArg(g_k_sam, arg++, sizeof(cl_mem), &d));
        HSI_CL_CHECK(clSetKernelArg(g_k_sam, arg++, sizeof(cl_mem), &r));
        HSI_CL_CHECK(clSetKernelArg(g_k_sam, arg++, sizeof(cl_int), &n_int));
        HSI_CL_CHECK(clSetKernelArg(g_k_sam, arg++, sizeof(cl_int), &B));
        HSI_CL_CHECK(clSetKernelArg(g_k_sam, arg++, sizeof(cl_float), &refNorm));
        HSI_CL_CHECK(clSetKernelArg(g_k_sam, arg++, sizeof(cl_mem), &oa));
        HSI_CL_CHECK(clSetKernelArg(g_k_sam, arg++, sizeof(cl_mem), &os));

        const size_t global = ((n + g_local_size - 1) / g_local_size) * g_local_size;
        HSI_CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_k_sam, 1, nullptr, &global, &g_local_size, 0, nullptr, nullptr));
        HSI_CL_CHECK(clEnqueueReadBuffer(g_queue, oa, CL_TRUE, 0, bytes, outAngle.data(), 0, nullptr, nullptr));
        HSI_CL_CHECK(clEnqueueReadBuffer(g_queue, os, CL_TRUE, 0, bytes, outSim.data(), 0, nullptr, nullptr));
        HSI_CL_CHECK(clFinish(g_queue));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[opencl] SAM dispatch failed, using CPU for this tile: " << e.what() << "\n";
        return false;
    }
}

bool compute_pca_cov_gpu(const HyperspectralTile& tile,
                         const std::vector<int>& keptBands,
                         const std::vector<float>& mean,
                         const std::vector<unsigned char>& valid,
                         std::vector<double>& covPartialOut) {
    if (!g_ctx || tile.bands.empty()) return false;
    if (!ensure_ops_kernels()) return false;

    const int    Bk = static_cast<int>(keptBands.size());
    const size_t n  = tile.bands[0].values.size();
    covPartialOut.assign(static_cast<size_t>(Bk) * Bk, 0.0);
    if (n == 0 || Bk == 0) return true;

    std::lock_guard<std::mutex> lock(g_dispatch_mutex);
    try {
        cl_int err;
        ClMem buf_data = upload_band_major(tile, keptBands, n, err);

        cl_mem m_raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      Bk * sizeof(float), (void*)mean.data(), &err);
        HSI_CL_CHECK(err); ClMem buf_mean(m_raw);

        cl_mem v_raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      n * sizeof(unsigned char), (void*)valid.data(), &err);
        HSI_CL_CHECK(err); ClMem buf_valid(v_raw);

        const size_t covBytes = static_cast<size_t>(Bk) * Bk * sizeof(float);
        cl_mem c_raw = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, covBytes, nullptr, &err);
        HSI_CL_CHECK(err); ClMem buf_cov(c_raw);

        cl_mem d = buf_data, m = buf_mean, v = buf_valid, c = buf_cov;
        int n_int = static_cast<int>(n), arg = 0;
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_cov, arg++, sizeof(cl_mem), &d));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_cov, arg++, sizeof(cl_mem), &m));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_cov, arg++, sizeof(cl_mem), &v));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_cov, arg++, sizeof(cl_int), &n_int));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_cov, arg++, sizeof(cl_int), &Bk));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_cov, arg++, sizeof(cl_mem), &c));

        const size_t entries = static_cast<size_t>(Bk) * Bk;
        const size_t global  = ((entries + g_local_size - 1) / g_local_size) * g_local_size;
        HSI_CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_k_pca_cov, 1, nullptr, &global, &g_local_size, 0, nullptr, nullptr));

        std::vector<float> covF(entries);
        HSI_CL_CHECK(clEnqueueReadBuffer(g_queue, c, CL_TRUE, 0, covBytes, covF.data(), 0, nullptr, nullptr));
        HSI_CL_CHECK(clFinish(g_queue));
        for (size_t i = 0; i < entries; ++i) covPartialOut[i] = static_cast<double>(covF[i]);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[opencl] PCA cov dispatch failed, using CPU for this tile: " << e.what() << "\n";
        return false;
    }
}

bool compute_pca_project_gpu(const HyperspectralTile& tile,
                             const std::vector<int>& keptBands,
                             const std::vector<float>& mean,
                             const std::vector<float>& invStd,
                             const std::vector<float>& Vk, int k,
                             const std::vector<unsigned char>& valid,
                             float nodataVal,
                             std::vector<std::vector<float>>& outComponents) {
    if (!g_ctx || tile.bands.empty()) return false;
    if (!ensure_ops_kernels()) return false;

    const int    Bk = static_cast<int>(keptBands.size());
    const size_t n  = tile.bands[0].values.size();
    outComponents.assign(k, std::vector<float>(n, nodataVal));
    if (n == 0 || Bk == 0 || k == 0) return true;

    std::lock_guard<std::mutex> lock(g_dispatch_mutex);
    try {
        cl_int err;
        ClMem buf_data = upload_band_major(tile, keptBands, n, err);

        cl_mem m_raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      Bk * sizeof(float), (void*)mean.data(), &err);
        HSI_CL_CHECK(err); ClMem buf_mean(m_raw);
        cl_mem is_raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       Bk * sizeof(float), (void*)invStd.data(), &err);
        HSI_CL_CHECK(err); ClMem buf_istd(is_raw);
        cl_mem vk_raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       Vk.size() * sizeof(float), (void*)Vk.data(), &err);
        HSI_CL_CHECK(err); ClMem buf_vk(vk_raw);
        cl_mem v_raw = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      n * sizeof(unsigned char), (void*)valid.data(), &err);
        HSI_CL_CHECK(err); ClMem buf_valid(v_raw);

        const size_t outBytes = static_cast<size_t>(k) * n * sizeof(float);
        cl_mem o_raw = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, outBytes, nullptr, &err);
        HSI_CL_CHECK(err); ClMem buf_out(o_raw);

        cl_mem d = buf_data, m = buf_mean, is = buf_istd, vk = buf_vk, v = buf_valid, o = buf_out;
        int n_int = static_cast<int>(n), arg = 0;
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_mem), &d));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_mem), &m));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_mem), &is));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_mem), &vk));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_mem), &v));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_int), &n_int));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_int), &Bk));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_int), &k));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_float), &nodataVal));
        HSI_CL_CHECK(clSetKernelArg(g_k_pca_proj, arg++, sizeof(cl_mem), &o));

        const size_t global = ((n + g_local_size - 1) / g_local_size) * g_local_size;
        HSI_CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_k_pca_proj, 1, nullptr, &global, &g_local_size, 0, nullptr, nullptr));

        std::vector<float> flat(static_cast<size_t>(k) * n);
        HSI_CL_CHECK(clEnqueueReadBuffer(g_queue, o, CL_TRUE, 0, outBytes, flat.data(), 0, nullptr, nullptr));
        HSI_CL_CHECK(clFinish(g_queue));
        for (int c = 0; c < k; ++c)
            std::copy(flat.begin() + static_cast<size_t>(c) * n,
                      flat.begin() + static_cast<size_t>(c + 1) * n,
                      outComponents[c].begin());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[opencl] PCA projection dispatch failed, using CPU for this tile: " << e.what() << "\n";
        return false;
    }
}

} // namespace gpu
} // namespace hsi
