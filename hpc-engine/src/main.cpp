// =============================================================================
// main.cpp  —  C++ HPC Preprocessing Engine (Serial + OpenMP)
// =============================================================================

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <cstdlib>
#include <thread>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <cmath>
#include <set>
#include <memory>
#include <fstream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "RasterReader.hpp"
#include "Tiler.hpp"
#include "NoDataHandler.hpp"
#include "BadBandRemover.hpp"
#include "Normalizer.hpp"
#include "SpectralFeatures.hpp"
#include "TileWriter.hpp"
#include "BandResolver.hpp"
#include "SAM.hpp"
#include "PCA.hpp"

#ifdef HSI_HAVE_OPENCL
#include "gpu/gpu_backend_opencl.hpp"
#endif

// Default mode/thread-count are chosen so the engine runs at full speed on
// any machine out of the box: when built with OpenMP support, default to
// parallel mode using all detected cores; otherwise fall back to serial.
// Both remain overridable via --mode/--threads.
#ifdef _OPENMP
static constexpr const char* kDefaultMode = "openmp";
#else
static constexpr const char* kDefaultMode = "serial";
#endif

struct CliArgs {
    std::string input;
    std::string output;
    int tileSize = 512;
    int threads  = 0; // 0 = auto-detect hardware_concurrency() at runtime
    bool nodata    = true;
    bool badband   = true;
    bool normalize = true;
    bool spectral  = true;
    BandConfig bands = {75, 47, 30, 151, 14}; // standard-EnMAP fallback (order: nir,red,green,swir,blue)
    BandOverrides bandOverrides;              // which roles the user pinned on the CLI
    float nodataValue = -9999.0f;
    std::string mode = kDefaultMode;

    // --- Operation dispatch (default "indices" = legacy behaviour) ---
    std::string operation = "indices";        // indices | pca | sam

    // PCA options
    int         pcaComponents = 3;
    std::string pcaMatrix     = "covariance"; // covariance | correlation
    std::string pcaExclude;                   // e.g. "128-138,160-170" (1-based)

    // SAM options
    std::string samSpectrum;                  // "v1,v2,...,vB"
    int         samRefRow = -1;               // 0-based pixel row
    int         samRefCol = -1;               // 0-based pixel col
    bool        samHasThreshold = false;
    float       samThreshold = 0.0f;          // radians
    bool        samEmitSimilarity = false;
};

CliArgs parseArgs(int argc, char** argv) {
    CliArgs args;
    auto toInt = [](const char* s, int def)   { try { return std::stoi(s); } catch (...) { return def; } };
    auto toFlt = [](const char* s, float def) { try { return std::stof(s); } catch (...) { return def; } };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--input"      && i+1 < argc) args.input        = argv[++i];
        else if (a == "--output"     && i+1 < argc) args.output       = argv[++i];
        else if (a == "--mode"       && i+1 < argc) args.mode         = argv[++i];
        else if (a == "--tile-size"  && i+1 < argc) args.tileSize     = toInt(argv[++i], 512);
        else if (a == "--threads"    && i+1 < argc) args.threads      = toInt(argv[++i], 0);
        else if (a == "--nodata-val" && i+1 < argc) args.nodataValue  = toFlt(argv[++i], -9999.0f);
        else if (a == "--nir-band"   && i+1 < argc) { args.bands.nir   = toInt(argv[++i], 75);  args.bandOverrides.nir   = true; }
        else if (a == "--red-band"   && i+1 < argc) { args.bands.red   = toInt(argv[++i], 47);  args.bandOverrides.red   = true; }
        else if (a == "--green-band" && i+1 < argc) { args.bands.green = toInt(argv[++i], 30);  args.bandOverrides.green = true; }
        else if (a == "--swir-band"  && i+1 < argc) { args.bands.swir  = toInt(argv[++i], 151); args.bandOverrides.swir  = true; }
        else if (a == "--blue-band"  && i+1 < argc) { args.bands.blue  = toInt(argv[++i], 14);  args.bandOverrides.blue  = true; }
        // --- new dispatch + operation options ---
        else if (a == "--operation"      && i+1 < argc) args.operation     = argv[++i];
        else if (a == "--pca-components" && i+1 < argc) args.pcaComponents  = toInt(argv[++i], 3);
        else if (a == "--pca-matrix"     && i+1 < argc) args.pcaMatrix      = argv[++i];
        else if (a == "--pca-exclude-bands" && i+1 < argc) args.pcaExclude  = argv[++i];
        else if (a == "--sam-spectrum"   && i+1 < argc) args.samSpectrum    = argv[++i];
        else if (a == "--sam-ref-row"    && i+1 < argc) args.samRefRow      = toInt(argv[++i], -1);
        else if (a == "--sam-ref-col"    && i+1 < argc) args.samRefCol      = toInt(argv[++i], -1);
        else if (a == "--sam-threshold"  && i+1 < argc) { args.samThreshold = toFlt(argv[++i], 0.0f); args.samHasThreshold = true; }
        else if (a == "--sam-emit-similarity") args.samEmitSimilarity = true;
    }
    return args;
}

// =============================================================================
// SAM (Spectral Angle Mapper) — whole-scene single pass.
//   Writes sam_angle.tif (always), sam_mask.tif (if --sam-threshold),
//   sam_similarity.tif (if --sam-emit-similarity). Uses RAW spectra: no
//   BadBandRemover / Normalizer; NoDataHandler only maps the nodata sentinel to
//   0 so an all-zero pixel is treated as nodata (angle -1.0).
// =============================================================================
static size_t runSAM(const CliArgs& args, const RasterReader& reader,
                     const std::vector<TileBounds>& tiles, bool use_gpu,
                     std::atomic<bool>& failed, std::string& firstError) {
    const int B = reader.bandCount();

    // --- resolve the reference spectrum (length must equal B) ---
    std::vector<float> ref;
    if (!args.samSpectrum.empty()) {
        ref = sam::parseSpectrum(args.samSpectrum);
        if (static_cast<int>(ref.size()) != B)
            throw std::runtime_error("SAM --sam-spectrum has " +
                std::to_string(ref.size()) + " values but the scene has " +
                std::to_string(B) + " bands");
    } else if (args.samRefRow >= 0 && args.samRefCol >= 0) {
        if (args.samRefRow >= reader.rows() || args.samRefCol >= reader.cols())
            throw std::runtime_error("SAM reference pixel out of range");
        TileBounds pb{args.samRefRow, args.samRefCol,
                      args.samRefRow + 1, args.samRefCol + 1};
        HyperspectralTile p = reader.readTile(pb);
        NoDataHandler::apply(p, args.nodataValue);
        ref.resize(B);
        for (int b = 0; b < B; ++b) ref[b] = p.bands[b].values[0];
    } else {
        throw std::runtime_error(
            "SAM requires --sam-spectrum OR --sam-ref-row/--sam-ref-col");
    }

    const float refNorm = sam::referenceNorm(ref);
    if (!(refNorm > 0.0f))
        throw std::runtime_error("SAM reference vector has zero norm (nodata?)");

    std::filesystem::create_directories(args.output);
    SceneRaster angleRaster(args.output + "/sam_angle.tif",
        reader.cols(), reader.rows(), 1, SceneRaster::Type::Float32,
        reader.geoTransform(), reader.projection(), -1.0,
        {"SAM spectral angle (radians)"});

    std::unique_ptr<SceneRaster> maskRaster, simRaster;
    if (args.samHasThreshold)
        maskRaster = std::make_unique<SceneRaster>(args.output + "/sam_mask.tif",
            reader.cols(), reader.rows(), 1, SceneRaster::Type::Byte,
            reader.geoTransform(), reader.projection(), 255.0,
            std::vector<std::string>{"SAM detection mask (<= threshold)"});
    if (args.samEmitSimilarity)
        simRaster = std::make_unique<SceneRaster>(args.output + "/sam_similarity.tif",
            reader.cols(), reader.rows(), 1, SceneRaster::Type::Float32,
            reader.geoTransform(), reader.projection(), -1.0,
            std::vector<std::string>{"SAM cosine similarity [0,1]"});

    std::mutex errMutex;
    const long long tileCount = static_cast<long long>(tiles.size());
    // In GPU mode the host-side tile loop runs single-threaded: Apple's OpenCL
    // command queue is not reliably usable from multiple host threads even when
    // the dispatch is mutex-serialised (it silently corrupts results), and the
    // GPU already parallelises the per-pixel work. CPU mode stays multithreaded.
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(!use_gpu)
#endif
    for (long long i = 0; i < tileCount; ++i) {
      try {
        HyperspectralTile tile = reader.readTile(tiles[i]);
        NoDataHandler::apply(tile, args.nodataValue);

        std::vector<float> angle, sim;
        bool g = false;
#ifdef HSI_HAVE_OPENCL
        if (use_gpu) g = hsi::gpu::compute_sam_gpu(tile, ref, refNorm, angle, sim);
#endif
        if (!g) sam::computeTileCPU(tile, ref, refNorm, angle, sim);

        const int w = tile.width(), h = tile.height();
        const int cs = tiles[i].colStart, rs = tiles[i].rowStart;
        angleRaster.writeWindow(0, cs, rs, w, h, angle.data());
        if (simRaster)  simRaster->writeWindow(0, cs, rs, w, h, sim.data());
        if (maskRaster) {
            std::vector<unsigned char> mask;
            sam::deriveMask(angle, args.samThreshold, mask);
            maskRaster->writeWindow(0, cs, rs, w, h, mask.data());
        }
      } catch (const std::exception& e) {
        if (!failed.exchange(true)) {
            std::lock_guard<std::mutex> lock(errMutex);
            firstError = e.what();
        }
      }
    }

    angleRaster.finalize();
    if (maskRaster) maskRaster->finalize();
    if (simRaster)  simRaster->finalize();
    return tiles.size();
}

// =============================================================================
// PCA (global, whole-scene) — two passes to never hold the cube:
//   pass 1a  per-band means (float64, over valid pixels)
//   pass 1b  Σ(x-μ)(x-μ)^T  (float64, exact-mean form)   [GPU map / host reduce]
//   eigensolve (cyclic Jacobi) + svd_flip, write pca_eigen.csv
//   pass 2   project top-k, write pca.tif                 [GPU per-pixel]
// Uses RAW spectra: NoDataHandler only (no BadBandRemover / Normalizer), so the
// per-tile band set stays identical to the global band set.
// =============================================================================
static constexpr float kPcaNodata = -9999.0f;

// Load a tile for PCA (raw + nodata→0) and compute its per-pixel valid mask:
// a pixel is valid iff at least one KEPT band is non-zero.
static void loadTileForPCA(const RasterReader& reader, const TileBounds& tb,
                           float nodataValue, const std::vector<int>& kept,
                           HyperspectralTile& tile,
                           std::vector<unsigned char>& valid) {
    tile = reader.readTile(tb);
    NoDataHandler::apply(tile, nodataValue);
    const size_t n = tile.bands.empty() ? 0 : tile.bands[0].values.size();
    valid.assign(n, 0);
    const int Bk = static_cast<int>(kept.size());
    std::vector<const float*> bp(Bk);
    for (int r = 0; r < Bk; ++r) bp[r] = tile.bands[kept[r] - 1].values.data();
    for (size_t i = 0; i < n; ++i) {
        for (int r = 0; r < Bk; ++r) {
            if (bp[r][i] != 0.0f) { valid[i] = 1; break; }
        }
    }
}

static size_t runPCA(const CliArgs& args, const RasterReader& reader,
                     const std::vector<TileBounds>& tiles, bool use_gpu,
                     std::atomic<bool>& failed, std::string& firstError) {
    const int B = reader.bandCount();

    // --- kept bands (drop --pca-exclude-bands, 1-based) ---
    std::set<int> excluded = pca::parseExcludeBands(args.pcaExclude);
    std::vector<int> kept;
    for (int b = 1; b <= B; ++b) if (!excluded.count(b)) kept.push_back(b);
    const int Bk = static_cast<int>(kept.size());
    if (Bk < 2) throw std::runtime_error("PCA needs at least 2 kept bands");

    int k = args.pcaComponents;
    if (k < 1) k = 1;
    if (k > Bk) {
        std::cout << "Warning: --pca-components " << k << " > kept bands " << Bk
                  << "; clamping to " << Bk << ".\n";
        k = Bk;
    }
    const bool correlation = (args.pcaMatrix == "correlation");
    std::cout << "PCA: " << Bk << " kept bands, k=" << k
              << ", matrix=" << (correlation ? "correlation" : "covariance") << "\n";
    std::filesystem::create_directories(args.output);

    const long long tileCount = static_cast<long long>(tiles.size());
    std::mutex mergeMutex, errMutex;
    auto recordError = [&](const char* what) {
        if (!failed.exchange(true)) {
            std::lock_guard<std::mutex> lock(errMutex);
            firstError = what;
        }
    };

    // ---- pass 1a: per-band means over valid pixels ----
    std::vector<double> gsum(Bk, 0.0);
    long long gcount = 0;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<double> lsum(Bk, 0.0);
        long long lcount = 0;
#ifdef _OPENMP
        #pragma omp for schedule(dynamic) nowait
#endif
        for (long long i = 0; i < tileCount; ++i) {
          try {
            HyperspectralTile tile;
            std::vector<unsigned char> valid;
            loadTileForPCA(reader, tiles[i], args.nodataValue, kept, tile, valid);
            const size_t n = valid.size();
            std::vector<const float*> bp(Bk);
            for (int r = 0; r < Bk; ++r) bp[r] = tile.bands[kept[r] - 1].values.data();
            for (size_t px = 0; px < n; ++px) {
                if (!valid[px]) continue;
                for (int r = 0; r < Bk; ++r) lsum[r] += bp[r][px];
                ++lcount;
            }
          } catch (const std::exception& e) { recordError(e.what()); }
        }
        std::lock_guard<std::mutex> lock(mergeMutex);
        for (int r = 0; r < Bk; ++r) gsum[r] += lsum[r];
        gcount += lcount;
    }
    if (failed) return tiles.size();
    if (gcount < 2) throw std::runtime_error("PCA: fewer than 2 valid pixels");

    std::vector<double> mean(Bk);
    for (int r = 0; r < Bk; ++r) mean[r] = gsum[r] / static_cast<double>(gcount);
    std::vector<float> meanF(mean.begin(), mean.end());

    // ---- pass 1b: covariance Σ(x-μ)(x-μ)^T (float64) ----
    std::vector<double> gcov(static_cast<size_t>(Bk) * Bk, 0.0);
#ifdef _OPENMP
    #pragma omp parallel if(!use_gpu)
#endif
    {
        std::vector<double> lcov(static_cast<size_t>(Bk) * Bk, 0.0);
        std::vector<double> dev(Bk);
#ifdef _OPENMP
        #pragma omp for schedule(dynamic) nowait
#endif
        for (long long i = 0; i < tileCount; ++i) {
          try {
            HyperspectralTile tile;
            std::vector<unsigned char> valid;
            loadTileForPCA(reader, tiles[i], args.nodataValue, kept, tile, valid);
            const size_t n = valid.size();

            bool g = false;
#ifdef HSI_HAVE_OPENCL
            if (use_gpu) {
                std::vector<double> part;
                if (hsi::gpu::compute_pca_cov_gpu(tile, kept, meanF, valid, part)) {
                    for (size_t e = 0; e < lcov.size(); ++e) lcov[e] += part[e];
                    g = true;
                }
            }
#endif
            if (!g) {
                std::vector<const float*> bp(Bk);
                for (int r = 0; r < Bk; ++r) bp[r] = tile.bands[kept[r] - 1].values.data();
                for (size_t px = 0; px < n; ++px) {
                    if (!valid[px]) continue;
                    for (int j = 0; j < Bk; ++j) dev[j] = bp[j][px] - mean[j];
                    for (int j = 0; j < Bk; ++j) {
                        const double dj = dev[j];
                        double* row = &lcov[static_cast<size_t>(j) * Bk];
                        for (int kk = 0; kk < Bk; ++kk) row[kk] += dj * dev[kk];
                    }
                }
            }
          } catch (const std::exception& e) { recordError(e.what()); }
        }
        std::lock_guard<std::mutex> lock(mergeMutex);
        for (size_t e = 0; e < gcov.size(); ++e) gcov[e] += lcov[e];
    }
    if (failed) return tiles.size();

    // Sample covariance (divide by N-1, matching numpy np.cov ddof=1).
    const double denom = static_cast<double>(gcount - 1);
    for (auto& v : gcov) v /= denom;

    // Per-band std (from the covariance diagonal); invStd used by projection.
    std::vector<double> stdv(Bk), invStd(Bk, 1.0);
    for (int j = 0; j < Bk; ++j) {
        stdv[j] = std::sqrt(std::max(0.0, gcov[static_cast<size_t>(j) * Bk + j]));
        if (correlation) invStd[j] = (stdv[j] > 0.0) ? 1.0 / stdv[j] : 0.0;
    }

    // Build the matrix to decompose: covariance (default) or correlation.
    std::vector<double> matrix = gcov;
    if (correlation) {
        for (int j = 0; j < Bk; ++j)
            for (int kk = 0; kk < Bk; ++kk) {
                const double d = stdv[j] * stdv[kk];
                matrix[static_cast<size_t>(j) * Bk + kk] =
                    (d > 0.0) ? gcov[static_cast<size_t>(j) * Bk + kk] / d : 0.0;
            }
    }

    // Eigensolve (descending) + deterministic sign convention.
    std::vector<double> evals, evecs;
    pca::jacobiEigenSymmetric(matrix, Bk, evals, evecs);
    pca::svdFlipColumns(evecs, Bk);

    // Write pca_eigen.csv (ALL kept components).
    {
        double total = 0.0;
        for (double e : evals) total += e;
        std::ofstream ef(args.output + "/pca_eigen.csv", std::ios::trunc);
        if (!ef.is_open())
            throw std::runtime_error("Cannot open pca_eigen.csv for writing");
        ef << "component,eigenvalue,variance_pct,cumulative_pct\n";
        ef << std::fixed << std::setprecision(8);
        double cum = 0.0;
        for (int c = 0; c < Bk; ++c) {
            const double pct = (total != 0.0) ? 100.0 * evals[c] / total : 0.0;
            cum += pct;
            ef << (c + 1) << "," << evals[c] << "," << pct << "," << cum << "\n";
        }
    }

    // Top-k eigenvectors, packed column-major Vk[c*Bk + b] for the projection.
    std::vector<float> Vk(static_cast<size_t>(Bk) * k);
    for (int c = 0; c < k; ++c)
        for (int b = 0; b < Bk; ++b)
            Vk[static_cast<size_t>(c) * Bk + b] =
                static_cast<float>(evecs[static_cast<size_t>(b) * Bk + c]);
    std::vector<float> invStdF(invStd.begin(), invStd.end());

    // ---- pass 2: project top-k and write pca.tif ----
    std::vector<std::string> pcDesc(k);
    for (int c = 0; c < k; ++c) pcDesc[c] = "PC" + std::to_string(c + 1);
    SceneRaster pcaRaster(args.output + "/pca.tif",
        reader.cols(), reader.rows(), k, SceneRaster::Type::Float32,
        reader.geoTransform(), reader.projection(), kPcaNodata, pcDesc);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(!use_gpu)
#endif
    for (long long i = 0; i < tileCount; ++i) {
      try {
        HyperspectralTile tile;
        std::vector<unsigned char> valid;
        loadTileForPCA(reader, tiles[i], args.nodataValue, kept, tile, valid);
        const size_t n = valid.size();
        const int w = tile.width(), h = tile.height();
        const int cs = tiles[i].colStart, rs = tiles[i].rowStart;

        std::vector<std::vector<float>> comp;
        bool g = false;
#ifdef HSI_HAVE_OPENCL
        if (use_gpu)
            g = hsi::gpu::compute_pca_project_gpu(tile, kept, meanF, invStdF,
                                                  Vk, k, valid, kPcaNodata, comp);
#endif
        if (!g) {
            comp.assign(k, std::vector<float>(n, kPcaNodata));
            std::vector<const float*> bp(Bk);
            for (int r = 0; r < Bk; ++r) bp[r] = tile.bands[kept[r] - 1].values.data();
            std::vector<double> dev(Bk);
            for (size_t px = 0; px < n; ++px) {
                if (!valid[px]) continue;
                for (int b = 0; b < Bk; ++b) dev[b] = (bp[b][px] - mean[b]) * invStd[b];
                for (int c = 0; c < k; ++c) {
                    double acc = 0.0;
                    const float* vcol = &Vk[static_cast<size_t>(c) * Bk];
                    for (int b = 0; b < Bk; ++b) acc += vcol[b] * dev[b];
                    comp[c][px] = static_cast<float>(acc);
                }
            }
        }
        for (int c = 0; c < k; ++c)
            pcaRaster.writeWindow(c, cs, rs, w, h, comp[c].data());
      } catch (const std::exception& e) { recordError(e.what()); }
    }

    pcaRaster.finalize();
    return tiles.size();
}

int main(int argc, char** argv) {
    CliArgs args = parseArgs(argc, argv);

    if (args.input.empty() || args.output.empty()) {
        std::cerr << "Usage: " << argv[0] << " --input <file> --output <dir> [options]\n";
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    RasterReader reader(args.input);
    Tiler tiler(reader.rows(), reader.cols(), args.tileSize);
    auto tiles = tiler.computeTiles();

    // Resolve the spectral-index bands from the scene's own wavelengths (nearest
    // band per role), unless the user pinned them on the CLI. This keeps the
    // indices physically correct for any cube, not just a hard-coded layout.
    resolveBandsFromWavelengths(args.bands, reader.wavelengths(), args.bandOverrides);
    auto wlOf = [&](int b1) -> float {
        const auto& wl = reader.wavelengths();
        return (b1 >= 1 && b1 <= (int)wl.size()) ? wl[b1 - 1] : 0.0f;
    };

    std::cout << "Scene: " << reader.rows() << "×" << reader.cols()
              << " pixels, " << reader.bandCount() << " bands\n";
    std::cout << "Tiles: " << tiles.size()
              << " (" << args.tileSize << "×" << args.tileSize << ")\n";
    std::cout << "Bands: NIR=" << args.bands.nir   << "(" << wlOf(args.bands.nir)   << "nm)"
              << " Red="        << args.bands.red   << "(" << wlOf(args.bands.red)   << "nm)"
              << " Green="      << args.bands.green << "(" << wlOf(args.bands.green) << "nm)"
              << " SWIR="       << args.bands.swir  << "(" << wlOf(args.bands.swir)  << "nm)"
              << " Blue="       << args.bands.blue  << "(" << wlOf(args.bands.blue)  << "nm)\n";
    std::cout << "Mode: " << args.mode << "\n";

    // Resolve the GPU backend up front. `--mode gpu` requests the cross-vendor
    // OpenCL path (Apple/NVIDIA/Intel/AMD); if it can't initialise (no OpenCL,
    // no GPU, or the build has no OpenCL support) we transparently fall back to
    // the CPU SpectralFeatures path so the run always completes.
    bool use_gpu = false;
    if (args.mode == "gpu") {
#ifdef HSI_HAVE_OPENCL
        use_gpu = hsi::gpu::init_opencl();
        if (!use_gpu)
            std::cout << "Warning: GPU/OpenCL unavailable — falling back to CPU.\n";
#else
        std::cout << "Warning: built without OpenCL support — falling back to CPU.\n";
#endif
    }

    // Threads are used for tile I/O in both openmp and gpu modes (GPU index
    // dispatch is serialised internally); only "serial" forces a single thread.
    const bool runParallel = (args.mode == "openmp" || args.mode == "gpu");

#ifdef _OPENMP
    if (runParallel) {
        int hw_cores = std::thread::hardware_concurrency();
        if (hw_cores == 0) hw_cores = 8;
        // threads == 0 means "not explicitly set" -> use every detected core
        // so the engine is fast by default on any machine.
        int requested = (args.threads > 0) ? args.threads : hw_cores;
        int safe_threads = std::min(requested, hw_cores);
        if (requested > hw_cores) {
            std::cout << "Warning: Requested " << requested
                      << " threads, but system only has " << hw_cores
                      << " cores. Clamping to " << safe_threads << ".\n";
        }
        std::cout << "Using " << safe_threads << " of " << hw_cores << " cores.\n";
        omp_set_num_threads(safe_threads);
    } else {
        omp_set_num_threads(1);
    }
#endif

    std::atomic<bool> failed{false};
    std::string firstError;

    if (args.operation == "sam") {
        try { runSAM(args, reader, tiles, use_gpu, failed, firstError); }
        catch (const std::exception& e) { failed = true; firstError = e.what(); }
    } else if (args.operation == "pca") {
        try { runPCA(args, reader, tiles, use_gpu, failed, firstError); }
        catch (const std::exception& e) { failed = true; firstError = e.what(); }
    } else if (args.operation == "indices") {
    std::mutex csvMutex;
    std::filesystem::create_directories(args.output);
    const std::string summaryCsvPath = args.output + "/summary.csv";

    // Fresh-run hygiene. summary.csv is opened in append mode (so OpenMP threads
    // can each add their tile rows), and tiles are written to deterministic
    // tile_r####_c####.tif names. Without clearing prior outputs first, re-running
    // the engine into an existing directory would:
    //   (1) append a *second* full set of rows to summary.csv — doubling every
    //       tile for the Spark/Sedona layer, which silently inflates counts and
    //       spatial-join results; and
    //   (2) leave stale tile_*.tif from an earlier, possibly differently-tiled
    //       scene that the QGIS plugin's gdalbuildvrt step would then fold into
    //       the mosaic.
    // Remove only files this engine itself produces (summary.csv and tile_r*).
    {
        std::error_code ec;
        std::filesystem::remove(summaryCsvPath, ec);
        // Collect first, then remove — mutating a directory while iterating it
        // with directory_iterator is unspecified behaviour.
        std::vector<std::filesystem::path> stale;
        for (const auto& entry : std::filesystem::directory_iterator(args.output, ec)) {
            const std::string name = entry.path().filename().string();
            const std::string ext  = entry.path().extension().string();
            if (name.rfind("tile_r", 0) == 0 && (ext == ".tif" || ext == ".xml"))
                stale.push_back(entry.path());
        }
        for (const auto& p : stale) std::filesystem::remove(p, ec);
    }

    // BUG #5 FIX: 'done' must NOT be static — static means it never resets
    // between runs (e.g. benchmark loop calling the engine multiple times).
    std::atomic<size_t> done{0};

    // An exception that escapes an OpenMP parallel region is undefined behaviour
    // (libgomp/libomp call std::terminate). readTile/writeGeoTIFF can throw on
    // I/O failure, so every iteration catches, records the first error, and the
    // run reports a non-zero exit afterwards instead of aborting mid-tile.
    std::mutex errMutex;

    // MSVC's OpenMP (2.0) requires a signed loop index (error C3016 otherwise);
    // precompute a signed bound so the parallel-for is valid on MSVC too.
    const long long tileCount = static_cast<long long>(tiles.size());
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) shared(tiles, args, reader) if(runParallel)
#endif
    for (long long i = 0; i < tileCount; ++i) {
      try {
        HyperspectralTile tile = reader.readTile(tiles[i]);

        if (args.nodata)  NoDataHandler::apply(tile, args.nodataValue);
        if (args.badband) BadBandRemover::apply(tile);

        // BUG #4 FIX: SpectralFeatures MUST run BEFORE Normalizer.
        // NDVI/NDWI/NDBI require raw/nodata-clean reflectance values.
        // Per-band min-max normalisation changes the relative magnitudes
        // between bands, which produces incorrect index values.
        SpectralFeatureSet sf;
        if (args.spectral) {
            bool done_on_gpu = false;
#ifdef HSI_HAVE_OPENCL
            if (use_gpu)
                done_on_gpu = hsi::gpu::compute_spectral_features_gpu(tile, args.bands, sf);
#endif
            // CPU path: used for serial/openmp, and as a fallback if the GPU
            // dispatch failed for this tile. Identical math, so results match.
            if (!done_on_gpu)
                sf = SpectralFeatures::compute(tile, args.bands);
        }

        // Normalise AFTER computing indices — only affects the visual GeoTIFF bands.
        if (args.normalize) Normalizer::apply(tile);

        TileWriteResult wr = TileWriter::writeGeoTIFF(tile, sf, args.output,
                                 tiles[i].rowStart, tiles[i].colStart);
        TileWriter::appendSummaryCSV(tile, sf, wr, summaryCsvPath,
                                     tiles[i].rowStart, tiles[i].colStart, csvMutex);

        size_t n = ++done;
        if (n % 10 == 0 || n == tiles.size())
            std::cout << "Progress: " << n << "/" << tiles.size() << "\n";
      } catch (const std::exception& e) {
        if (!failed.exchange(true)) {
            std::lock_guard<std::mutex> lock(errMutex);
            firstError = e.what();
        }
      }
    }
    } else {
        std::cerr << "ERROR: unknown --operation '" << args.operation
                  << "' (expected indices|pca|sam)\n";
#ifdef HSI_HAVE_OPENCL
        if (use_gpu) hsi::gpu::teardown_opencl();
#endif
        return 1;
    }

    if (failed) {
        std::cerr << "ERROR: at least one tile failed to process: "
                  << firstError << "\n";
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

#ifdef HSI_HAVE_OPENCL
    if (use_gpu) hsi::gpu::teardown_opencl();
#endif

    // The exact string that the QGIS plugin regex will match. `mode` reflects
    // what actually ran (gpu only when the GPU backend initialised). The
    // `operation` and `bands` fields are the contract additions; `bands` is the
    // scene's input band count B (same for every operation).
    const std::string ran_mode = (args.mode == "gpu" && !use_gpu)
                               ? std::string(kDefaultMode) : args.mode;
    std::cout << "RESULT mode=" << ran_mode
              << " operation=" << args.operation
              << " tiles=" << tiles.size()
              << " bands=" << reader.bandCount()
              << " time_sec=" << elapsed << "\n";

    return failed ? 2 : 0;
}
