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
    }
    return args;
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
            std::cout << "Warning: GPU/OpenCL unavailable — falling back to CPU for indices.\n";
#else
        std::cout << "Warning: built without OpenCL support — falling back to CPU for indices.\n";
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
    std::atomic<bool> failed{false};
    std::string firstError;
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
    // what actually ran (gpu only when the GPU backend initialised).
    const std::string ran_mode = (args.mode == "gpu" && !use_gpu)
                               ? std::string(kDefaultMode) : args.mode;
    std::cout << "RESULT mode=" << ran_mode << " tiles=" << tiles.size()
              << " time_sec=" << elapsed << "\n";

    return failed ? 2 : 0;
}
