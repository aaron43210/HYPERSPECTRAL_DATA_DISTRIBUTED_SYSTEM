// =============================================================================
// main_mpi.cpp  —  C++ HPC Preprocessing Engine (MPI + OpenMP)
// =============================================================================

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <thread>   // std::thread::hardware_concurrency() (used in parseArgs)

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

struct CliArgs {
    std::string input;
    std::string output;
    int tileSize  = 512;
    int threads   = 1;
    bool nodata    = true;
    bool badband   = true;
    bool normalize = true;
    bool spectral  = true;
    BandConfig bands = {75, 47, 30, 151, 14}; // standard-EnMAP fallback (nir,red,green,swir,blue)
    BandOverrides bandOverrides;
    float nodataValue = -9999.0f;
    std::string mode = "mpi";
};

CliArgs parseArgs(int argc, char** argv) {
    CliArgs args;
    // Guarded conversions: a non-numeric value (e.g. "--tile-size foo") must not
    // throw std::invalid_argument and abort the whole MPI job — fall back to the
    // documented default instead. (Mirrors the toInt/toFlt helpers in main.cpp.)
    auto toInt = [](const char* s, int def)   { try { return std::stoi(s); } catch (...) { return def; } };
    auto toFlt = [](const char* s, float def) { try { return std::stof(s); } catch (...) { return def; } };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--input"      && i+1 < argc) args.input        = argv[++i];
        else if (arg == "--output"     && i+1 < argc) args.output       = argv[++i];
        else if (arg == "--tile-size"  && i+1 < argc) args.tileSize     = toInt(argv[++i], 512);
        else if (arg == "--threads"    && i+1 < argc) args.threads      = toInt(argv[++i], 1);
        else if (arg == "--nodata-val" && i+1 < argc) args.nodataValue  = toFlt(argv[++i], -9999.0f);
        else if (arg == "--nir-band"   && i+1 < argc) { args.bands.nir   = toInt(argv[++i], 75);  args.bandOverrides.nir   = true; }
        else if (arg == "--red-band"   && i+1 < argc) { args.bands.red   = toInt(argv[++i], 47);  args.bandOverrides.red   = true; }
        else if (arg == "--green-band" && i+1 < argc) { args.bands.green = toInt(argv[++i], 30);  args.bandOverrides.green = true; }
        else if (arg == "--swir-band"  && i+1 < argc) { args.bands.swir  = toInt(argv[++i], 151); args.bandOverrides.swir  = true; }
        else if (arg == "--blue-band"  && i+1 < argc) { args.bands.blue  = toInt(argv[++i], 14);  args.bandOverrides.blue  = true; }
        else if (arg == "--mode"       && i+1 < argc) args.mode         = argv[++i];
    }
    if (args.threads <= 0) args.threads = std::thread::hardware_concurrency();
    return args;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    CliArgs args = parseArgs(argc, argv);

#ifdef _OPENMP
    omp_set_num_threads(args.threads);
#endif

    if (args.input.empty() || args.output.empty()) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " --input <file> --output <dir> [options]\n";
        MPI_Finalize();
        return 1;
    }

    // BUG #6 FIX: Use std::filesystem instead of std::system("mkdir -p ...")
    // The old code was a shell injection vulnerability and non-portable.
    if (rank == 0) {
        std::filesystem::create_directories(args.output);
        // Fresh-run hygiene (see main.cpp): remove stale tile_*.tif from a prior,
        // possibly differently-tiled run so the plugin's VRT mosaic can't fold
        // them in. The final summary.csv is rewritten with std::ios::trunc below,
        // so it needs no separate clearing here.
        std::error_code ec;
        std::vector<std::filesystem::path> stale;
        for (const auto& entry : std::filesystem::directory_iterator(args.output, ec)) {
            const std::string name = entry.path().filename().string();
            const std::string ext  = entry.path().extension().string();
            if (name.rfind("tile_r", 0) == 0 && (ext == ".tif" || ext == ".xml"))
                stale.push_back(entry.path());
        }
        for (const auto& p : stale) std::filesystem::remove(p, ec);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // All ranks read the metadata to determine the tile grid
    RasterReader reader(args.input);
    Tiler tiler(reader.rows(), reader.cols(), args.tileSize);
    auto allTiles = tiler.computeTiles();

    // Resolve index bands from the scene wavelengths (identically on every rank,
    // since all ranks read the same metadata) unless pinned on the CLI.
    resolveBandsFromWavelengths(args.bands, reader.wavelengths(), args.bandOverrides);

    // MPI Round-robin decomposition
    std::vector<TileBounds> myTiles;
    for (size_t i = rank; i < allTiles.size(); i += worldSize)
        myTiles.push_back(allTiles[i]);

    if (rank == 0) {
        std::cout << "MPI ranks=" << worldSize
                  << "  OMP threads/rank=" << args.threads
                  << "  Total tiles=" << allTiles.size() << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    // BUG #7 FIX: Each rank writes to a rank-specific temp CSV, then rank 0
    // merges all temp files into summary.csv after the barrier.
    // This avoids simultaneous file writes from different MPI processes.
    std::string rankCsvPath = args.output + "/summary_rank_" + std::to_string(rank) + ".csv";
    {
        // Truncate/create the rank file before the parallel loop
        std::ofstream f(rankCsvPath, std::ios::trunc);
    }

    // Per-rank mutex protects against concurrent OpenMP threads within this rank
    std::mutex rankMutex;

    // Optional cross-vendor GPU backend. Each rank initialises its own OpenCL
    // context (picking a GPU); on a single-GPU node the ranks share it and the
    // per-process dispatch mutex serialises them. Falls back to CPU per tile.
    bool use_gpu = false;
    if (args.mode == "gpu") {
#ifdef HSI_HAVE_OPENCL
        use_gpu = hsi::gpu::init_opencl();
        if (!use_gpu && rank == 0)
            std::cout << "Warning: GPU/OpenCL unavailable — using CPU for indices.\n";
#else
        if (rank == 0)
            std::cout << "Warning: built without OpenCL — using CPU for indices.\n";
#endif
    }

    std::atomic<bool> failed{false};
    std::string firstError;
    std::mutex errMutex;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) shared(myTiles, args, reader)
#endif
    for (size_t i = 0; i < myTiles.size(); ++i) {
      try {
        HyperspectralTile tile = reader.readTile(myTiles[i]);

        if (args.nodata)  NoDataHandler::apply(tile, args.nodataValue);
        if (args.badband) BadBandRemover::apply(tile);

        // BUG #4 FIX: SpectralFeatures BEFORE Normalizer (same fix as main.cpp)
        SpectralFeatureSet sf;
        if (args.spectral) {
            bool done_on_gpu = false;
#ifdef HSI_HAVE_OPENCL
            if (use_gpu)
                done_on_gpu = hsi::gpu::compute_spectral_features_gpu(tile, args.bands, sf);
#endif
            if (!done_on_gpu)
                sf = SpectralFeatures::compute(tile, args.bands);
        }
        if (args.normalize) Normalizer::apply(tile);

        TileWriteResult wr = TileWriter::writeGeoTIFF(tile, sf, args.output,
                                 myTiles[i].rowStart, myTiles[i].colStart);

        // Write to rank-specific temp CSV (mutex protects OpenMP threads only)
        TileWriter::appendSummaryCSV(tile, sf, wr, rankCsvPath,
                                     myTiles[i].rowStart, myTiles[i].colStart, rankMutex);
      } catch (const std::exception& e) {
        // Never let an exception escape the OpenMP region (UB). Record and go on.
        if (!failed.exchange(true)) {
            std::lock_guard<std::mutex> lock(errMutex);
            firstError = e.what();
        }
      }
    }

    if (failed) {
        std::cerr << "ERROR (rank " << rank << "): tile processing failed: "
                  << firstError << "\n";
    }

    // All ranks wait for tile processing to finish
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 merges all rank CSV files into summary.csv
    if (rank == 0) {
        const std::string finalCsv = args.output + "/summary.csv";
        std::ofstream out(finalCsv, std::ios::trunc);
        out << "tile_id,tile_path,epsg,xmin,ymin,xmax,ymax,"
               "ndvi,ndwi,ndbi,savi,evi,ndmi,msavi,"
               "first_index_band,geometry_wkt\n";

        for (int r = 0; r < worldSize; ++r) {
            std::string src = args.output + "/summary_rank_" + std::to_string(r) + ".csv";
            std::ifstream in(src);
            if (in.is_open()) {
                // Skip the header written by appendSummaryCSV in the rank file
                std::string line;
                std::getline(in, line); // discard header
                out << in.rdbuf();
                in.close();
                std::filesystem::remove(src);
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();

#ifdef HSI_HAVE_OPENCL
    if (use_gpu) hsi::gpu::teardown_opencl();
#endif

    if (rank == 0) {
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "RESULT mode=mpi" << (use_gpu ? "+gpu" : "")
                  << " ranks=" << worldSize
                  << " threads=" << args.threads
                  << " tiles=" << allTiles.size()
                  << " time_sec=" << elapsed << "\n";
    }

    // Combine per-rank failure flags so every rank returns the same exit code.
    int localFail = failed ? 1 : 0, globalFail = 0;
    MPI_Allreduce(&localFail, &globalFail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return globalFail ? 2 : 0;
}
