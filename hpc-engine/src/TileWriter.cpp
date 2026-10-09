// =============================================================================
// TileWriter.cpp  —  Implementation for writing processed tiles
// =============================================================================

#include "TileWriter.hpp"
#include <gdal_priv.h>
#include <cpl_string.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <stdexcept>

std::string TileWriter::tilePath(const std::string& outDir, int rowStart, int colStart)
{
    // Naming convention: tile_r0000_c0000.tif
    std::ostringstream ss;
    ss << outDir << "/tile_r"
       << std::setw(4) << std::setfill('0') << rowStart
       << "_c"
       << std::setw(4) << std::setfill('0') << colStart
       << ".tif";
    return ss.str();
}

TileWriteResult TileWriter::writeGeoTIFF(const HyperspectralTile& tile,
                              const SpectralFeatureSet& sf,
                              const std::string& outDir,
                              int rowStart, int colStart)
{
    TileWriteResult result;
    std::string outPath = tilePath(outDir, rowStart, colStart);

    GDALAllRegister();
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) {
        throw std::runtime_error("GTiff driver not available");
    }

    int w = tile.width();
    int h = tile.height();

    // Total bands = original bands + optional spectral features.
    // Index bands are always written as the trailing SpectralFeatureSet::kCount
    // bands, in fixed order (NDVI,NDWI,NDBI,SAVI,EVI,NDMI,MSAVI), regardless of
    // how many spectral bands BadBandRemover left — so their position is always
    // (totalBands - kCount + 1) and never needs to be hard-coded downstream.
    int numSpectral = sf.computed ? SpectralFeatureSet::kCount : 0;
    int totalBands = (int)tile.bands.size() + numSpectral;

    if (totalBands == 0) {
        // Tile was completely purged (e.g. 100% nodata edge tile).
        // Skip creating an empty file to save disk space and prevent GDAL errors.
        return result; // path stays empty, firstIndexBand stays 0
    }

    char** papszOptions = nullptr;
    papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "LZW");
    papszOptions = CSLSetNameValue(papszOptions, "TILED", "YES");

    GDALDataset* outDs = driver->Create(outPath.c_str(), w, h, totalBands, GDT_Float32, papszOptions);
    CSLDestroy(papszOptions);
    if (!outDs) {
        throw std::runtime_error("Cannot create output tile: " + outPath);
    }

    // Set georeferencing
    outDs->SetGeoTransform(const_cast<double*>(tile.geoTransform));
    if (!tile.wkt.empty()) {
        outDs->SetProjection(tile.wkt.c_str());
    }

    // Write original (processed) bands
    for (size_t i = 0; i < tile.bands.size(); ++i) {
        int bIdx = (int)i + 1;
        auto outBand = outDs->GetRasterBand(bIdx);
        CPLErr err = outBand->RasterIO(GF_Write, 0, 0, w, h,
                          (void*)tile.bands[i].values.data(),
                          w, h, GDT_Float32, 0, 0);
        if (err != CE_None) {
            GDALClose(outDs);
            throw std::runtime_error("RasterIO write failed for band " +
                                     std::to_string(bIdx) + " of " + outPath);
        }
        outBand->SetNoDataValue(0.0);  // nodata was replaced with 0 by NoDataHandler
    }

    // Write spectral index bands (always the trailing kCount bands).
    if (sf.computed) {
        int bIdx = (int)tile.bands.size() + 1;
        result.firstIndexBand = bIdx;
        auto writeIndex = [&](const std::vector<float>& v) {
            CPLErr err = outDs->GetRasterBand(bIdx)->RasterIO(GF_Write, 0, 0, w, h,
                (void*)v.data(), w, h, GDT_Float32, 0, 0);
            if (err != CE_None) {
                GDALClose(outDs);
                throw std::runtime_error("RasterIO write failed for index band " +
                                         std::to_string(bIdx) + " of " + outPath);
            }
            ++bIdx;
        };
        writeIndex(sf.ndvi);
        writeIndex(sf.ndwi);
        writeIndex(sf.ndbi);
        writeIndex(sf.savi);
        writeIndex(sf.evi);
        writeIndex(sf.ndmi);
        writeIndex(sf.msavi);

        for (int i = (int)tile.bands.size() + 1; i <= totalBands; ++i) {
            outDs->GetRasterBand(i)->SetNoDataValue(0.0);
        }
    }

    GDALClose(outDs);
    result.path = outPath;
    return result;
}

void TileWriter::appendSummaryCSV(const HyperspectralTile& tile,
                                   const SpectralFeatureSet& sf,
                                   const TileWriteResult& writeResult,
                                   const std::string& csvPath,
                                   int rowStart, int colStart,
                                   std::mutex& csvMutex)
{
    if (writeResult.path.empty()) return; // tile was fully purged, no file written

    // Compute bounding box from geotransform
    // geoTransform[0] = xmin (top-left X), geoTransform[3] = ymax (top-left Y)
    // geoTransform[1] = pixel width,       geoTransform[5] = pixel height (negative)
    double xmin = tile.geoTransform[0];
    double ymax = tile.geoTransform[3];
    double xmax = xmin + tile.geoTransform[1] * tile.width();
    double ymin = ymax + tile.geoTransform[5] * tile.height(); // geoTransform[5] < 0

    // Compute mean spectral indices, ignoring nodata (0.0) pixels
    auto nanmean = [](const std::vector<float>& v) -> double {
        double s = 0.0; size_t n = 0;
        for (float x : v) {
            if (std::isfinite(x) && x != 0.0f) { s += x; ++n; }
        }
        return n > 0 ? s / n : 0.0;
    };

    double mean_ndvi  = sf.computed ? nanmean(sf.ndvi)  : 0.0;
    double mean_ndwi  = sf.computed ? nanmean(sf.ndwi)  : 0.0;
    double mean_ndbi  = sf.computed ? nanmean(sf.ndbi)  : 0.0;
    double mean_savi  = sf.computed ? nanmean(sf.savi)  : 0.0;
    double mean_evi   = sf.computed ? nanmean(sf.evi)   : 0.0;
    double mean_ndmi  = sf.computed ? nanmean(sf.ndmi)  : 0.0;
    double mean_msavi = sf.computed ? nanmean(sf.msavi) : 0.0;

    // Build WKT polygon: exterior ring in CCW order (OGC standard)
    // Vertices: LL → LR → UR → UL → LL
    std::ostringstream wkt;
    wkt << std::fixed << std::setprecision(8)
        << "POLYGON(("
        << xmin << " " << ymin << ","
        << xmax << " " << ymin << ","
        << xmax << " " << ymax << ","
        << xmin << " " << ymax << ","
        << xmin << " " << ymin << "))";

    std::ostringstream tileId;
    tileId << "tile_r" << std::setw(4) << std::setfill('0') << rowStart
           << "_c"     << std::setw(4) << std::setfill('0') << colStart;

    std::lock_guard<std::mutex> lock(csvMutex);
    std::ofstream f(csvPath, std::ios::app);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open summary CSV for writing: " + csvPath);
    }

    // Determine whether the existing file has a header by attempting to read the
    // first line.  This is safe under the mutex so there is no TOCTOU race.
    // Cases:
    //   - Empty / new file  → peek == EOF  → write header, no line to skip.
    //   - File with header  → first line matches column names → skip it.
    //   - File w/o header  → first line is a data row            → do NOT skip.
    bool fileHasHeader = false;
    {
        std::ifstream in(csvPath);
        if (in.is_open() && in.peek() != std::ifstream::traits_type::eof()) {
            std::string firstLine;
            std::getline(in, firstLine);
            fileHasHeader = (firstLine.find("tile_id,tile_path") == 0);
        }
    }

    // BUG #15 FIX: Write 'epsg' column so Spark can call ST_Transform correctly.
    // first_index_band: 1-based band number of NDVI in the tile's GeoTIFF; the
    // remaining 6 indices (NDWI,NDBI,SAVI,EVI,NDMI,MSAVI) follow immediately.
    // Readers must use this offset instead of a hard-coded band number, since
    // BadBandRemover can change how many spectral bands precede the indices.
    if (!fileHasHeader)
        f << "tile_id,tile_path,epsg,xmin,ymin,xmax,ymax,"
             "ndvi,ndwi,ndbi,savi,evi,ndmi,msavi,"
             "first_index_band,geometry_wkt\n";

    f << std::fixed << std::setprecision(8)
      << tileId.str() << ","
      << "\"" << writeResult.path << "\"" << ","
      << tile.epsg << ","
      << xmin << "," << ymin << "," << xmax << "," << ymax << ","
      << mean_ndvi << "," << mean_ndwi << "," << mean_ndbi << ","
      << mean_savi << "," << mean_evi << "," << mean_ndmi << "," << mean_msavi << ","
      << writeResult.firstIndexBand << ","
      << "\"" << wkt.str() << "\"\n";
}
