#pragma once
// =============================================================================
// TileWriter.hpp  —  Write processed tiles + features to GeoTIFF
// =============================================================================

#include "RasterReader.hpp"
#include "SpectralFeatures.hpp"
#include <string>
#include <vector>
#include <mutex>

/// Result of writing a tile's GeoTIFF, needed to describe it in summary.csv.
struct TileWriteResult {
    std::string path;         ///< Absolute/relative path to the written .tif (empty if skipped)
    int firstIndexBand = 0;   ///< 1-based band number of the first (NDVI) index band; 0 if none
};

class TileWriter {
public:
    /// Build the deterministic output path for a tile (shared by writeGeoTIFF
    /// and appendSummaryCSV so the naming convention can't drift between them).
    static std::string tilePath(const std::string& outDir, int rowStart, int colStart);

    /// Write a processed tile and its spectral features to a GeoTIFF.
    /// Returns the output path and first index-band offset (empty path if the
    /// tile was fully purged and no file was written).
    static TileWriteResult writeGeoTIFF(const HyperspectralTile& tile,
                             const SpectralFeatureSet& sf,
                             const std::string& outDir,
                             int rowStart, int colStart);

    /// Append one row to the summary CSV at `csvPath` (thread-safe).
    /// Creates the file with a header on first call.
    /// Columns: tile_id,tile_path,epsg,xmin,ymin,xmax,ymax,
    ///          ndvi,ndwi,ndbi,savi,evi,ndmi,msavi,
    ///          first_index_band,geometry_wkt
    /// `first_index_band` is the 1-based GDAL band number of the first
    /// (NDVI) index band in the tile's GeoTIFF; the remaining six index
    /// bands follow immediately in the fixed order NDVI,NDWI,NDBI,SAVI,
    /// EVI,NDMI,MSAVI. Index bands are always the LAST 7 bands written,
    /// regardless of how many spectral bands BadBandRemover kept — so
    /// downstream readers never need to hard-code a band number.
    static void appendSummaryCSV(const HyperspectralTile& tile,
                                  const SpectralFeatureSet& sf,
                                  const TileWriteResult& writeResult,
                                  const std::string& csvPath,
                                  int rowStart, int colStart,
                                  std::mutex& csvMutex);
};

// -----------------------------------------------------------------------------
// SceneRaster  —  a single scene-sized GeoTIFF written window-by-window
//
// Whole-scene operations (SAM, PCA) produce one output raster covering the
// entire scene, but the engine still processes it tile-by-tile (never holding
// the cube). SceneRaster owns one LZW+TILED GeoTIFF and lets each worker write
// its tile's window; an internal mutex serialises the RasterIO writes because a
// shared GDALDataset handle is not thread-safe. Band descriptions, geotransform,
// CRS and a nodata value are set once at construction.
// -----------------------------------------------------------------------------
class SceneRaster {
public:
    enum class Type { Float32, Byte };

    SceneRaster(const std::string& path, int cols, int rows, int nbands,
                Type type, const double* geoTransform, const std::string& wkt,
                double nodata, const std::vector<std::string>& bandDescriptions);
    ~SceneRaster();

    SceneRaster(const SceneRaster&)            = delete;
    SceneRaster& operator=(const SceneRaster&) = delete;

    /// Write a float32 window into 0-based band `band0`. `data` is w*h row-major.
    void writeWindow(int band0, int colStart, int rowStart, int w, int h,
                     const float* data);
    /// Write a Byte window into 0-based band `band0`. `data` is w*h row-major.
    void writeWindow(int band0, int colStart, int rowStart, int w, int h,
                     const unsigned char* data);

    /// Flush and close the dataset (also called by the destructor).
    void finalize();

private:
    GDALDataset* ds_ = nullptr;
    std::mutex   mtx_;
};
