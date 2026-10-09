#pragma once
// =============================================================================
// TileWriter.hpp  —  Write processed tiles + features to GeoTIFF
// =============================================================================

#include "RasterReader.hpp"
#include "SpectralFeatures.hpp"
#include <string>
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
