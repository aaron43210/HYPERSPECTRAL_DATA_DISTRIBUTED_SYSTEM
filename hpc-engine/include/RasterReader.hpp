#pragma once
// =============================================================================
// RasterReader.hpp  —  Core hyperspectral data structures + reader class
//
// Design rules:
//   - RasterReader opens the file once on construction to read metadata only.
//   - readTile() opens a *fresh* GDALDataset handle each call, making it
//     safe to call from any thread without additional locking.
//   - HyperspectralTile is a value type: copyable, movable, no heap sharing.
// =============================================================================

#include <string>
#include <vector>
#include <stdexcept>

// Forward declaration — avoids pulling gdal_priv.h into every TU that
// includes this header.
class GDALDataset;

// -----------------------------------------------------------------------------
// Band  — one spectral band within a tile
// -----------------------------------------------------------------------------
struct Band {
    int   index;               ///< 1-based GDAL band number
    float wavelength_nm;       ///< Centre wavelength in nm; 0.0 if unknown
    float nodata_value;        ///< Sentinel value used for missing pixels
    /// Pixel values in row-major order: values[row * tile_width + col]
    std::vector<float> values;
};

// -----------------------------------------------------------------------------
// SpectralFeatureSet  — derived spectral indices per pixel
// -----------------------------------------------------------------------------
struct SpectralFeatureSet {
    std::vector<float> ndvi;   ///< Normalised Difference Vegetation Index
    std::vector<float> ndwi;   ///< Normalised Difference Water Index
    std::vector<float> ndbi;   ///< Normalised Difference Built-up Index
    std::vector<float> savi;   ///< Soil-Adjusted Vegetation Index
    std::vector<float> evi;    ///< Enhanced Vegetation Index
    std::vector<float> ndmi;   ///< Normalised Difference Moisture Index
    std::vector<float> msavi;  ///< Modified Soil-Adjusted Vegetation Index
    bool computed = false;

    /// Number of index bands written per tile, in fixed order:
    /// NDVI, NDWI, NDBI, SAVI, EVI, NDMI, MSAVI.
    static constexpr int kCount = 7;
};

// -----------------------------------------------------------------------------
// BandConfig  — 1-based GDAL band indices for spectral index computation
// -----------------------------------------------------------------------------
// Defaults are the bands NEAREST the canonical centre wavelengths in a standard
// 224-band EnMAP L2A cube (VNIR+SWIR, ~418–2445 nm). They are used only as a
// last-resort fallback: when the scene exposes per-band wavelengths (EnMAP ENVI
// and GeoTIFF both do) the engine auto-selects the nearest band per role at
// runtime (see BandResolver.hpp), so these values are correct even for a scene
// with a slightly different band table.
//
// NOTE: the previous hard-coded values {nir:88,red:40,green:25,swir:160,blue:10}
// were WRONG for real EnMAP wavelengths — band 88 ≈ 936 nm (NIR water-vapour),
// band 40 ≈ 617 nm (orange), etc. — which systematically biased every index.
// Validated against the shipped scene's HDR: 860→75, 660→47, 560→30, 1640→151,
// 480→14.
struct BandConfig {
    int nir   = 75;    ///< Near-infrared  (~863 nm, EnMAP band 75)
    int red   = 47;    ///< Red            (~660 nm, EnMAP band 47)
    int green = 30;    ///< Green          (~561 nm, EnMAP band 30)
    int swir  = 151;   ///< Short-wave IR  (~1642 nm, EnMAP band 151)
    int blue  = 14;    ///< Blue           (~482 nm, EnMAP band 14) — used by EVI
};

// -----------------------------------------------------------------------------
// TileBounds  — pixel-space extent of a tile within the scene
// -----------------------------------------------------------------------------
struct TileBounds {
    int rowStart;   ///< Inclusive row offset from scene top-left
    int colStart;   ///< Inclusive col offset from scene top-left
    int rowEnd;     ///< Exclusive row end
    int colEnd;     ///< Exclusive col end

    int width()  const noexcept { return colEnd - colStart; }
    int height() const noexcept { return rowEnd - rowStart; }
};

// -----------------------------------------------------------------------------
// HyperspectralTile  — fundamental processing unit
// -----------------------------------------------------------------------------
struct HyperspectralTile {
    TileBounds bounds;

    int width()  const noexcept { return bounds.width();  }
    int height() const noexcept { return bounds.height(); }

    std::vector<Band> bands;

    /// GDAL GeoTransform for the tile origin (adjusted from scene GT)
    double geoTransform[6] = {};

    /// CRS as WKT string
    std::string wkt;

    /// EPSG authority code extracted from CRS WKT; 0 if unknown
    int epsg = 0;
};

// -----------------------------------------------------------------------------
// RasterReader
// -----------------------------------------------------------------------------
/// Opens a hyperspectral raster file and provides thread-safe tile reading.
///
/// Thread safety: Multiple threads may call readTile() concurrently on the
/// same RasterReader instance because each call opens its own GDALDataset
/// handle internally.
class RasterReader {
public:
    /// Open a raster file and read its metadata.
    /// @param path         Absolute or relative path to a GeoTIFF / ENVI file.
    /// @param wavelengths  Optional per-band centre wavelengths (nm, 0-based).
    ///                     If empty, Band::wavelength_nm is set to 0.
    /// @throws std::runtime_error if GDAL cannot open the file.
    explicit RasterReader(const std::string& path,
                          const std::vector<float>& wavelengths = {});

    ~RasterReader() = default;

    // Non-copyable — holds file state; copy would be surprising
    RasterReader(const RasterReader&)            = delete;
    RasterReader& operator=(const RasterReader&) = delete;

    // Movable
    RasterReader(RasterReader&&)            = default;
    RasterReader& operator=(RasterReader&&) = default;

    int rows()      const noexcept { return rows_;  }
    int cols()      const noexcept { return cols_;  }
    int bandCount() const noexcept { return bands_; }
    int epsg()      const noexcept { return epsg_;  }

    /// Per-band centre wavelengths in nm (0-based, size == bandCount()).
    /// Populated from the file's band metadata when the caller did not supply
    /// an explicit list; entries are 0.0f where a wavelength is unknown.
    const std::vector<float>& wavelengths() const noexcept { return wavelengths_; }

    const std::string& path()       const noexcept { return path_; }
    const std::string& projection() const noexcept { return wkt_;  }
    const double* geoTransform()    const noexcept { return geoTransform_; }
    float nodataValue()             const noexcept { return nodataValue_; }

    /// Read a tile from the raster.
    /// Opens a new GDALDataset handle — safe to call from any thread.
    /// @throws std::runtime_error on I/O failure.
    HyperspectralTile readTile(const TileBounds& tb) const;

private:
    std::string        path_;
    std::vector<float> wavelengths_;
    int   rows_  = 0;
    int   cols_  = 0;
    int   bands_ = 0;
    int   epsg_  = 0;
    float nodataValue_ = -9999.0f;
    double geoTransform_[6] = {};
    std::string wkt_;
};
