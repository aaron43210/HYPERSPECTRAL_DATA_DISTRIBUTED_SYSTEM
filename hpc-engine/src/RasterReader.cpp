// =============================================================================
// RasterReader.cpp  —  Implementation of the GDAL-based reader
// =============================================================================

#include "RasterReader.hpp"
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ogr_spatialref.h>
#include <stdexcept>
#include <iostream>
#include <cctype>
#include <cstdlib>

namespace {
// Parse a centre wavelength (nm) from a GDAL band's metadata/description.
// Handles the common encodings produced by GDAL's ENVI and GeoTIFF drivers:
//   * per-band metadata item "wavelength" (optional "wavelength_units")
//   * a band description such as "Band 10 (463.55 nm)".
// Returns 0.0f when no wavelength can be recovered.
float parseBandWavelengthNm(GDALRasterBand* rb) {
    if (!rb) return 0.0f;

    auto toNm = [](double value, const char* units) -> float {
        if (value <= 0.0) return 0.0f;
        // Default assumption is nm; convert only when units say micrometres.
        if (units) {
            std::string u(units);
            for (auto& c : u) c = static_cast<char>(std::tolower(c));
            if (u.find("micro") != std::string::npos || u == "um" || u == "µm")
                return static_cast<float>(value * 1000.0);
        }
        // A bare value < 100 is almost certainly micrometres (EnMAP: 0.42–2.45).
        if (value < 100.0) return static_cast<float>(value * 1000.0);
        return static_cast<float>(value);
    };

    if (const char* w = rb->GetMetadataItem("wavelength")) {
        const char* units = rb->GetMetadataItem("wavelength_units");
        float nm = toNm(std::atof(w), units);
        if (nm > 0.0f) return nm;
    }

    // Fall back to the human-readable description: "...(463.55 nm)" or similar.
    if (const char* desc = rb->GetDescription()) {
        std::string d(desc);
        std::string lower = d;
        for (auto& c : lower) c = static_cast<char>(std::tolower(c));
        size_t nmPos = lower.find("nm");
        size_t umPos = (nmPos == std::string::npos)
                           ? std::min(lower.find("um"), lower.find("micro"))
                           : std::string::npos;
        size_t unitPos = (nmPos != std::string::npos) ? nmPos : umPos;
        if (unitPos != std::string::npos) {
            // Scan backwards from the unit token to grab the preceding number.
            size_t end = unitPos;
            while (end > 0 && std::isspace(static_cast<unsigned char>(d[end - 1]))) --end;
            size_t start = end;
            while (start > 0 &&
                   (std::isdigit(static_cast<unsigned char>(d[start - 1])) ||
                    d[start - 1] == '.' || d[start - 1] == '+' || d[start - 1] == '-'))
                --start;
            if (start < end) {
                double v = std::atof(d.substr(start, end - start).c_str());
                return toNm(v, (nmPos != std::string::npos) ? "nm" : "um");
            }
        }
    }
    return 0.0f;
}
} // namespace

RasterReader::RasterReader(const std::string& path,
                           const std::vector<float>& wavelengths)
    : path_(path), wavelengths_(wavelengths)
{
    GDALAllRegister();
    // Disable noisy GDAL warnings/errors that might mess up our stdout
    CPLPushErrorHandler(CPLQuietErrorHandler);

    GDALDataset* ds = (GDALDataset*)GDALOpen(path.c_str(), GA_ReadOnly);
    if (!ds) {
        CPLPopErrorHandler();
        throw std::runtime_error("Cannot open: " + path);
    }
    
    rows_  = ds->GetRasterYSize();
    cols_  = ds->GetRasterXSize();
    bands_ = ds->GetRasterCount();
    
    if (ds->GetGeoTransform(geoTransform_) != CE_None) {
        // Fallback to identity transform
        geoTransform_[0] = 0.0; geoTransform_[1] = 1.0; geoTransform_[2] = 0.0;
        geoTransform_[3] = 0.0; geoTransform_[4] = 0.0; geoTransform_[5] = 1.0;
    }
    
    const char* proj = ds->GetProjectionRef();
    if (proj && proj[0] != '\0') {
        wkt_ = proj;
        // Extract EPSG authority code from the WKT so downstream
        // components (CSV, Spark) can perform CRS-aware spatial ops.
        OGRSpatialReference srs;
        const char* wktCStr = wkt_.c_str();
        if (srs.importFromWkt(&wktCStr) == OGRERR_NONE) {
            srs.AutoIdentifyEPSG();
            const char* code = srs.GetAuthorityCode(nullptr);
            if (code) {
                try { epsg_ = std::stoi(code); } catch (...) { epsg_ = 0; }
            } else {
                // Fallback for ENVI / ENMAP files that are UTM but lack a top-level EPSG node.
                int north = 1;
                int zone = srs.GetUTMZone(&north);
                if (zone > 0) {
                    epsg_ = (north ? 32600 : 32700) + zone;
                }
            }
        }
    }
    
    // Read nodata value from band 1 (assume uniform across bands)
    if (bands_ > 0) {
        auto rb = ds->GetRasterBand(1);
        int hasNodata = 0;
        double nd = rb->GetNoDataValue(&hasNodata);
        if (hasNodata) {
            nodataValue_ = static_cast<float>(nd);
        }
    }

    // Auto-detect per-band centre wavelengths from the file metadata unless the
    // caller supplied an explicit list. This lets the engine pick the correct
    // NIR/Red/Green/SWIR/Blue bands for ANY cube (not just a hard-coded EnMAP
    // layout) — see BandResolver.hpp. Entries are 0.0f where unknown.
    if (wavelengths_.empty() && bands_ > 0) {
        wavelengths_.assign(bands_, 0.0f);
        for (int b = 1; b <= bands_; ++b) {
            wavelengths_[b - 1] = parseBandWavelengthNm(ds->GetRasterBand(b));
        }
    }

    GDALClose(ds);
    CPLPopErrorHandler();
}

HyperspectralTile RasterReader::readTile(const TileBounds& tb) const {
    // Open a fresh dataset handle — GDAL is NOT thread-safe on shared handles
    CPLPushErrorHandler(CPLQuietErrorHandler);
    GDALDataset* ds = (GDALDataset*)GDALOpen(path_.c_str(), GA_ReadOnly);
    if (!ds) {
        CPLPopErrorHandler();
        throw std::runtime_error("Cannot re-open: " + path_);
    }

    HyperspectralTile tile;
    tile.bounds = tb;
    tile.epsg   = epsg_;  // propagate EPSG to tile so TileWriter can write it
    std::copy(std::begin(geoTransform_), std::end(geoTransform_), tile.geoTransform);

    // Adjust tile geoTransform to tile origin (handles rotated rasters too)
    tile.geoTransform[0] += tb.colStart * geoTransform_[1] + tb.rowStart * geoTransform_[2];
    tile.geoTransform[3] += tb.colStart * geoTransform_[4] + tb.rowStart * geoTransform_[5];
    tile.wkt = wkt_;

    int w = tile.width();
    int h = tile.height();

    for (int b = 1; b <= bands_; ++b) {
        Band band;
        band.index = b;
        band.nodata_value = nodataValue_;
        band.wavelength_nm = (b - 1 < (int)wavelengths_.size()) ? wavelengths_[b - 1] : 0.0f;
        band.values.resize(w * h);
        
        auto rb = ds->GetRasterBand(b);
        if (rb) {
            CPLErr err = rb->RasterIO(
                GF_Read,
                tb.colStart, tb.rowStart, w, h,
                band.values.data(), w, h,
                GDT_Float32, 0, 0
            );
            if (err != CE_None) {
                // A failed read must not masquerade as a tile of valid zeros
                // (which the rest of the pipeline would treat as nodata). Fail
                // loudly so the orchestrator sees a non-zero exit, not silent
                // holes in the mosaic.
                GDALClose(ds);
                CPLPopErrorHandler();
                throw std::runtime_error("RasterIO read failed for band " +
                                         std::to_string(b) + " of " + path_);
            }
        }
        tile.bands.push_back(std::move(band));
    }
    
    GDALClose(ds);
    CPLPopErrorHandler();
    return tile;
}
