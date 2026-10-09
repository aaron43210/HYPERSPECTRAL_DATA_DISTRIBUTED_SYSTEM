#pragma once
// =============================================================================
// NoDataHandler.hpp  —  Replace nodata sentinel values with 0.0f (header-only)
// =============================================================================

#include "RasterReader.hpp"
#include <cmath>   // std::abs, std::isnan, std::isfinite (used below)

class NoDataHandler {
public:
    /// Replace all pixels equal to `nodataValue` with 0.0f in every band.
    /// Works in-place on the tile.
    static void apply(HyperspectralTile& tile,
                      float nodataValue = -9999.0f) noexcept
    {
        for (auto& band : tile.bands) {
            const float nd = (band.nodata_value != 0.0f)
                              ? band.nodata_value
                              : nodataValue;
            for (float& v : band.values) {
            // BUG #9 FIX: Use epsilon tolerance instead of exact equality.
            // GDAL stores nodata as double; converting to float32 can introduce
            // sub-ulp rounding so v == nd fails for true nodata pixels.
            if (std::abs(v - nd) < 1e-3f || std::isnan(v) || !std::isfinite(v))
                v = 0.0f;
        }
        }
    }
};
