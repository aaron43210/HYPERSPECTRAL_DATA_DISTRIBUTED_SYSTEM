#pragma once
// =============================================================================
// BadBandRemover.hpp  —  Remove bands with excessive NoData or zero variance
// =============================================================================

#include "RasterReader.hpp"
#include <algorithm>
#include <cmath>

class BadBandRemover {
public:
    /// Remove bands that are > 50% nodata (0.0f after NoDataHandler)
    /// or have zero variance.
    /// Works in-place on the tile.
    static void apply(HyperspectralTile& tile) {
        if (tile.bands.empty()) return;

        const size_t totalPixels = tile.width() * tile.height();
        if (totalPixels == 0) return;

        auto isBad = [totalPixels](const Band& band) {
            size_t zeros = 0;
            double sum   = 0.0;  // BUG #8 FIX: use double to avoid float32 cancellation
            double sumSq = 0.0;

            for (float v : band.values) {
                if (v == 0.0f) {
                    zeros++;
                } else {
                    sum   += static_cast<double>(v);
                    sumSq += static_cast<double>(v) * static_cast<double>(v);
                }
            }

            // > 50% nodata
            if (zeros > totalPixels / 2) return true;

            // Zero variance (all non-zero pixels are identical)
            const size_t valid = totalPixels - zeros;
            if (valid > 1) {
                // Two-pass variance in double to prevent float32 cancellation
                // on high-DN sensor data (e.g. raw EnMAP counts up to 65535).
                const double mean     = sum / static_cast<double>(valid);
                const double variance = (sumSq / static_cast<double>(valid)) - (mean * mean);
                if (variance <= 1e-7) return true;
            }

            return false;
        };

        // BUG FIX (uniform band layout): MASK (zero-fill) bad bands instead of
        // ERASING them. Erasing changed the band count on a per-tile basis
        // (edge tiles with lots of nodata lose many bands), which silently
        // broke the VRT mosaic step in the QGIS plugin: gdalbuildvrt skips any
        // tile whose band count differs from the first one it sees, leaving
        // holes — or selecting the wrong band — in the merged NDVI/NDWI/...
        // layers. Zero-filling keeps an identical band layout across EVERY tile
        // (all raw bands + 7 trailing index bands), so the mosaic is always
        // complete and the trailing-index-band offset is constant. A masked
        // band reads as 0.0 (nodata) everywhere downstream, exactly like a
        // fully-nodata band, so SpectralFeatures/Normalizer already handle it.
        for (auto& band : tile.bands) {
            if (isBad(band)) {
                std::fill(band.values.begin(), band.values.end(), 0.0f);
            }
        }
    }
};
