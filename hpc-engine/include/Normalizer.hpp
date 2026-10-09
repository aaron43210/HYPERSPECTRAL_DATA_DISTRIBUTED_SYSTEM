#pragma once
// =============================================================================
// Normalizer.hpp  —  Min-max normalisation per band (header-only)
// =============================================================================

#include "RasterReader.hpp"
#include <algorithm>

class Normalizer {
public:
    /// Min-max normalise each band into the [kEps, 1] range.
    /// Values of 0.0f (nodata) are ignored during min/max finding and
    /// remain 0.0f after normalisation.
    ///
    /// NOTE (per-tile stretch): min/max are computed *per tile*, so the same
    /// physical reflectance maps to different display values in different tiles.
    /// In a mosaic this shows up as tile-to-tile brightness seams. It only
    /// affects the raw visual bands (spectral indices are computed *before*
    /// normalisation and are left untouched), so index layers are unaffected.
    /// For a seamless raw mosaic, disable normalisation (physical reflectance is
    /// already mosaic-consistent) or add a global two-pass min/max.
    ///
    /// Works in-place on the tile.
    static void apply(HyperspectralTile& tile) {
        // Reserve exactly 0.0 for nodata. If we normalised into [0, 1], every
        // pixel equal to the band minimum would become 0.0 and then be
        // re-interpreted as nodata by the GeoTIFF writer and downstream readers.
        // Mapping valid data into [kEps, 1] keeps the darkest valid pixel
        // distinguishable from a true nodata (0.0) pixel.
        constexpr float kEps = 1e-6f;

        for (auto& band : tile.bands) {
            float minVal = 1e30f;
            float maxVal = -1e30f;

            const size_t sz = band.values.size();

#ifdef _OPENMP
            #pragma omp simd reduction(min:minVal) reduction(max:maxVal)
#endif
            for (size_t i = 0; i < sz; ++i) {
                float v = band.values[i];
                if (v != 0.0f) {
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }
            }

            // Normalise if there is a valid range
            if (minVal < maxVal && minVal != 1e30f) {
                const float range = maxVal - minVal;
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (size_t i = 0; i < sz; ++i) {
                    float v = band.values[i];
                    if (v != 0.0f) {
                        v = kEps + (1.0f - kEps) * ((v - minVal) / range);
                        if (v < kEps) v = kEps;
                        if (v > 1.0f) v = 1.0f;
                        band.values[i] = v;
                    }
                }
            }
        }
    }
};
