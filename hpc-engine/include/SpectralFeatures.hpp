#pragma once
// =============================================================================
// SpectralFeatures.hpp  —  Compute spectral indices
//   NDVI, NDWI, NDBI, SAVI, EVI, NDMI, MSAVI
// =============================================================================

#include "RasterReader.hpp"
#include <vector>
#include <cmath>

class SpectralFeatures {
public:
    /// Compute spectral features for a tile given a band configuration.
    static SpectralFeatureSet compute(const HyperspectralTile& tile, const BandConfig& config) {
        SpectralFeatureSet sf;
        if (tile.bands.empty()) return sf;

        const size_t numPixels = tile.bands[0].values.size();
        sf.ndvi.resize(numPixels, 0.0f);
        sf.ndwi.resize(numPixels, 0.0f);
        sf.ndbi.resize(numPixels, 0.0f);
        sf.savi.resize(numPixels, 0.0f);
        sf.evi.resize(numPixels, 0.0f);
        sf.ndmi.resize(numPixels, 0.0f);
        sf.msavi.resize(numPixels, 0.0f);
        sf.computed = true;

        // Find pointers to the required bands.
        // The bands vector might have been modified by BadBandRemover,
        // so we must search by original band index.
        const float* nir   = nullptr;
        const float* red   = nullptr;
        const float* green = nullptr;
        const float* swir  = nullptr;
        const float* blue  = nullptr;

        for (const auto& b : tile.bands) {
            if (b.index == config.nir)   nir   = b.values.data();
            if (b.index == config.red)   red   = b.values.data();
            if (b.index == config.green) green = b.values.data();
            if (b.index == config.swir)  swir  = b.values.data();
            if (b.index == config.blue)  blue  = b.values.data();
        }

        const float L = 0.5f; // SAVI soil-brightness correction factor

#if defined(_OPENMP) && !defined(_MSC_VER)
        #pragma omp simd
#endif
        for (size_t i = 0; i < numPixels; ++i) {
            // EnMAP L2A data is typically scaled by 10,000.
            // We must convert to 0.0 - 1.0 reflectance for indices with constants (EVI, SAVI, MSAVI)
            float n = nir ? (nir[i] / 10000.0f) : 0.0f;
            float r = red ? (red[i] / 10000.0f) : 0.0f;
            float g = green ? (green[i] / 10000.0f) : 0.0f;
            float s = swir ? (swir[i] / 10000.0f) : 0.0f;
            float b = blue ? (blue[i] / 10000.0f) : 0.0f;

            // A value of 0.0 marks nodata (set by NoDataHandler) OR a band that
            // BadBandRemover masked out for this tile. An index is only valid
            // where ALL of its operand bands are non-zero; otherwise leave it at
            // 0.0. Without this guard a half-nodata pixel (e.g. NIR=0, RED>0)
            // produces a spurious index of ±1 that pollutes the per-tile means
            // written to summary.csv and shows up as fake data in the raster.
            auto clamp = [](float v) { return std::fmax(-1.0f, std::fmin(1.0f, v)); };

            if (nir && red && n != 0.0f && r != 0.0f) {
                float denom = n + r;
                if (denom != 0.0f) sf.ndvi[i] = clamp((n - r) / denom);

                float sdenom = n + r + L;
                if (sdenom != 0.0f) sf.savi[i] = clamp(((n - r) / sdenom) * (1.0f + L));

                // MSAVI (iteration-free form):
                float term = 2.0f * n + 1.0f;
                float disc = term * term - 8.0f * (n - r);
                if (disc >= 0.0f) sf.msavi[i] = clamp((term - std::sqrt(disc)) / 2.0f);
            }
            if (green && nir && g != 0.0f && n != 0.0f) {
                float denom = g + n;
                if (denom != 0.0f) sf.ndwi[i] = clamp((g - n) / denom);
            }
            if (swir && nir && s != 0.0f && n != 0.0f) {
                float denom = s + n;                           // NDBI and NDMI share n+s
                if (denom != 0.0f) sf.ndbi[i] = clamp((s - n) / denom);
                if (denom != 0.0f) sf.ndmi[i] = clamp((n - s) / denom);
            }
            if (nir && red && blue && n != 0.0f && r != 0.0f && b != 0.0f) {
                float edenom = n + 6.0f * r - 7.5f * b + 1.0f;
                if (edenom != 0.0f) sf.evi[i] = clamp(2.5f * (n - r) / edenom);
            }
        }

        return sf;
    }
};
