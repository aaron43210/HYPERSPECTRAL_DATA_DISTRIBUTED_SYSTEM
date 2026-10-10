#pragma once
// =============================================================================
// SAM.hpp  —  Spectral Angle Mapper (CPU reference implementation)
//
//   θ = arccos( (t·r) / (||t|| ||r||) )  over the B bands, radians, [0, π/2]
//   for non-negative reflectance spectra.
//
// SCALE-INVARIANT: the ×10000 EnMAP reflectance scale cancels in the ratio, so
// both the pixel spectrum `t` and the reference `r` are fed RAW (un-descaled).
//
// IMPORTANT (band-set contract): SAM does NOT run BadBandRemover or Normalizer —
// it operates on the raw spectra straight out of RasterReader (NoDataHandler is
// applied first only to map the nodata sentinel to 0 so an all-zero pixel is
// recognised as nodata). The reference vector therefore has exactly `bandCount`
// entries, one per original GDAL band, in band order. This keeps the reference
// aligned with the image band set with no band dropping.
//
// This CPU path mirrors the OpenCL kernel in gpu_backend_opencl.cpp EXACTLY
// (same float math, same clamp of cos to [-1,1] before acos, same nodata rule),
// so --mode gpu and --mode openmp/serial agree to within float rounding.
// =============================================================================

#include "RasterReader.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <stdexcept>

namespace sam {

/// Parse a comma-separated spectrum "v1,v2,...,vB" into a float vector.
inline std::vector<float> parseSpectrum(const std::string& csv) {
    std::vector<float> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // Trim surrounding whitespace.
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        out.push_back(std::stof(tok.substr(a, b - a + 1)));
    }
    return out;
}

/// Euclidean norm of the reference, accumulated in float to match the kernel.
inline float referenceNorm(const std::vector<float>& ref) {
    float s = 0.0f;
    for (float v : ref) s += v * v;
    return std::sqrt(s);
}

/// Compute the spectral angle (radians) for every pixel of a tile.
///   outAngle : size n, radians in [0,π]; -1.0f for nodata/zero-norm pixels.
///   outSim   : size n, cosine similarity clamped to [0,1]; -1.0f for nodata.
/// `tile.bands` must hold all B bands in band order (no bands dropped).
inline void computeTileCPU(const HyperspectralTile& tile,
                           const std::vector<float>& ref,
                           float refNorm,
                           std::vector<float>& outAngle,
                           std::vector<float>& outSim) {
    const int B = static_cast<int>(tile.bands.size());
    const size_t n = tile.bands.empty() ? 0 : tile.bands[0].values.size();
    outAngle.assign(n, -1.0f);
    outSim.assign(n, -1.0f);

    const long long nn = static_cast<long long>(n);
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < nn; ++i) {
        float dot = 0.0f, nt = 0.0f;
        for (int b = 0; b < B; ++b) {
            const float t  = tile.bands[b].values[i];
            const float rv = ref[b];
            dot += t * rv;
            nt  += t * t;
        }
        if (nt <= 0.0f || refNorm <= 0.0f) {
            outAngle[i] = -1.0f;
            outSim[i]   = -1.0f;
            continue;
        }
        float cosv = dot / (std::sqrt(nt) * refNorm);
        // Clamp BEFORE acos: fast-relaxed-math (and float rounding) can push the
        // ratio slightly past ±1, which would make acos() return NaN.
        cosv = std::fmax(-1.0f, std::fmin(1.0f, cosv));
        outAngle[i] = std::acos(cosv);
        outSim[i]   = cosv < 0.0f ? 0.0f : cosv;
    }
}

/// Derive the binary detection mask from the angle band (both paths identical).
///   mask : size n, 1 where 0<=angle<=threshold, 0 where angle>threshold,
///          255 (nodata) where angle<0 (input nodata).
inline void deriveMask(const std::vector<float>& angle, float threshold,
                       std::vector<unsigned char>& mask) {
    const size_t n = angle.size();
    mask.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (angle[i] < 0.0f)            mask[i] = 255;              // nodata
        else                            mask[i] = (angle[i] <= threshold) ? 1 : 0;
    }
}

} // namespace sam
