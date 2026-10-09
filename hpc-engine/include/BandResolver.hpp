#pragma once
// =============================================================================
// BandResolver.hpp  —  Resolve NIR/Red/Green/SWIR/Blue band indices from the
//                      scene's per-band wavelengths (header-only, shared by
//                      main.cpp and main_mpi.cpp).
//
// Why this exists:
//   The spectral indices are only meaningful if NIR/Red/Green/SWIR/Blue point
//   at the right physical wavelengths. Hard-coding GDAL band numbers is fragile
//   — the correct band for ~860 nm differs between sensors and even between
//   EnMAP processor versions. When the file exposes per-band wavelengths
//   (RasterReader auto-detects them), we pick the nearest band per role at
//   runtime. Explicit --nir-band/... flags always win; the BandConfig defaults
//   are the last-resort fallback for a cube with no wavelength metadata.
// =============================================================================

#include "RasterReader.hpp"
#include <vector>
#include <cmath>

/// Canonical centre wavelengths (nm) for each spectral role.
namespace BandTargets {
    constexpr float NIR   = 860.0f;
    constexpr float RED   = 660.0f;
    constexpr float GREEN = 560.0f;
    constexpr float SWIR  = 1640.0f;
    constexpr float BLUE  = 480.0f;
}

/// Which roles the user pinned on the command line (those are never overridden).
struct BandOverrides {
    bool nir = false, red = false, green = false, swir = false, blue = false;
};

/// 1-based band whose wavelength is nearest `target_nm`; -1 if none known.
inline int nearestBand(const std::vector<float>& wl, float target_nm) {
    int best = -1;
    float bestDist = 1e30f;
    for (std::size_t i = 0; i < wl.size(); ++i) {
        if (wl[i] <= 0.0f) continue;              // unknown wavelength
        const float d = std::fabs(wl[i] - target_nm);
        if (d < bestDist) { bestDist = d; best = static_cast<int>(i) + 1; }
    }
    return best;
}

/// Fill the non-overridden roles in `b` from the scene wavelengths. No-op when
/// no wavelengths are known (keeps the BandConfig defaults).
inline void resolveBandsFromWavelengths(BandConfig& b,
                                        const std::vector<float>& wl,
                                        const BandOverrides& ov) {
    bool anyKnown = false;
    for (float w : wl) if (w > 0.0f) { anyKnown = true; break; }
    if (!anyKnown) return;

    if (!ov.nir)   { int x = nearestBand(wl, BandTargets::NIR);   if (x > 0) b.nir   = x; }
    if (!ov.red)   { int x = nearestBand(wl, BandTargets::RED);   if (x > 0) b.red   = x; }
    if (!ov.green) { int x = nearestBand(wl, BandTargets::GREEN); if (x > 0) b.green = x; }
    if (!ov.swir)  { int x = nearestBand(wl, BandTargets::SWIR);  if (x > 0) b.swir  = x; }
    if (!ov.blue)  { int x = nearestBand(wl, BandTargets::BLUE);  if (x > 0) b.blue  = x; }
}
