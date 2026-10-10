#pragma once
// =============================================================================
// PCA.hpp  —  Global (whole-scene) Principal Component Analysis helpers
//
// Design (never holds the cube in memory):
//   Pass 1a  accumulate per-band means          (float64, over valid pixels)
//   Pass 1b  accumulate Σ(x-μ)(x-μ)^T            (float64, EXACT-MEAN two-sub-
//            pass form — μ is known before the deviations are formed, which
//            avoids the catastrophic cancellation of the E[xx]-μμ^T shortcut on
//            EnMAP reflectance ×10000, where the mean is large and the variance
//            small).
//   eigensolve  compact cyclic Jacobi (below), eigenpairs sorted DESCENDING,
//               with the sklearn svd_flip deterministic sign convention.
//   Pass 2   project y = V_k^T (x-μ)  (covariance) or on standardised data
//            (correlation).
//
// `covariance` (default) divides by (N-1); `correlation` divides the centred
// bands by the per-band std first (so the matrix diagonal is 1).
//
// Excluded bands (--pca-exclude-bands) are dropped from the matrix entirely:
// everything here works on the kept-band subset of size Bk.
// =============================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <set>

namespace pca {

// -----------------------------------------------------------------------------
// Parse "128-138,160-170" (1-based, inclusive) into a set of band numbers.
// -----------------------------------------------------------------------------
inline std::set<int> parseExcludeBands(const std::string& spec) {
    std::set<int> out;
    if (spec.empty()) return out;
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = tok.find_last_not_of(" \t\r\n");
        tok = tok.substr(a, b - a + 1);
        size_t dash = tok.find('-');
        if (dash == std::string::npos) {
            out.insert(std::stoi(tok));
        } else {
            int lo = std::stoi(tok.substr(0, dash));
            int hi = std::stoi(tok.substr(dash + 1));
            if (lo > hi) std::swap(lo, hi);
            for (int b1 = lo; b1 <= hi; ++b1) out.insert(b1);
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Cyclic Jacobi eigensolver for a real symmetric matrix (double precision).
//   A : n×n row-major symmetric input (not modified).
//   Returns eigenvalues (desc) in `evals` and eigenvectors as COLUMNS of `evecs`
//   (row-major n×n): the i-th eigenvector is evecs[row*n + i].
// -----------------------------------------------------------------------------
inline void jacobiEigenSymmetric(const std::vector<double>& A, int n,
                                 std::vector<double>& evals,
                                 std::vector<double>& evecs) {
    std::vector<double> a = A;                 // working copy, mutated in place
    std::vector<double> v(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) v[static_cast<size_t>(i) * n + i] = 1.0;

    auto at = [&](int i, int j) -> double& { return a[static_cast<size_t>(i) * n + j]; };
    auto vt = [&](int i, int j) -> double& { return v[static_cast<size_t>(i) * n + j]; };

    const int    maxSweeps = 100;
    const double eps = 1e-300;
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        // Sum of squares of off-diagonal entries → convergence test.
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += at(p, q) * at(p, q);
        if (off <= eps) break;

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = at(p, q);
                if (std::fabs(apq) <= 0.0) continue;
                const double app = at(p, p);
                const double aqq = at(q, q);
                // Rotation angle (Golub & Van Loan, symmetric Schur 2x2).
                const double tau = (aqq - app) / (2.0 * apq);
                double t;
                if (tau >= 0.0) t =  1.0 / (tau + std::sqrt(1.0 + tau * tau));
                else            t = -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                // Apply the rotation to rows/cols p and q of A.
                for (int i = 0; i < n; ++i) {
                    const double aip = at(i, p);
                    const double aiq = at(i, q);
                    at(i, p) = c * aip - s * aiq;
                    at(i, q) = s * aip + c * aiq;
                }
                for (int i = 0; i < n; ++i) {
                    const double api = at(p, i);
                    const double aqi = at(q, i);
                    at(p, i) = c * api - s * aqi;
                    at(q, i) = s * api + c * aqi;
                }
                // Accumulate the rotation into the eigenvector matrix.
                for (int i = 0; i < n; ++i) {
                    const double vip = vt(i, p);
                    const double viq = vt(i, q);
                    vt(i, p) = c * vip - s * viq;
                    vt(i, q) = s * vip + c * viq;
                }
            }
        }
    }

    // Collect eigenvalues (diagonal) and sort DESCENDING, permuting eigenvectors.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> diag(n);
    for (int i = 0; i < n; ++i) diag[i] = a[static_cast<size_t>(i) * n + i];
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return diag[x] > diag[y]; });

    evals.resize(n);
    evecs.assign(static_cast<size_t>(n) * n, 0.0);
    for (int newCol = 0; newCol < n; ++newCol) {
        const int oldCol = order[newCol];
        evals[newCol] = diag[oldCol];
        for (int row = 0; row < n; ++row)
            evecs[static_cast<size_t>(row) * n + newCol] =
                v[static_cast<size_t>(row) * n + oldCol];
    }
}

// -----------------------------------------------------------------------------
// sklearn svd_flip sign convention: for each eigenvector (column), find the
// entry of largest |value| and flip the whole column so that entry is positive.
// Deterministic → stable PC band colours across runs.
// -----------------------------------------------------------------------------
inline void svdFlipColumns(std::vector<double>& evecs, int n) {
    for (int col = 0; col < n; ++col) {
        int    bestRow = 0;
        double bestAbs = -1.0;
        for (int row = 0; row < n; ++row) {
            const double a = std::fabs(evecs[static_cast<size_t>(row) * n + col]);
            if (a > bestAbs) { bestAbs = a; bestRow = row; }
        }
        if (evecs[static_cast<size_t>(bestRow) * n + col] < 0.0) {
            for (int row = 0; row < n; ++row)
                evecs[static_cast<size_t>(row) * n + col] *= -1.0;
        }
    }
}

} // namespace pca
