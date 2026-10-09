#pragma once
// =============================================================================
// Tiler.hpp  —  Computes the list of TileBounds for a scene (header-only)
// =============================================================================

#include "RasterReader.hpp"
#include <vector>
#include <algorithm>

/// Decomposes a scene of (totalRows × totalCols) pixels into a grid of
/// non-overlapping tiles, each at most (tileSize × tileSize) pixels.
/// Edge tiles are clipped to the scene boundary.
class Tiler {
public:
    Tiler(int totalRows, int totalCols, int tileSize)
        : rows_(totalRows), cols_(totalCols), size_(tileSize)
    {
        if (tileSize <= 0)
            throw std::invalid_argument("tileSize must be positive");
    }

    /// Return the ordered list of tile bounds, row-major (left→right, top→bottom).
    std::vector<TileBounds> computeTiles() const {
        std::vector<TileBounds> tiles;
        tiles.reserve(
            ((rows_ + size_ - 1) / size_) *
            ((cols_ + size_ - 1) / size_)
        );
        for (int r = 0; r < rows_; r += size_) {
            for (int c = 0; c < cols_; c += size_) {
                tiles.push_back({
                    r,
                    c,
                    std::min(r + size_, rows_),
                    std::min(c + size_, cols_)
                });
            }
        }
        return tiles;
    }

    int tileSize()   const noexcept { return size_; }
    int sceneRows()  const noexcept { return rows_; }
    int sceneCols()  const noexcept { return cols_; }

    /// Number of tile columns (ceil division)
    int tileGridCols() const noexcept {
        return (cols_ + size_ - 1) / size_;
    }

    /// Number of tile rows
    int tileGridRows() const noexcept {
        return (rows_ + size_ - 1) / size_;
    }

private:
    int rows_, cols_, size_;
};
