"""
Shared raster loading for the Sedona raster-analytics jobs.

Instead of the old tile-mean summary.csv bridge, the analytics now read the
ACTUAL per-tile GeoTIFFs the C++ engine wrote (tile_r*.tif, 224 raw + 7 trailing
index bands) and compute per-pixel statistics with Sedona's raster functions
(RS_*). This is "direct image" analysis: NDVI/zonal stats come from the real
pixels, not a precomputed per-tile average.

Requires Apache Sedona >= 1.5.1 with raster support (geotools-wrapper, already
configured in utils/sedona_session.py).
"""
from __future__ import annotations


# Fixed trailing-index-band order written by the C++ engine (TileWriter):
# NDVI, NDWI, NDBI, SAVI, EVI, NDMI, MSAVI  -> 7 bands at the end of each tile.
# NDVI is therefore band (num_bands - 6); the rest follow it. Using an offset
# from num_bands (not a hard-coded number) keeps this correct regardless of how
# many raw bands survived BadBandRemover.
INDEX_OFFSETS = {
    "ndvi": 6, "ndwi": 5, "ndbi": 4, "savi": 3, "evi": 2, "ndmi": 1, "msavi": 0,
}


def load_tiles(sedona, rasters_dir: str, view: str = "tiles"):
    """Load every tile_r*.tif under `rasters_dir` into a Sedona raster view.

    The resulting temp view `view` has columns:
        path       : source GeoTIFF path
        rast       : Sedona Raster (all bands)
        nbands     : RS_NumBands(rast)
        ndvi_band .. msavi_band : 1-based band numbers of each index
        srid       : raster CRS authority code (RS_SRID)
        env4326    : tile footprint polygon reprojected to EPSG:4326
    Rows whose CRS is unknown (srid = 0) are dropped (can't be reprojected).
    """
    raw = (sedona.read.format("binaryFile")
           .option("pathGlobFilter", "tile_r*.tif")
           .option("recursiveFileLookup", "true")
           .load(rasters_dir))
    raw.createOrReplaceTempView(f"{view}_raw")

    band_cols = ",\n            ".join(
        f"RS_NumBands(rast) - {off} AS {name}_band"
        for name, off in INDEX_OFFSETS.items()
    )

    sedona.sql(f"""
        WITH r AS (
            SELECT path, RS_FromGeoTiff(content) AS rast
            FROM {view}_raw
        )
        SELECT
            path,
            rast,
            RS_NumBands(rast) AS nbands,
            {band_cols},
            RS_SRID(rast) AS srid,
            ST_Transform(
                ST_SetSRID(RS_Envelope(rast), RS_SRID(rast)),
                'epsg:4326'
            ) AS env4326
        FROM r
        WHERE RS_SRID(rast) <> 0
    """).createOrReplaceTempView(view)
    return sedona.table(view)
