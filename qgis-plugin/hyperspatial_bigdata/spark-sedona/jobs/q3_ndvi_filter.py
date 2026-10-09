"""
Q3: NDVI filter — PER-PIXEL vegetation analysis on the real rasters.

Direct-image replacement for the old tile-mean CSV filter: for every tile_r*.tif
produced by the C++ engine, Sedona computes the per-pixel mean NDVI and the
fraction of pixels whose NDVI >= threshold (healthy vegetation), using the
actual pixels (nodata = 0.0 excluded), not a precomputed tile average.

Requires Apache Sedona >= 1.5.1 (raster support).
"""
import argparse, sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from utils.sedona_session import get_sedona_session
from utils.raster_io import load_tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rasters",   required=True,
                        help="Directory containing the engine's tile_r*.tif outputs")
    parser.add_argument("--output",    required=True, help="Spark CSV output directory")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    thr = float(args.threshold)

    sedona = get_sedona_session("Q3-NDVIFilter-Raster")
    tiles = load_tiles(sedona, args.rasters, view="tiles")
    if tiles.rdd.isEmpty():
        print("Q3: no tiles found under --rasters")
        return

    # All tiles share one band layout (the engine guarantees it), so read the
    # NDVI band number once and derive the 0-based index map-algebra needs.
    nbands = int(tiles.select("nbands").first()["nbands"])
    ndvi_band = nbands - 6     # 1-based (RS_ZonalStats)
    ndvi0 = ndvi_band - 1      # 0-based (RS_MapAlgebra)

    # Per-pixel vegetation mask: 1 where NDVI >= threshold, else 0. nodata NDVI
    # is 0.0 (< threshold) so it contributes 0, i.e. counts as non-vegetation.
    veg_expr = (f"RS_MapAlgebra(rast, 'D', "
                f"'out = (rast[{ndvi0}] >= {thr}) ? 1.0 : 0.0;')")

    sedona.sql(f"""
        SELECT
            path,
            RS_ZonalStats(rast, RS_Envelope(rast), {ndvi_band}, 'mean',  true, true) AS mean_ndvi,
            RS_ZonalStats(rast, RS_Envelope(rast), {ndvi_band}, 'count', true, true) AS valid_px,
            RS_ZonalStats({veg_expr}, RS_Envelope(rast), 1, 'sum', false, true)      AS veg_px
        FROM tiles
    """).createOrReplaceTempView("stats")

    result = sedona.sql(f"""
        SELECT path, mean_ndvi, valid_px, veg_px,
               CASE WHEN valid_px > 0 THEN veg_px / valid_px ELSE 0.0 END AS veg_fraction
        FROM stats
        WHERE mean_ndvi >= {thr}
    """)
    result.cache()
    count = result.count()
    result.write.mode("overwrite").csv(args.output, header=True)
    print(f"Q3: {count} tiles with per-pixel mean NDVI >= {thr}")


if __name__ == "__main__":
    main()
