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
    parser.add_argument("--raster-out", default=None,
                        help="Optional directory: write a per-pixel NDVI>=threshold "
                             "veg-mask GeoTIFF per tile (for loading onto the canvas)")
    args = parser.parse_args()
    thr = float(args.threshold)

    sedona = get_sedona_session("Q3-NDVIFilter-Raster")
    tiles = load_tiles(sedona, args.rasters, view="tiles")
    # Emptiness check must stay JVM-side: tiles.rdd / take / head would serialize
    # the `rast` RasterType column to Python, which Sedona 1.5.1 cannot do
    # (RasterType.deserialize is unimplemented). limit(1).count() never crosses.
    if tiles.limit(1).count() == 0:
        print("Q3: no tiles found under --rasters")
        return

    # All tiles share one band layout (the engine guarantees it), so read the
    # NDVI band number once and derive the 0-based index map-algebra needs.
    nbands = int(tiles.select("nbands").first()["nbands"])
    ndvi_band = nbands - 6     # 1-based (RS_ZonalStats)
    ndvi0 = ndvi_band - 1      # 0-based (RS_MapAlgebra)

    # Per-pixel vegetation mask: 1 where NDVI >= threshold, else 0. Used for the
    # optional --raster-out GeoTIFF (the visible per-pixel product).
    #
    # NOTE: we deliberately do NOT derive a scalar veg_px/veg_fraction here.
    # Sedona 1.5.1's RS_ZonalStats cannot count value-matching pixels: 'sum'/
    # 'mean' on a Jiffle raster return NaN (source-nodata -> NaN, inexcludable
    # since NaN != NaN) and 'count' counts the whole footprint regardless of
    # null/con(). The reliable per-tile scalars are mean_ndvi and valid_px; the
    # per-pixel vegetation answer lives in the mask raster below.
    veg_expr = (f"RS_MapAlgebra(rast, 'D', "
                f"'out = (rast[{ndvi0}] >= {thr}) ? 1.0 : 0.0;')")

    sedona.sql(f"""
        SELECT
            path,
            RS_ZonalStats(rast, RS_Envelope(rast), {ndvi_band}, 'mean',  true) AS mean_ndvi,
            RS_ZonalStats(rast, RS_Envelope(rast), {ndvi_band}, 'count', true) AS valid_px
        FROM tiles
    """).createOrReplaceTempView("stats")

    # isnan guard: a tile with no valid pixels has mean_ndvi = NaN, and Spark
    # SQL orders NaN above every number, so `NaN >= thr` is TRUE - empty
    # background tiles would otherwise be reported as vegetated.
    result = sedona.sql(f"""
        SELECT path, mean_ndvi, valid_px
        FROM stats
        WHERE NOT isnan(mean_ndvi) AND mean_ndvi >= {thr}
    """)
    result.cache()
    count = result.count()
    result.write.mode("overwrite").csv(args.output, header=True)
    print(f"Q3: {count} tiles with per-pixel mean NDVI >= {thr}")

    # Optional "pixel-based raster processing you can see": materialise the
    # per-pixel veg mask (1 = NDVI>=thr, 0 = not) as one GeoTIFF per tile.
    #
    # It MUST stay JVM-side: Sedona 1.5.1's Python RasterType.deserialize is
    # unimplemented, so collect()/toLocalIterator() on a raster (or RS_AsGeoTiff)
    # column throws "RasterType.deserialize is not implemented yet". The native
    # `raster` data source (RasterFileFormat) writes one GeoTIFF per row entirely
    # in the JVM - no Python round-trip - naming each file from `rasterPathField`.
    if args.raster_out:
        veg_df = sedona.sql(f"""
            SELECT
                RS_AsGeoTiff({veg_expr}) AS content,
                concat(regexp_replace(element_at(split(path, '/'), -1),
                                      '\\.tif$', ''), '_vegmask') AS name
            FROM tiles
        """)
        (veg_df.write.format("raster")
               .option("rasterField", "content")
               .option("rasterPathField", "name")
               .option("fileExtension", ".tif")
               .mode("overwrite")
               .save(args.raster_out))
        print(f"Q3: wrote per-pixel veg-mask GeoTIFF(s) to {args.raster_out}")


if __name__ == "__main__":
    main()
