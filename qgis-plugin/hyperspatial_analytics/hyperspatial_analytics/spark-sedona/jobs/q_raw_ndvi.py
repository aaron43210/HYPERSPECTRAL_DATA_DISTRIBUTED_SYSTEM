"""
Raw-cube NDVI — compute NDVI per-pixel straight from a hyperspectral cube,
with NO dependency on the C++ engine or precomputed index tiles.

Input is a GeoTIFF holding (at least) a RED and a NIR band; the plugin's GDAL
prep step extracts those two bands from a BSQ/ENVI/other cube first. NDVI is
computed in the JVM with Sedona map algebra:

    out = (nir + red) == 0 ? NODATA : (nir - red) / (nir + red)

Background pixels (nir+red == 0) are mapped to a real sentinel (-9999), NOT NaN,
so RS_ZonalStats can exclude them by value - otherwise mean/count come back NaN
(NaN != NaN is inexcludable in Sedona 1.5.1).

Requires Apache Sedona >= 1.5.1 (raster) with geotools-wrapper 1.5.1-28.2 (Jiffle).
"""
import argparse, sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from utils.sedona_session import get_sedona_session

NODATA = -9999.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rasters",   required=True, help="Dir with the prepared RED+NIR GeoTIFF(s)")
    p.add_argument("--output",    required=True, help="Spark CSV output directory")
    p.add_argument("--red-band",  type=int, required=True, help="1-based RED band in the GeoTIFF")
    p.add_argument("--nir-band",  type=int, required=True, help="1-based NIR band in the GeoTIFF")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--raster-out", default=None,
                   help="Optional dir: write the per-pixel NDVI GeoTIFF for the canvas")
    args = p.parse_args()
    red0, nir0 = args.red_band - 1, args.nir_band - 1

    sedona = get_sedona_session("RawCube-NDVI")
    raw = (sedona.read.format("binaryFile")
           .option("pathGlobFilter", "*.tif")
           .option("recursiveFileLookup", "true")
           .load(args.rasters))
    raw.createOrReplaceTempView("raw")
    sedona.sql("SELECT path, RS_FromGeoTiff(content) AS rast FROM raw") \
          .createOrReplaceTempView("tiles")
    if sedona.table("tiles").limit(1).count() == 0:
        print("RawNDVI: no .tif found under --rasters")
        return

    # Per-pixel NDVI, background -> sentinel nodata (not NaN) so stats can exclude it.
    ndvi = (f"RS_MapAlgebra(rast, 'D', "
            f"'out = (rast[{nir0}] + rast[{red0}]) == 0 ? {NODATA} : "
            f"(rast[{nir0}] - rast[{red0}]) / (rast[{nir0}] + rast[{red0}]);', {NODATA})")

    sedona.sql(f"""
        SELECT path,
            RS_ZonalStats({ndvi}, RS_Envelope(rast), 1, 'mean',  true) AS mean_ndvi,
            RS_ZonalStats({ndvi}, RS_Envelope(rast), 1, 'count', true) AS valid_px
        FROM tiles
    """).createOrReplaceTempView("stats")

    result = sedona.sql(f"SELECT path, mean_ndvi, valid_px FROM stats")
    result.cache()
    n = result.count()
    result.write.mode("overwrite").csv(args.output, header=True)
    print(f"RawNDVI: computed per-pixel NDVI for {n} raster(s); threshold={args.threshold}")

    if args.raster_out:
        # Write the per-pixel NDVI raster (JVM-side; never pull rasters to Python).
        # rasterField must be BINARY (RS_AsGeoTiff), not a live raster expression.
        (sedona.sql(f"SELECT RS_AsGeoTiff({ndvi}) AS content, "
                    f"concat(regexp_replace(element_at(split(path,'/'), -1), "
                    f"'\\.tif$', ''), '_ndvi') AS name FROM tiles")
                .write.format("raster")
                .option("rasterField", "content").option("rasterPathField", "name")
                .option("fileExtension", ".tif").mode("overwrite").save(args.raster_out))
        print(f"RawNDVI: wrote per-pixel NDVI GeoTIFF(s) to {args.raster_out}")


if __name__ == "__main__":
    main()
