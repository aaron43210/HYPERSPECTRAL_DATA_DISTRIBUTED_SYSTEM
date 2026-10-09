"""
Q1: AOI filter — PER-PIXEL analysis inside an Area of Interest.

Direct-image replacement for the old tile-mean CSV filter. For each tile_r*.tif
whose footprint intersects the AOI, Sedona computes zonal statistics of NDVI
over the pixels that actually fall inside the AOI polygon (RS_ZonalStats
reprojects the zone into each raster's CRS automatically).

--aoi accepts a GeoJSON file (geometry / Feature / FeatureCollection; a
collection is dissolved) OR a raw GeoJSON geometry string, interpreted as
EPSG:4326 per the GeoJSON spec.

Requires Apache Sedona >= 1.5.1 (raster support).
"""
import argparse, sys, os, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from utils.sedona_session import get_sedona_session
from utils.raster_io import load_tiles


def _load_aoi_wkt(aoi_arg: str) -> str:
    """Return the AOI as a single WKT string (EPSG:4326)."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    if os.path.exists(aoi_arg):
        with open(aoi_arg, "r") as f:
            data = json.load(f)
    else:
        data = json.loads(aoi_arg)
    gtype = data.get("type")
    if gtype == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
        if not geoms:
            raise ValueError("AOI FeatureCollection contains no geometries")
        geom = unary_union(geoms)
    elif gtype == "Feature":
        geom = shape(data["geometry"])
    else:
        geom = shape(data)
    return geom.wkt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rasters", required=True,
                        help="Directory containing the engine's tile_r*.tif outputs")
    parser.add_argument("--output",  required=True, help="Spark CSV output directory")
    parser.add_argument("--aoi",     required=True,
                        help="GeoJSON AOI: a file path OR a raw geometry string (EPSG:4326)")
    args = parser.parse_args()

    sedona = get_sedona_session("Q1-AOIFilter-Raster")
    tiles = load_tiles(sedona, args.rasters, view="tiles")
    if tiles.rdd.isEmpty():
        print("Q1: no tiles found under --rasters")
        return

    aoi_wkt = _load_aoi_wkt(args.aoi).replace("'", "''")
    sedona.sql(
        f"SELECT ST_SetSRID(ST_GeomFromWKT('{aoi_wkt}'), 4326) AS geom"
    ).createOrReplaceTempView("aoi")

    # Keep tiles that intersect the AOI, and summarise NDVI over ONLY the pixels
    # inside the AOI polygon (RS_ZonalStats transforms the AOI into each raster's
    # CRS; nodata pixels are excluded).
    result = sedona.sql("""
        SELECT
            t.path,
            RS_ZonalStats(t.rast, a.geom, t.ndvi_band, 'mean',  true, true) AS mean_ndvi_in_aoi,
            RS_ZonalStats(t.rast, a.geom, t.ndvi_band, 'count', true, true) AS valid_px_in_aoi
        FROM tiles t, aoi a
        WHERE ST_Intersects(t.env4326, a.geom)
    """)
    result = result.where("valid_px_in_aoi > 0")
    result.cache()
    count = result.count()
    result.write.mode("overwrite").csv(args.output, header=True)
    print(f"Q1: {count} tiles contribute pixels inside the AOI")


if __name__ == "__main__":
    main()
