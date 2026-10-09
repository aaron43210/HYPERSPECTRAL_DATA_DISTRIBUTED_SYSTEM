"""
Q2: Spatial join — PER-PIXEL zonal statistics on agricultural boundaries.

Direct-image replacement for the old tile-mean CSV join. For each field polygon,
Sedona computes zonal statistics (mean NDVI/NDWI/NDBI) over the actual pixels of
every tile_r*.tif that overlaps the field, then aggregates across tiles weighted
by the in-field valid-pixel count — a true per-field mean, not an average of
per-tile averages.

--agri: agricultural boundaries GeoJSON (EPSG:4326), with per-feature properties
field_id and crop_type.

Requires Apache Sedona >= 1.5.1 (raster support).
"""
import argparse, sys, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from utils.sedona_session import get_sedona_session
from utils.raster_io import load_tiles
from shapely.geometry import shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rasters", required=True,
                        help="Directory containing the engine's tile_r*.tif outputs")
    parser.add_argument("--agri",    required=True, help="Boundaries GeoJSON (EPSG:4326)")
    parser.add_argument("--output",  required=True, help="Spark CSV output directory")
    args = parser.parse_args()

    sedona = get_sedona_session("Q2-SpatialJoin-Raster")
    tiles = load_tiles(sedona, args.rasters, view="tiles")
    if tiles.rdd.isEmpty():
        print("Q2: no tiles found under --rasters")
        return

    # Load fields with json+shapely (no GeoPandas/Fiona). Pass geometry as WKT so
    # we don't depend on the geometry-UDT auto-conversion of createDataFrame.
    with open(args.agri, "r") as f:
        gj = json.load(f)
    rows = []
    for feat in gj.get("features", []):
        if not feat.get("geometry"):
            continue
        props = feat.get("properties", {})
        rows.append((str(props.get("field_id", "unknown")),
                     str(props.get("crop_type", "unknown")),
                     shape(feat["geometry"]).wkt))
    if not rows:
        print("Q2: boundaries file contains no geometries")
        return
    sedona.createDataFrame(rows, ["field_id", "crop_type", "wkt"]).createOrReplaceTempView("agri_raw")
    sedona.sql("""
        SELECT field_id, crop_type, ST_SetSRID(ST_GeomFromWKT(wkt), 4326) AS geom
        FROM agri_raw
    """).createOrReplaceTempView("agri")

    # Per (field, overlapping tile): zonal mean + in-field pixel count.
    sedona.sql("""
        SELECT
            a.field_id, a.crop_type, t.path,
            COALESCE(RS_ZonalStats(t.rast, a.geom, t.ndvi_band, 'count', true, true), 0) AS px,
            RS_ZonalStats(t.rast, a.geom, t.ndvi_band, 'mean', true, true) AS m_ndvi,
            RS_ZonalStats(t.rast, a.geom, t.ndwi_band, 'mean', true, true) AS m_ndwi,
            RS_ZonalStats(t.rast, a.geom, t.ndbi_band, 'mean', true, true) AS m_ndbi
        FROM tiles t JOIN agri a ON ST_Intersects(t.env4326, a.geom)
    """).createOrReplaceTempView("per_tile")

    # Aggregate across tiles, weighting each tile's mean by its in-field pixels.
    result = sedona.sql("""
        SELECT
            field_id, crop_type,
            COUNT(DISTINCT path)                     AS tile_count,
            CAST(SUM(px) AS LONG)                    AS pixel_count,
            SUM(m_ndvi * px) / NULLIF(SUM(px), 0)    AS mean_ndvi,
            SUM(m_ndwi * px) / NULLIF(SUM(px), 0)    AS mean_ndwi,
            SUM(m_ndbi * px) / NULLIF(SUM(px), 0)    AS mean_ndbi
        FROM per_tile
        WHERE px > 0
        GROUP BY field_id, crop_type
    """)
    result.cache()
    count = result.count()
    result.write.mode("overwrite").csv(args.output, header=True)
    print(f"Q2: {count} field-level per-pixel results")


if __name__ == "__main__":
    main()
