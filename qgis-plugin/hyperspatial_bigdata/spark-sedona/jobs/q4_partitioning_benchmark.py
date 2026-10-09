"""
Q4: Benchmark the raster zonal spatial-join with vs without a Spark repartition.

Standalone (not wired into the plugin). Uses the same direct-image path as Q2:
join every tile_r*.tif against field polygons and count in-field pixels, timing
the join once as loaded and once after a plain hash repartition.

NOTE: `repartition(n)` is a HASH shuffle, not Sedona spatial partitioning, so
treat the "repartition" number as a shuffle-overhead comparison, not a spatial
locality speed-up. For true spatial partitioning use Sedona's SpatialRDD API.

Requires Apache Sedona >= 1.5.1 (raster support).
"""
import argparse, time, csv, sys, json, os
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from utils.sedona_session import get_sedona_session
from utils.raster_io import load_tiles
from shapely.geometry import shape


def _load_agri(sedona, agri_path: str, view: str):
    with open(agri_path, "r") as f:
        gj = json.load(f)
    rows = [(str(ft.get("properties", {}).get("field_id", "unknown")),
             shape(ft["geometry"]).wkt)
            for ft in gj.get("features", []) if ft.get("geometry")]
    sedona.createDataFrame(rows, ["field_id", "wkt"]).createOrReplaceTempView(view + "_raw")
    sedona.sql(f"""
        SELECT field_id, ST_SetSRID(ST_GeomFromWKT(wkt), 4326) AS geom FROM {view}_raw
    """).createOrReplaceTempView(view)


def run_join(sedona, rasters_dir, agri_path, repartition):
    suffix = f"p{repartition}" if repartition else "nopart"
    tiles = load_tiles(sedona, rasters_dir, view=f"tiles_{suffix}")
    _load_agri(sedona, agri_path, f"agri_{suffix}")
    if repartition:
        tiles = tiles.repartition(repartition)
        tiles.createOrReplaceTempView(f"tiles_{suffix}")

    start = time.perf_counter()
    result = sedona.sql(f"""
        SELECT a.field_id,
               RS_ZonalStats(t.rast, a.geom, t.ndvi_band, 'count', true, true) AS px
        FROM tiles_{suffix} t JOIN agri_{suffix} a ON ST_Intersects(t.env4326, a.geom)
    """)
    result.cache()
    count = result.count()
    elapsed = time.perf_counter() - start
    for v in (f"tiles_{suffix}", f"tiles_{suffix}_raw", f"agri_{suffix}", f"agri_{suffix}_raw"):
        try:
            sedona.catalog.dropTempView(v)
        except Exception:
            pass
    return elapsed, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rasters", required=True)
    parser.add_argument("--agri",    required=True)
    parser.add_argument("--csv-out", default="../../benchmarks/results/q4_partitioning.csv")
    args = parser.parse_args()

    sedona = get_sedona_session("Q4-PartitioningBenchmark-Raster")
    t1, c1 = run_join(sedona, args.rasters, args.agri, repartition=0)
    t2, c2 = run_join(sedona, args.rasters, args.agri, repartition=200)

    speedup = t1 / t2 if t2 > 0 else float("inf")
    print(f"No repartition : {t1:.2f}s ({c1} rows)")
    print(f"repartition(200): {t2:.2f}s ({c2} rows)")
    print(f"Ratio           : {speedup:.2f}x (hash shuffle, not spatial)")

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "time_sec", "rows", "ratio"])
        w.writerow(["no_repartition", t1, c1, 1.0])
        w.writerow(["repartition_200", t2, c2, round(speedup, 3)])


if __name__ == "__main__":
    main()
