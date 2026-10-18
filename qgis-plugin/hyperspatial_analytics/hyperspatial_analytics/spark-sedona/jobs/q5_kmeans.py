"""
Q5: Unsupervised land-cover clustering of a raw hyperspectral cube with Spark
MLlib K-Means - no training labels, no C++ engine.

Input is a directory of tile_r*.tif GeoTIFFs holding a band subset of the cube
(the plugin's raster_prep.prepare_band_tiles extracts blue/green/red/nir/swir1/
swir2 by wavelength, keeping raw stored values). Everything per-pixel stays in
the JVM:

  1. RS_BandAsArray + arrays_zip + posexplode   -> one row per valid pixel
  2. ml_jiffle.add_spark_features               -> feature vector (default:
                                                   spectral shape + brightness)
  3. KMeans.fit on a --sample of the pixels     -> k centres
  4. model.transform on all pixels              -> per-cluster pixel count and
                                                   mean reflectance -> labels
  5. ml_jiffle.kmeans_to_jiffle -> RS_MapAlgebra -> class map, one byte GeoTIFF
                                                   per tile (nodata 255)

Outputs (--output): clusters.csv (cluster, suggested_label, pixels, pct,
ndvi/ndwi/ndsi, mean reflectance per band) and run_info.csv (parameters and
timings). --raster-out receives the class-map tiles. Plain CSV via the csv
module (no GeoParquet / pyarrow). --verify also reports the per-pixel agreement
between MLlib's predictions and the Jiffle map (a test hook; it adds a join).

Tiling matters: a whole EnMAP scene as one raster row exhausts spark-submit's
default 1 GB driver heap in RS_BandAsArray/posexplode; 256-px tiles do not.

Requires Apache Sedona >= 1.5.1 (raster) with geotools-wrapper 1.5.1-28.2 (Jiffle).
"""
import argparse, csv, os, sys, time
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from utils.sedona_session import get_sedona_session
from utils import ml_jiffle as mj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rasters", required=True, help="Dir with the prepared tile_r*.tif band subset")
    p.add_argument("--output", required=True, help="Dir for clusters.csv and run_info.csv")
    p.add_argument("--raster-out", default=None, help="Optional dir: class-map GeoTIFF per tile")
    p.add_argument("--k", type=int, default=8, help="Number of clusters (2-254)")
    p.add_argument("--features", choices=mj.FEATURE_MODES, default="shape_brightness")
    p.add_argument("--sample", type=float, default=0.1, help="Training fraction of valid pixels (0-1]")
    p.add_argument("--nodata", type=float, default=None, help="Source nodata value (stored units)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Stored value -> reflectance factor (EnMAP 0.0001); used for labels/CSV")
    p.add_argument("--roles", default="",
                   help="Comma list naming each band in order, e.g. blue,green,red,nir,swir1,swir2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--verify", action="store_true",
                   help="Report MLlib-vs-Jiffle per-pixel agreement (test hook)")
    args = p.parse_args()
    if not 2 <= args.k <= 254:
        sys.exit("Q5: --k must be between 2 and 254")
    if not 0.0 < args.sample <= 1.0:
        sys.exit("Q5: --sample must be in (0, 1]")
    t0 = time.time()

    from pyspark.sql import functions as F
    from pyspark.ml.clustering import KMeans

    sedona = get_sedona_session("Q5-KMeans-RawCube")
    raw = (sedona.read.format("binaryFile")
           .option("pathGlobFilter", "tile_r*.tif")
           .option("recursiveFileLookup", "true")
           .load(args.rasters))
    raw.createOrReplaceTempView("q5_raw")
    sedona.sql("SELECT path, RS_FromGeoTiff(content) AS rast FROM q5_raw") \
          .createOrReplaceTempView("q5_tiles")
    # JVM-side emptiness check: never pull the raster column into Python.
    n_tiles = sedona.table("q5_tiles").count()
    if n_tiles == 0:
        sys.exit("Q5: no tile_r*.tif found under --rasters")

    nb = int(sedona.sql("SELECT RS_NumBands(rast) AS n FROM q5_tiles").first()["n"])
    cols = [f"b{i}" for i in range(1, nb + 1)]
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    if roles and len(roles) != nb:
        sys.exit(f"Q5: --roles names {len(roles)} bands but the tiles have {nb}")
    roles = roles or cols

    # 1. Pixels -> rows (per tile, so each task holds one tile's arrays only).
    arrays = ", ".join(f"RS_BandAsArray(rast, {i}) AS b{i}" for i in range(1, nb + 1))
    sedona.sql(f"SELECT path, {arrays} FROM q5_tiles").createOrReplaceTempView("q5_arr")
    px = (sedona.sql(f"""
              SELECT path, pos, p.* FROM q5_arr
              LATERAL VIEW posexplode(arrays_zip({", ".join(cols)})) t AS pos, p""")
          .where(mj.valid_pixel_condition(cols, args.nodata))
          .cache())
    n_valid = px.count()
    if n_valid == 0:
        sys.exit("Q5: no valid pixels (all nodata?)")
    print(f"Q5: {n_tiles} tiles, {nb} bands ({','.join(roles)}), {n_valid} valid pixels")

    # 2-3. Features + K-Means on a sample.
    feats = mj.add_spark_features(px, cols, args.features)
    train = feats.sample(fraction=args.sample, seed=args.seed)
    t_fit = time.time()
    model = KMeans(k=args.k, seed=args.seed, maxIter=args.max_iter,
                   featuresCol="features").fit(train)
    fit_sec = time.time() - t_fit
    centres = [list(map(float, c)) for c in model.clusterCenters()]
    print(f"Q5: K-Means k={len(centres)} features={args.features} "
          f"trained on ~{args.sample:.0%} of pixels in {fit_sec:.1f}s")

    # 4. All pixels -> cluster sizes and mean spectra (in reflectance units).
    pred = model.transform(feats).select("path", "pos", "prediction", *cols)
    stats = (pred.groupBy("prediction")
                 .agg(F.count(F.lit(1)).alias("pixels"), *[F.avg(c).alias(c) for c in cols])
                 .collect())
    rows = []
    for s in sorted(stats, key=lambda r: int(r["prediction"])):
        mean = [float(s[c]) * args.scale for c in cols]
        idx = mj.cluster_indices(mean, roles)
        rows.append({"cluster": int(s["prediction"]),
                     "suggested_label": mj.suggest_label(mean, roles),
                     "pixels": int(s["pixels"]),
                     "pct": 100.0 * int(s["pixels"]) / n_valid,
                     **{k: v for k, v in idx.items()},
                     **{f"refl_{r}": m for r, m in zip(roles, mean)}})

    # 5. Class map: the model compiled to Jiffle, applied to every pixel in the JVM.
    script = mj.kmeans_to_jiffle(centres, nb, args.features, args.nodata, mj.NODATA_OUT)
    sedona.sql(f"""SELECT path, RS_MapAlgebra(rast, 'B', '{script}', {mj.NODATA_OUT}) AS cls
                   FROM q5_tiles""").createOrReplaceTempView("q5_cls")
    if args.raster_out:
        # rasterField must be BINARY (RS_AsGeoTiff); one file per tile, named
        # after its source tile so the plugin can mosaic them into one VRT.
        (sedona.sql("""SELECT RS_AsGeoTiff(cls) AS content,
                              concat(regexp_replace(element_at(split(path, '/'), -1),
                                                    '\\\\.tif$', ''), '_kmeans') AS name
                       FROM q5_cls""")
               .write.format("raster")
               .option("rasterField", "content").option("rasterPathField", "name")
               .option("fileExtension", ".tif").mode("overwrite")
               .save(args.raster_out))
        print(f"Q5: wrote class-map GeoTIFF tile(s) to {args.raster_out}")

    agreement = None
    if args.verify:
        jmap = sedona.sql("""SELECT path, pos, CAST(v AS INT) AS jcls FROM
                                 (SELECT path, RS_BandAsArray(cls, 1) AS a FROM q5_cls)
                             LATERAL VIEW posexplode(a) t AS pos, v""")
        joined = pred.select("path", "pos", "prediction").join(jmap, ["path", "pos"])
        agreement = joined.where("prediction = jcls").count() / n_valid
        background = jmap.where(f"jcls = {mj.NODATA_OUT}").count()
        print(f"Q5: verify agreement MLlib vs Jiffle = {agreement:.6%}; "
              f"nodata pixels in map = {background}")

    os.makedirs(args.output, exist_ok=True)
    fields = ["cluster", "suggested_label", "pixels", "pct", "ndvi", "ndwi", "ndsi"] + \
             [f"refl_{r}" for r in roles]
    with open(os.path.join(args.output, "clusters.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    wall = time.time() - t0
    info = {"k": len(centres), "features": args.features, "sample": args.sample,
            "seed": args.seed, "bands": ",".join(roles), "tiles": n_tiles,
            "valid_pixels": n_valid, "fit_sec": f"{fit_sec:.2f}", "wall_sec": f"{wall:.2f}",
            "jiffle_chars": len(script),
            "agreement": "" if agreement is None else f"{agreement:.8f}"}
    with open(os.path.join(args.output, "run_info.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(info.items())
    px.unpersist()

    def _fmt(v):
        return "  n/a" if v is None else f"{v:+.2f}"
    for r in rows:
        print(f"Q5:   cluster {r['cluster']:>2}  {r['suggested_label']:<24} {r['pct']:5.1f}%  "
              f"NDVI={_fmt(r['ndvi'])} NDWI={_fmt(r['ndwi'])} NDSI={_fmt(r['ndsi'])}")
    print(f"Q5: done in {wall:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
