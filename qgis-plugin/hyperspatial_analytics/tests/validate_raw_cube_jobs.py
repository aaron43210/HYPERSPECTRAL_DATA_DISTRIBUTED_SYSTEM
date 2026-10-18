"""End-to-end validation of the raw-cube Sedona jobs (Q4 NDVI, Q5 K-Means).

Runs each job the way the plugin does - raster_prep (GDAL) -> bundled job ->
spark-submit - on a real EnMAP L2A scene, then checks the outputs:

  Q4: mean NDVI / valid-pixel count equal a NumPy computation over the
      non-nodata pixels of the same RED/NIR bands.
  Q5: MLlib-vs-Jiffle agreement, background == EnMAP's QL background, and the
      clusters labelled Water / Snow/ice scored against the scene's
      QL_QUALITY_CLASSES (water = 2) and QL_QUALITY_SNOW masks (DLR processor
      outputs: a reference, not ground truth).

Usage (a Python that has osgeo/GDAL, e.g. QGIS's or Homebrew's):
  python3 validate_raw_cube_jobs.py --cube <...SPECTRAL_IMAGE.BSQ> \
      [--spark-submit ~/hyperspatial-venv/bin/spark-submit] [--save results.csv]
Exit code 0 = every check passed.
"""
import argparse, csv, glob, importlib.util, os, subprocess, sys, tempfile, time

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.normpath(os.path.join(HERE, "..", "hyperspatial_analytics"))
JOBS = os.path.join(PLUGIN, "spark-sedona", "jobs")
PACKAGES = ("org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.5.1,"
            "org.datasyslab:geotools-wrapper:1.5.1-28.2")
REPOS = "https://artifacts.unidata.ucar.edu/repository/unidata-all"

spec = importlib.util.spec_from_file_location("raster_prep", os.path.join(PLUGIN, "raster_prep.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def spark(submit, job, *args):
    env = dict(os.environ, HADOOP_CONF_DIR="")
    py = os.path.join(os.path.dirname(submit), "python")
    if os.path.exists(py):
        env.setdefault("PYSPARK_PYTHON", py)
        env.setdefault("PYSPARK_DRIVER_PYTHON", py)
    cmd = [submit, "--packages", PACKAGES, "--repositories", REPOS, os.path.join(JOBS, job), *args]
    t = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    lines = [l for l in (res.stdout + res.stderr).splitlines()
             if l.startswith(("Q5:", "RawNDVI:")) or "Exception" in l or "Error" in l]
    print("\n".join("    " + l for l in lines[:40]))
    if res.returncode != 0:
        raise RuntimeError(f"{job} exited {res.returncode}")
    return time.time() - t


def ql(scene_dir, name):
    hits = glob.glob(os.path.join(scene_dir, f"*{name}.TIF"))
    return gdal.Open(hits[0]).ReadAsArray() if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--spark-submit", default=os.path.expanduser("~/hyperspatial-venv/bin/spark-submit"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--features", default="shape_brightness")
    ap.add_argument("--save", default=None, help="Write the metrics table to this CSV")
    a = ap.parse_args()
    work = a.work or tempfile.mkdtemp(prefix="hs_validate_")
    scene = os.path.dirname(os.path.abspath(a.cube))
    checks, metrics = [], []

    def check(name, ok, value):
        checks.append((name, bool(ok), value))
        metrics.append((name, value, "PASS" if ok else "FAIL"))

    # ------------------------------------------------------------------ Q4
    print("== Q4 raw-cube NDVI")
    red, nir, _ = rp.detect_red_nir(a.cube)
    info = rp.prepare_red_nir_tile(a.cube, os.path.join(work, "q4_tiles"), red, nir)
    q4_out = os.path.join(work, "q4_out")
    args = ["--rasters", os.path.join(work, "q4_tiles"), "--output", q4_out,
            "--red-band", "1", "--nir-band", "2"]
    if info["nodata"] is not None:
        args += ["--nodata", str(info["nodata"])]
    spark(a.spark_submit, "q_raw_ndvi.py", *args)
    row = next(csv.DictReader(open(glob.glob(os.path.join(q4_out, "part-*.csv"))[0])))
    ds = gdal.Open(a.cube)
    r = ds.GetRasterBand(red).ReadAsArray().astype("f8")
    n = ds.GetRasterBand(nir).ReadAsArray().astype("f8")
    valid = (r != info["nodata"]) & (n != info["nodata"]) & ((r + n) != 0)
    truth = float(((n - r) / np.where(valid, n + r, 1))[valid].mean())
    check("Q4 mean NDVI == NumPy truth (±0.001)",
          abs(float(row["mean_ndvi"]) - truth) < 1e-3, f"{float(row['mean_ndvi']):.4f} vs {truth:.4f}")
    # RS_ZonalStats over RS_Envelope(rast) leaves out the raster's outermost
    # pixel ring (Sedona 1.5.1 edge effect), so the job's count must equal the
    # NumPy count minus the valid pixels on that ring - no other difference.
    edge = np.zeros_like(valid)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    expected_px = int(valid.sum()) - int((valid & edge).sum())
    check("Q4 valid pixels == NumPy truth minus outer-ring pixels",
          int(float(row["valid_px"])) == expected_px,
          f"{int(float(row['valid_px']))} vs {expected_px} "
          f"({int((valid & edge).sum())} valid pixels on the raster edge)")

    # ------------------------------------------------------------------ Q5
    print(f"== Q5 K-Means (k={a.k}, features={a.features})")
    bands = rp.detect_bands(a.cube)
    tiles = os.path.join(work, "q5_tiles")
    info = rp.prepare_band_tiles(a.cube, tiles, list(bands.values()), tile_size=256)
    out, rast = os.path.join(work, "q5_out"), os.path.join(work, "q5_raster")
    args = ["--rasters", tiles, "--output", out, "--raster-out", rast, "--k", str(a.k),
            "--features", a.features, "--roles", ",".join(bands), "--scale", str(info["scale"]),
            "--verify"]
    if info["nodata"] is not None:
        args += ["--nodata", str(info["nodata"])]
    elapsed = spark(a.spark_submit, "q5_kmeans.py", *args)
    run = dict(csv.reader(open(os.path.join(out, "run_info.csv"))))
    clusters = list(csv.DictReader(open(os.path.join(out, "clusters.csv"))))
    check("Q5 MLlib vs Jiffle agreement >= 99.99%", float(run["agreement"]) >= 0.9999,
          f"{float(run['agreement']):.6%}")
    check("Q5 job wall time < 30 s", float(run["wall_sec"]) < 30,
          f"{float(run['wall_sec']):.1f} s (spark-submit total {elapsed:.1f} s)")

    vrt = os.path.join(work, "q5_classes.vrt")
    gdal.BuildVRT(vrt, sorted(glob.glob(os.path.join(rast, "**", "*.tif"), recursive=True)),
                  options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255))
    cls = gdal.Open(vrt).ReadAsArray()
    classes, snow = ql(scene, "QL_QUALITY_CLASSES"), ql(scene, "QL_QUALITY_SNOW")
    if classes is not None and classes.shape == cls.shape:
        check("Q5 nodata pixels == QL background (class 3)",
              int((cls == 255).sum()) == int((classes == 3).sum()),
              f"{int((cls == 255).sum())} vs {int((classes == 3).sum())}")
        for label, ref, metric in (("Water", classes == 2, "iou"), ("Snow/ice", snow == 1, "precision")):
            ids = [int(c["cluster"]) for c in clusters if c["suggested_label"] == label]
            if not ids:
                check(f"Q5 a cluster is labelled {label}", False, "none")
                continue
            m = np.isin(cls, ids)
            inter = float((m & ref).sum())
            iou, prec, rec = inter / float((m | ref).sum()), inter / m.sum(), inter / ref.sum()
            want = 0.5 if metric == "iou" else 0.85
            got = iou if metric == "iou" else prec
            check(f"Q5 '{label}' clusters {ids} {metric} >= {want}", got >= want,
                  f"IoU={iou:.3f} precision={prec:.3f} recall={rec:.3f}")
    else:
        print("  (QL masks not found next to the cube - skipping reference checks)")

    print("\n== Results")
    for name, ok, value in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value}")
    print(f"\n  work dir: {work}")
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        with open(a.save, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["check", "value", "result"])
            w.writerows(metrics)
            w.writerow(["scene", os.path.basename(a.cube), ""])
            w.writerow(["k / features", f"{a.k} / {a.features}", ""])
        print(f"  saved: {a.save}")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
