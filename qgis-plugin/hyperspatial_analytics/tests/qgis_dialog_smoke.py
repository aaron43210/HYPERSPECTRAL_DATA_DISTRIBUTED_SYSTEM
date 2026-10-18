"""Headless smoke test of the real SedonaAnalyticsDialog inside QGIS's Python.

Drives Q4 (raw-cube NDVI) and Q5 (K-Means) through the dialog exactly as a
user would - fill the widgets, click RUN - while pumping the Qt event loop, and
checks: the job completes, the expected layer lands in QgsProject (Q5: one
paletted class map with one legend entry per cluster), and the GUI thread was
never blocked for long (the plugin's never-block-the-UI rule).

Run with QGIS's interpreter, offscreen (A=/Applications/QGIS-<ver>.app/Contents):
  PROJ_DATA=$A/Resources/qgis/proj QT_QPA_PLATFORM=offscreen \
  PYTHONPATH=$A/Resources/python3.12:$A/Resources/python3.12/lib-dynload:$A/Resources/python3.12/site-packages \
  QGIS_PREFIX_PATH=$A/MacOS PATH=~/hyperspatial-venv/bin:$PATH \
  $A/MacOS/python3.12 qgis_dialog_smoke.py --cube <BSQ> [--tiles <engine output dir>]
PROJ_DATA matters: without proj.db, GDAL writes the prepared tiles without an
EPSG code and load_tiles drops them (RS_SRID = 0) - a real QGIS session sets it.
spark-submit is found on PATH; QSettings use a throwaway test domain so the
user's real QGIS settings are neither read nor written.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))   # the plugin package's parent

from qgis.core import QgsApplication, QgsProject                   # noqa: E402
from qgis.PyQt.QtCore import QCoreApplication                      # noqa: E402


def pump(app, dialog, timeout_s):
    """Process events until the RUN button re-enables; return the longest stall."""
    start, last, worst = time.time(), time.time(), 0.0
    time.sleep(0.2)
    while time.time() - start < timeout_s:
        app.processEvents()
        now = time.time()
        worst = max(worst, now - last)
        last = now
        if dialog.btn_spark.isEnabled():
            app.processEvents()
            return worst, True
        time.sleep(0.05)
    return worst, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--tiles", default=None,
                    help="Optional engine output dir (tile_r*.tif) to also drive Q3 through the dialog")
    ap.add_argument("--timeout", type=float, default=300)
    a = ap.parse_args()

    app = QgsApplication([], False)
    app.initQgis()
    QCoreApplication.setOrganizationName("HyperSpatialSmokeTest")
    QCoreApplication.setApplicationName("HyperSpatialSmokeTest")
    from hyperspatial_analytics.dialog import SedonaAnalyticsDialog

    dlg = SedonaAnalyticsDialog(iface=None)
    dlg.le_raw.setText(a.cube)
    results = []

    # ---- Q3 on real engine tiles (index-band extraction runs in the worker) ----
    if a.tiles:
        import glob
        n_tiles = len(glob.glob(os.path.join(a.tiles, "tile_r*.tif")))
        dlg.le_rasters.setText(a.tiles)
        dlg.radio_q3.setChecked(True)
        dlg.chk_load_raster.setChecked(True)
        before = set(QgsProject.instance().mapLayers())
        t = time.time()
        dlg.run_analytics()
        stall, done = pump(app, dlg, a.timeout)
        log = dlg.txt_spark_log.toPlainText()
        masks = [l for k, l in QgsProject.instance().mapLayers().items() if k not in before]
        csvs = glob.glob(os.path.join(a.tiles, "sedona_results", "q3_ndvi_filter", "part-*.csv"))
        print("--- Q3 dialog log (job lines) ---\n" + "\n".join(
            l for l in log.splitlines() if l.startswith(("Q3:", "Prepared", "Extracting", "Executing"))
            or "Exception" in l or "Error" in l))
        results.append(("Q3 job completed on full engine tiles",
                        done and "SPARK JOB COMPLETE" in log and bool(csvs), f"{time.time() - t:.1f} s"))
        results.append(("Q3 index-band extraction ran", f"Prepared {n_tiles} index tile(s)" in log,
                        f"{n_tiles} engine tiles"))
        results.append(("Q3 GUI thread never stalled > 1 s", stall < 1.0, f"longest stall {stall:.2f} s"))
        results.append(("Q3 one veg-mask layer per tile", len(masks) == n_tiles, f"{len(masks)} layer(s)"))
        for l in masks:
            QgsProject.instance().removeMapLayer(l.id())

    # ---- Q4 -------------------------------------------------------------
    from hyperspatial_analytics.raster_prep import detect_red_nir
    red, nir, _ = detect_red_nir(a.cube)
    dlg.spin_red.setValue(red)
    dlg.spin_nir.setValue(nir)
    dlg.radio_q4.setChecked(True)
    t = time.time()
    dlg.run_analytics()          # Q4 prepares its tile on the GUI thread (pre-existing)
    stall, done = pump(app, dlg, a.timeout)
    log = dlg.txt_spark_log.toPlainText()
    q4_layers = list(QgsProject.instance().mapLayers().values())
    results.append(("Q4 job completed", done and "SPARK JOB COMPLETE" in log, f"{time.time() - t:.1f} s"))
    results.append(("Q4 NDVI layer added", len(q4_layers) == 1,
                    ", ".join(l.name() for l in q4_layers) or "none"))
    print("--- Q4 dialog log (tail) ---\n" + "\n".join(
        l for l in log.splitlines()[-6:]))

    # ---- Q5 -------------------------------------------------------------
    dlg.radio_q5.setChecked(True)
    dlg.spin_k.setValue(8)
    t = time.time()
    dlg.run_analytics()
    stall, done = pump(app, dlg, a.timeout)
    log = dlg.txt_spark_log.toPlainText()
    q5 = [l for l in QgsProject.instance().mapLayers().values()
          if l.name() == "K-Means land cover (Q5)"]
    results.append(("Q5 job completed", done and "SPARK JOB COMPLETE" in log, f"{time.time() - t:.1f} s"))
    results.append(("Q5 GUI thread never stalled > 1 s", stall < 1.0, f"longest stall {stall:.2f} s"))
    ok_layer = len(q5) == 1 and q5[0].renderer().type() == "paletted"
    legend = q5[0].renderer().classes() if ok_layer else []
    results.append(("Q5 one paletted class-map layer", ok_layer,
                    f"{q5[0].width()}x{q5[0].height()} px" if q5 else "missing"))
    results.append(("Q5 legend has one entry per cluster", len(legend) == 8, f"{len(legend)} entries"))

    print("\nLegend:")
    for c in legend:
        print(f"  {int(c.value):>3}  {c.color.name()}  {c.label}")
    print("\nResults:")
    for name, ok, info in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")
    if not all(ok for _, ok, _ in results):
        print("\n--- dialog log (tail) ---\n" + "\n".join(log.splitlines()[-30:]))
    app.exitQgis()
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
