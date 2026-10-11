# -*- coding: utf-8 -*-
"""Single-purpose QGIS dialog: distributed per-pixel spatial SQL over the
hyperspectral index rasters produced by the HyperSpatial processing engine,
via PySpark + Apache Sedona.

Logic-preserving extraction of the "Sedona Analytics" tab from the original
3-tab HyperSpatialDialog monolith. Only this concern is kept here: no STAC
download, no HPC preprocessing, no QTabWidget. The two CRITICAL rules of the
monolith are preserved verbatim:
  * the Spark child process runs with HADOOP_CONF_DIR="" (prevents a macOS
    local-XML-parse crash of the JVM), and
  * spark-submit is resolved with shutil.which first, falling back to the
    user's login shell ($SHELL -l -c) when QGIS was launched from the GUI and
    spark-submit is not on PATH.
Long work runs in a threading.Thread and marshals back to the GUI via the
spark_log / spark_finished pyqtSignals; Qt widgets are only touched from slots.
"""
import os
import sys
import subprocess
import threading

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QDoubleSpinBox, QSpinBox, QTextEdit, QFileDialog, QMessageBox, QCheckBox,
)
from qgis.PyQt.QtCore import pyqtSignal, QSettings
from qgis.core import QgsRasterLayer, QgsProject

from .spark_locator import resolve_spark_submit, spark_env, SPARK_SETTINGS_KEY

# Sedona JVM jars for the bundled raster jobs. These MUST go on the spark-submit
# command line (--packages): setting spark.jars.packages programmatically in
# SedonaContext has no effect once spark-submit has already started the JVM, so
# the jars would never reach the driver classpath (RS_* -> ClassNotFound).
# geotools-wrapper 1.5.1-28.2 is required - it bundles jt-jiffle, which
# RS_MapAlgebra needs; 1.5.0 does not. Keep in sync with
# spark-sedona/utils/sedona_session.py.
SEDONA_PACKAGES = ("org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.5.1,"
                   "org.datasyslab:geotools-wrapper:1.5.1-28.2")
SEDONA_REPOSITORIES = "https://artifacts.unidata.ucar.edu/repository/unidata-all"


class SedonaAnalyticsDialog(QDialog):

    # Thread-safe bridges from the Spark worker thread back to the GUI thread.
    spark_log = pyqtSignal(str)
    spark_finished = pyqtSignal()
    rasters_ready = pyqtSignal(str)

    def __init__(self, iface, parent=None):
        super(SedonaAnalyticsDialog, self).__init__(parent)
        self.iface = iface
        self.setWindowTitle('HyperSpatial Sedona Analytics')
        self.resize(850, 600)

        self.init_analytics_ui()

        self.spark_log.connect(self.append_spark_log)
        self.spark_finished.connect(self.on_spark_finished)
        self.rasters_ready.connect(self.on_rasters_ready)

    def append_spark_log(self, msg):
        self.txt_spark_log.append(msg)

    def on_spark_finished(self):
        # Re-enable the button from the GUI thread (safe: driven by a signal).
        self.btn_spark.setEnabled(True)

    def on_rasters_ready(self, raster_dir):
        """Load any veg-mask GeoTIFFs the job wrote onto the QGIS canvas.

        Runs on the GUI thread (signal slot) - QgsProject/QgsRasterLayer must not
        be touched from the Spark worker thread.
        """
        import glob as _glob
        # Sedona's native raster writer lays files out as <dir>/part-*/<uuid>.tif,
        # so search recursively, not just the top level.
        tifs = sorted(_glob.glob(os.path.join(raster_dir, "**", "*.tif"), recursive=True))
        if not tifs:
            self.txt_spark_log.append(f"No result rasters found in {raster_dir}")
            return
        loaded = 0
        for tif in tifs:
            name = os.path.splitext(os.path.basename(tif))[0]
            layer = QgsRasterLayer(tif, name)
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                loaded += 1
        self.txt_spark_log.append(f"Loaded {loaded} result raster(s) onto the canvas.")

    def _browse_raw_cube(self):
        path = QFileDialog.getOpenFileName(
            self, "Select raw hyperspectral cube",
            filter="Rasters (*.bsq *.img *.dat *.tif *.tiff *.hdr);;All files (*)")[0]
        if not path:
            return
        # ENVI users often pick the .hdr; GDAL wants the data file next to it.
        if path.lower().endswith(".hdr"):
            for ext in (".bsq", ".img", ".dat", ".raw"):
                cand = path[:-4] + ext
                if os.path.exists(cand):
                    path = cand
                    break
        self.le_raw.setText(path)
        self.radio_q4.setChecked(True)
        try:
            from .raster_prep import detect_red_nir
            red, nir, nb = detect_red_nir(path)
            if nb:
                self.spin_red.setRange(1, nb)
                self.spin_nir.setRange(1, nb)
            if red and nir:
                self.spin_red.setValue(red)
                self.spin_nir.setValue(nir)
                self.txt_spark_log.append(
                    f"Auto-detected RED=band {red}, NIR=band {nir} from wavelengths "
                    f"({nb} bands). Adjust if needed.")
            else:
                self.txt_spark_log.append(
                    f"No wavelength metadata in {os.path.basename(path)} ({nb} bands) "
                    f"- set the RED / NIR band numbers manually.")
        except Exception as e:
            self.txt_spark_log.append(f"Could not inspect cube: {e}")

    # =========================================================================
    # SEDONA ANALYTICS UI
    # =========================================================================
    def init_analytics_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select PySpark Query (per-pixel, via Sedona raster):"))

        self.radio_q1 = QRadioButton("AOI Filter: per-pixel NDVI stats for tiles intersecting an Area of Interest.")
        self.radio_q2 = QRadioButton("Spatial Join: per-pixel zonal stats of NDVI/NDWI/NDBI per field boundary.")
        self.radio_q2.setChecked(True)
        self.radio_q3 = QRadioButton("NDVI Filter: per-pixel vegetation fraction / mean NDVI per tile.")
        self.radio_q4 = QRadioButton("Raw cube NDVI: compute NDVI per-pixel straight from a BSQ/ENVI/GeoTIFF cube (no engine).")

        layout.addWidget(self.radio_q1)
        layout.addWidget(self.radio_q2)
        layout.addWidget(self.radio_q3)
        layout.addWidget(self.radio_q4)

        # Input: the engine's output DIRECTORY of tile_r*.tif rasters. Sedona
        # reads the real GeoTIFFs (RS_FromGeoTiff) and computes on actual pixels,
        # so there is no summary.csv dependency. Use VRT output mode in the HPC
        # tab so the per-tile GeoTIFFs are retained (merged-TIFF mode deletes them).
        h1 = QHBoxLayout()
        self.le_rasters = QLineEdit()
        self.le_rasters.setPlaceholderText("Tiles directory (contains tile_r*.tif from the HPC tab)")
        btn_rasters = QPushButton("Browse")
        btn_rasters.clicked.connect(lambda: self.le_rasters.setText(QFileDialog.getExistingDirectory(self, "Select tiles directory")))
        h1.addWidget(QLabel("Tiles dir:"))
        h1.addWidget(self.le_rasters)
        h1.addWidget(btn_rasters)
        layout.addLayout(h1)

        # AOI / agricultural-boundaries GeoJSON (required by Q1 and Q2),
        # EPSG:4326 per the GeoJSON spec. Sedona reprojects it into each raster's
        # CRS automatically for the zonal statistics.
        h2 = QHBoxLayout()
        self.le_aoi = QLineEdit()
        self.le_aoi.setPlaceholderText("AOI / boundaries GeoJSON (required for AOI Filter & Spatial Join)")
        btn_aoi = QPushButton("Browse")
        btn_aoi.clicked.connect(lambda: self.le_aoi.setText(QFileDialog.getOpenFileName(self, "Select AOI / boundaries GeoJSON", filter="GeoJSON (*.geojson *.json)")[0]))
        h2.addWidget(QLabel("AOI/Agri:"))
        h2.addWidget(self.le_aoi)
        h2.addWidget(btn_aoi)
        layout.addLayout(h2)

        # NDVI threshold for Q3.
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("NDVI threshold (Q3):"))
        self.spin_ndvi = QDoubleSpinBox()
        self.spin_ndvi.setRange(-1.0, 1.0)
        self.spin_ndvi.setSingleStep(0.05)
        self.spin_ndvi.setValue(0.4)
        h3.addWidget(self.spin_ndvi)
        h3.addStretch()
        layout.addLayout(h3)

        # Q3 only: materialise the per-pixel vegetation mask as GeoTIFFs and drop
        # them on the canvas (pixel-based raster processing you can see).
        self.chk_load_raster = QCheckBox(
            "Q3: write per-pixel veg-mask GeoTIFFs and load them onto the canvas")
        layout.addWidget(self.chk_load_raster)

        # Raw-cube NDVI (Q4): a single hyperspectral file (BSQ/ENVI/GeoTIFF). RED
        # and NIR are auto-detected from the cube's wavelength metadata when the
        # file is chosen; both are editable. The result NDVI raster always loads
        # onto the canvas.
        h4 = QHBoxLayout()
        self.le_raw = QLineEdit()
        self.le_raw.setPlaceholderText("Raw cube file (BSQ/ENVI/GeoTIFF) - for Raw cube NDVI")
        btn_raw = QPushButton("Browse")
        btn_raw.clicked.connect(self._browse_raw_cube)
        h4.addWidget(QLabel("Raw cube:"))
        h4.addWidget(self.le_raw)
        h4.addWidget(btn_raw)
        layout.addLayout(h4)

        h5 = QHBoxLayout()
        h5.addWidget(QLabel("RED band:"))
        self.spin_red = QSpinBox()
        self.spin_red.setRange(1, 100000)
        h5.addWidget(self.spin_red)
        h5.addWidget(QLabel("NIR band:"))
        self.spin_nir = QSpinBox()
        self.spin_nir.setRange(1, 100000)
        self.spin_nir.setValue(2)
        h5.addWidget(self.spin_nir)
        h5.addStretch()
        layout.addLayout(h5)

        # Spark locator row (mirrors the processing plugin's engine Browse):
        # resolve spark-submit explicitly so a Finder-launched QGIS - which does
        # NOT inherit the login-shell PATH on macOS - can still find it. The
        # chosen path is remembered in QSettings.
        h_spark = QHBoxLayout()
        self.le_spark = QLineEdit()
        self.le_spark.setReadOnly(True)
        btn_spark_browse = QPushButton("Browse")
        btn_spark_browse.clicked.connect(self._browse_spark_submit)
        h_spark.addWidget(QLabel("Spark:"))
        h_spark.addWidget(self.le_spark)
        h_spark.addWidget(btn_spark_browse)
        layout.addLayout(h_spark)

        self.btn_spark = QPushButton("RUN SPARK JOB")
        self.btn_spark.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_spark.clicked.connect(self.run_analytics)
        layout.addWidget(self.btn_spark)

        self.txt_spark_log = QTextEdit()
        self.txt_spark_log.setReadOnly(True)
        layout.addWidget(self.txt_spark_log)

        self.setLayout(layout)
        self._spark_submit = None
        self._refresh_spark_label()

    def _refresh_spark_label(self):
        """Resolve spark-submit and reflect it in the Spark row (red if missing)."""
        path, source, _err = resolve_spark_submit()
        self._spark_submit = path
        if path:
            self.le_spark.setStyleSheet("")
            self.le_spark.setText("%s  (%s)" % (path, source))
        else:
            self.le_spark.setStyleSheet("color: #c0392b;")
            self.le_spark.setText("spark-submit not found - click Browse to set it")

    def _browse_spark_submit(self):
        start = "/opt/homebrew/bin" if sys.platform == "darwin" else ""
        path = QFileDialog.getOpenFileName(self, "Select spark-submit", start)[0]
        if path:
            QSettings().setValue(SPARK_SETTINGS_KEY, path)
            self._refresh_spark_label()

    def _run_raw_cube(self):
        """Q4: compute NDVI per-pixel straight from a BSQ/ENVI/GeoTIFF cube.

        Extract RED+NIR into a GeoTIFF (QGIS GDAL), then run the Sedona raw-NDVI
        job on it. Independent of the C++ engine and the processing plugin.
        """
        import tempfile
        src = self.le_raw.text().strip()
        if not src or not os.path.isfile(src):
            QMessageBox.warning(self, "Warning",
                "Pick a raw cube file (BSQ/ENVI/GeoTIFF) with the Raw cube Browse button.")
            return
        self.txt_spark_log.clear()
        self.btn_spark.setEnabled(False)

        jobs_dir = os.path.join(os.path.dirname(__file__), 'spark-sedona', 'jobs')
        script_path = os.path.join(jobs_dir, "q_raw_ndvi.py")
        if not os.path.exists(script_path):
            self.txt_spark_log.append("ERROR: q_raw_ndvi.py not found in the bundled jobs.")
            self.btn_spark.setEnabled(True)
            return

        work = tempfile.mkdtemp(prefix="hs_rawndvi_")
        tiles_dir = os.path.join(work, "tiles")
        try:
            from .raster_prep import prepare_red_nir_tile
            prepare_red_nir_tile(src, tiles_dir, self.spin_red.value(), self.spin_nir.value())
        except Exception as e:
            self.txt_spark_log.append(f"ERROR preparing RED/NIR GeoTIFF: {e}")
            self.btn_spark.setEnabled(True)
            return

        out_dir = os.path.join(work, "sedona_results")
        raster_out = os.path.join(work, "ndvi_raster")
        # prepare_red_nir_tile wrote RED as band 1, NIR as band 2.
        spark_args = ["spark-submit", script_path,
                      "--rasters", tiles_dir, "--output", out_dir,
                      "--red-band", "1", "--nir-band", "2",
                      "--threshold", str(self.spin_ndvi.value()),
                      "--raster-out", raster_out]

        spark_submit, _source, err = resolve_spark_submit()
        if not spark_submit:
            QMessageBox.warning(self, "spark-submit not found", err)
            self.btn_spark.setEnabled(True)
            self._refresh_spark_label()
            return
        self._refresh_spark_label()
        cmd = ([spark_submit, "--packages", SEDONA_PACKAGES,
                "--repositories", SEDONA_REPOSITORIES] + spark_args[1:])
        self.txt_spark_log.append(
            f"Raw cube NDVI: RED=band {self.spin_red.value()}, NIR=band "
            f"{self.spin_nir.value()} -> {os.path.basename(src)}")
        threading.Thread(target=self._spark_task,
                         args=(cmd, out_dir, spark_submit, raster_out), daemon=True).start()

    def run_analytics(self):
        import glob as _glob
        if self.radio_q4.isChecked():
            self._run_raw_cube()
            return
        rasters_dir = self.le_rasters.text().strip()
        if not rasters_dir:
            QMessageBox.warning(self, "Warning", "Please specify the tiles directory (the HPC output folder).")
            return
        if not os.path.isdir(rasters_dir):
            QMessageBox.warning(self, "Warning", f"Not a directory:\n{rasters_dir}")
            return
        if not _glob.glob(os.path.join(rasters_dir, "tile_r*.tif")):
            QMessageBox.warning(self, "Warning",
                "No tile_r*.tif found in that directory.\n\n"
                "Run the HPC engine first, and use the Virtual Raster (.vrt) output "
                "mode so the per-tile GeoTIFFs are kept (merged-TIFF mode deletes them).")
            return

        self.txt_spark_log.clear()
        self.btn_spark.setEnabled(False)

        # The plugin is self-contained: the Sedona raster jobs are bundled.
        jobs_dir = os.path.join(os.path.dirname(__file__), 'spark-sedona', 'jobs')
        if not os.path.exists(os.path.join(jobs_dir, 'q1_aoi_filter.py')):
            self.txt_spark_log.append("ERROR: Could not find bundled spark-sedona directory inside plugin.")
            self.btn_spark.setEnabled(True)
            return

        if self.radio_q1.isChecked():
            script_name = "q1_aoi_filter.py"
        elif self.radio_q2.isChecked():
            script_name = "q2_spatial_join.py"
        else:
            script_name = "q3_ndvi_filter.py"
        script_path = os.path.join(jobs_dir, script_name)

        # Each job writes a Spark CSV output *directory*; keep them under the
        # tiles directory, one sub-folder per query, overwritten on re-run.
        out_dir = os.path.join(rasters_dir, "sedona_results", os.path.splitext(script_name)[0])
        try:
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        except OSError as e:
            self.txt_spark_log.append(f"ERROR: cannot create output directory: {e}")
            self.btn_spark.setEnabled(True)
            return

        # Match each raster job's argparse contract exactly:
        #   q1_aoi_filter.py  : --rasters <dir> --output <dir> --aoi  <geojson>
        #   q2_spatial_join.py: --rasters <dir> --agri   <geojson> --output <dir>
        #   q3_ndvi_filter.py : --rasters <dir> --output <dir> --threshold <float>
        raster_out = None
        if script_name == "q3_ndvi_filter.py":
            spark_args = ["spark-submit", script_path,
                          "--rasters", rasters_dir,
                          "--output", out_dir,
                          "--threshold", str(self.spin_ndvi.value())]
            if self.chk_load_raster.isChecked():
                # Sibling dir (NOT inside the Spark CSV output dir, which Spark
                # overwrites) to hold the per-pixel veg-mask GeoTIFFs.
                raster_out = out_dir + "_rasters"
                spark_args += ["--raster-out", raster_out]
        else:
            # Q1 and Q2 require an AOI / boundaries GeoJSON (no demo fallback).
            aoi_path = self.le_aoi.text().strip()
            if not aoi_path:
                QMessageBox.warning(self, "Warning",
                    "This query needs an AOI / boundaries GeoJSON. Pick one with the "
                    "AOI/Agri browse button.")
                self.btn_spark.setEnabled(True)
                return
            if not os.path.exists(aoi_path):
                QMessageBox.warning(self, "Warning", f"AOI file not found:\n{aoi_path}")
                self.btn_spark.setEnabled(True)
                return
            if script_name == "q1_aoi_filter.py":
                spark_args = ["spark-submit", script_path,
                              "--rasters", rasters_dir, "--output", out_dir, "--aoi", aoi_path]
            else:  # q2_spatial_join.py
                spark_args = ["spark-submit", script_path,
                              "--rasters", rasters_dir, "--agri", aoi_path, "--output", out_dir]

        # Resolve spark-submit via the shared locator (QSettings -> SPARK_HOME ->
        # PATH -> common dirs -> login shell), mirroring the engine Browse flow.
        # This fixes the Finder-launched-QGIS case where the GUI has no shell PATH.
        spark_submit, _source, err = resolve_spark_submit()
        if not spark_submit:
            QMessageBox.warning(self, "spark-submit not found", err)
            self.btn_spark.setEnabled(True)
            self._refresh_spark_label()
            return
        self._refresh_spark_label()
        cmd = ([spark_submit, "--packages", SEDONA_PACKAGES,
                "--repositories", SEDONA_REPOSITORIES] + spark_args[1:])

        threading.Thread(target=self._spark_task,
                         args=(cmd, out_dir, spark_submit, raster_out), daemon=True).start()

    def _spark_task(self, cmd, out_dir, spark_submit=None, raster_out=None):
        try:
            self.spark_log.emit(f"Executing: {' '.join(cmd)}\n")
            # spark_env() unsets HADOOP_CONF_DIR (a populated value crashes the
            # JVM on macOS), sets JAVA_HOME, and prepends the spark-submit / conda
            # bin dirs to PATH so the child finds python/pyspark/java even when
            # QGIS was launched from Finder without a shell environment.
            env = spark_env(spark_submit)

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in iter(proc.stdout.readline, ''):
                self.spark_log.emit(line.strip())
            proc.stdout.close()
            rc = proc.wait()
            if rc == 0:
                self.spark_log.emit(f"\n--- SPARK JOB COMPLETE ---\nResults written to: {out_dir}")
                # Load result rasters on the GUI thread (never from here).
                if raster_out:
                    self.rasters_ready.emit(raster_out)
            else:
                self.spark_log.emit(f"\n--- SPARK JOB FAILED (exit code {rc}) ---")
        except Exception as e:
            self.spark_log.emit(f"Process Failed: {str(e)}")
        finally:
            # Re-enable the button via a signal (GUI widgets must not be touched
            # directly from this worker thread).
            self.spark_finished.emit()
