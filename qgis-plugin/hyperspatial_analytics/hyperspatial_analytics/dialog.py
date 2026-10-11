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
    QRadioButton, QDoubleSpinBox, QTextEdit, QFileDialog, QMessageBox,
)
from qgis.PyQt.QtCore import pyqtSignal, QSettings

from .spark_locator import resolve_spark_submit, spark_env, SPARK_SETTINGS_KEY


class SedonaAnalyticsDialog(QDialog):

    # Thread-safe bridges from the Spark worker thread back to the GUI thread.
    spark_log = pyqtSignal(str)
    spark_finished = pyqtSignal()

    def __init__(self, iface, parent=None):
        super(SedonaAnalyticsDialog, self).__init__(parent)
        self.iface = iface
        self.setWindowTitle('HyperSpatial Sedona Analytics')
        self.resize(850, 600)

        self.init_analytics_ui()

        self.spark_log.connect(self.append_spark_log)
        self.spark_finished.connect(self.on_spark_finished)

    def append_spark_log(self, msg):
        self.txt_spark_log.append(msg)

    def on_spark_finished(self):
        # Re-enable the button from the GUI thread (safe: driven by a signal).
        self.btn_spark.setEnabled(True)

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

        layout.addWidget(self.radio_q1)
        layout.addWidget(self.radio_q2)
        layout.addWidget(self.radio_q3)

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

    def run_analytics(self):
        import glob as _glob
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
        if script_name == "q3_ndvi_filter.py":
            spark_args = ["spark-submit", script_path,
                          "--rasters", rasters_dir,
                          "--output", out_dir,
                          "--threshold", str(self.spin_ndvi.value())]
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
        cmd = [spark_submit] + spark_args[1:]

        threading.Thread(target=self._spark_task,
                         args=(cmd, out_dir, spark_submit), daemon=True).start()

    def _spark_task(self, cmd, out_dir, spark_submit=None):
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
            else:
                self.spark_log.emit(f"\n--- SPARK JOB FAILED (exit code {rc}) ---")
        except Exception as e:
            self.spark_log.emit(f"Process Failed: {str(e)}")
        finally:
            # Re-enable the button via a signal (GUI widgets must not be touched
            # directly from this worker thread).
            self.spark_finished.emit()
