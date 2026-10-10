# -*- coding: utf-8 -*-
# Standalone QGIS plugin dialog: hyperspectral spectral-index processing.
#
# Ported from the "HPC Preprocessing" tab of the HyperSpatial Big Data monolith
# as a single-purpose QDialog (no tabs, no STAC download, no Sedona analytics).
# It drives an EXTERNAL C++/OpenMP/OpenCL engine (resolved at run time; this
# plugin ships binary-free) to compute spectral indices from a hyperspectral
# raster and loads the resulting GeoTIFF/VRT layers into the QGIS canvas.
#
# AGENTS.md rules honoured here:
#   * The UI thread is NEVER blocked: the engine subprocess + GDAL merge run in a
#     threading.Thread and marshal back via pyqtSignal (hpc_log / hpc_finished).
#   * PyQt5 is imported via qgis.PyQt.
#   * Per-pixel math stays in the native engine (no rasterio / geopandas here).

# Number of spectral index bands the C++ engine appends to every tile GeoTIFF,
# in fixed order: NDVI, NDWI, NDBI, SAVI, EVI, NDMI, MSAVI.
_N_INDEX_BANDS = 7

import os
import sys
import glob
import stat
import subprocess
import threading
import multiprocessing

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QCheckBox, QComboBox, QTextEdit, QFileDialog,
    QRadioButton, QSpinBox
)
from qgis.PyQt.QtCore import pyqtSignal, QSettings
from osgeo import gdal

from .engine_locator import resolve_engine, engine_env, ENGINE_SETTINGS_KEY


class HyperspectralProcessingDialog(QDialog):
    """Single-purpose dialog that runs the C++ engine and builds index layers."""

    # Thread-safe channels from the worker thread back to the GUI thread.
    hpc_log = pyqtSignal(str)
    hpc_finished = pyqtSignal(str)

    def __init__(self, iface, parent=None):
        super(HyperspectralProcessingDialog, self).__init__(parent)
        self.iface = iface
        self.setWindowTitle('HyperSpectral Processer')
        self.resize(820, 560)

        self.init_hpc_ui()

        self.hpc_log.connect(self.append_hpc_log)
        self.hpc_finished.connect(self.on_hpc_finished)

        self._refresh_engine_display()

    # ------------------------------------------------------------------- UI
    def init_hpc_ui(self):
        layout = QVBoxLayout()

        from qgis.gui import QgsMapLayerComboBox
        from qgis.core import QgsMapLayerProxyModel

        # Input raster layer.
        h1 = QHBoxLayout()
        self.cb_input = QgsMapLayerComboBox()
        self.cb_input.setFilters(QgsMapLayerProxyModel.RasterLayer)
        h1.addWidget(QLabel("Input Layer:"))
        h1.addWidget(self.cb_input)
        layout.addLayout(h1)

        # Output directory.
        h2 = QHBoxLayout()
        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output Directory for Tiles and VRT")
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(lambda: self.le_output.setText(QFileDialog.getExistingDirectory(self, "Select Output Folder")))
        h2.addWidget(QLabel("Output:"))
        h2.addWidget(self.le_output)
        h2.addWidget(btn_out)
        layout.addLayout(h2)

        # Engine row: binary-free plugin, so the external engine is resolved at
        # run time; Browse lets the user point at it and persists the choice.
        he = QHBoxLayout()
        self.le_engine = QLineEdit()
        self.le_engine.setReadOnly(True)
        self.le_engine.setPlaceholderText("hyperspatial_cli not found - click Browse to set it")
        btn_engine = QPushButton("Browse")
        btn_engine.setToolTip("Select the compiled hyperspatial_cli engine binary.\n"
                              "The choice is saved for future sessions.")
        btn_engine.clicked.connect(self._browse_engine)
        he.addWidget(QLabel("Engine:"))
        he.addWidget(self.le_engine)
        he.addWidget(btn_engine)
        layout.addLayout(he)

        # Index selection (fixed order NDVI,NDWI,NDBI,SAVI,EVI,NDMI,MSAVI).
        opt = QHBoxLayout()
        self.chk_ndvi = QCheckBox("NDVI"); self.chk_ndvi.setChecked(True)
        self.chk_ndwi = QCheckBox("NDWI"); self.chk_ndwi.setChecked(True)
        self.chk_ndbi = QCheckBox("NDBI"); self.chk_ndbi.setChecked(True)
        self.chk_savi = QCheckBox("SAVI"); self.chk_savi.setChecked(True)
        self.chk_evi = QCheckBox("EVI"); self.chk_evi.setChecked(True)
        self.chk_ndmi = QCheckBox("NDMI"); self.chk_ndmi.setChecked(True)
        self.chk_msavi = QCheckBox("MSAVI"); self.chk_msavi.setChecked(True)
        for w in (self.chk_ndvi, self.chk_ndwi, self.chk_ndbi, self.chk_savi,
                  self.chk_evi, self.chk_ndmi, self.chk_msavi):
            opt.addWidget(w)
        layout.addLayout(opt)
        # Output format.
        fmt = QVBoxLayout()
        fmt.addWidget(QLabel("Output Format:"))
        self.radio_out_vrt = QRadioButton("Virtual Raster (.vrt) - Extremely Fast, saves disk space, keeps chunks")
        self.radio_out_vrt.setChecked(True)
        self.radio_out_tif = QRadioButton("Merged GeoTIFF (.tif) - Slower to save, cleans up temporary chunks")
        fmt.addWidget(self.radio_out_vrt)
        fmt.addWidget(self.radio_out_tif)
        layout.addLayout(fmt)

        self.chk_canvas = QCheckBox("Instantly load results into QGIS Canvas")
        self.chk_canvas.setChecked(True)
        layout.addWidget(self.chk_canvas)

        # Thread/core selection + compute backend.
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("CPU Cores (0 = Auto-detect max):"))
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(0, multiprocessing.cpu_count())
        self.spin_threads.setValue(0)
        h3.addWidget(self.spin_threads)

        # The C++ engine's --mode selects the index backend: openmp = multi-core
        # CPU (default); gpu = cross-vendor OpenCL. GPU transparently falls back
        # to CPU if no OpenCL GPU is available, so it is always safe to pick.
        h3.addSpacing(16)
        h3.addWidget(QLabel("Backend:"))
        self.cb_backend = QComboBox()
        self.cb_backend.addItem("CPU (OpenMP)", "openmp")
        self.cb_backend.addItem("GPU (OpenCL: Apple/NVIDIA/Intel/AMD)", "gpu")
        self.cb_backend.setToolTip(
            "Index compute backend passed to the C++ engine as --mode.\n"
            "GPU uses cross-vendor OpenCL (Apple on macOS, NVIDIA/Intel/AMD on\n"
            "Windows and Linux) and transparently falls back to CPU if no OpenCL\n"
            "GPU is present, so it is always safe to pick.")
        h3.addWidget(self.cb_backend)
        h3.addStretch()
        layout.addLayout(h3)

        self.btn_hpc = QPushButton("RUN HPC C++ ENGINE")
        self.btn_hpc.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_hpc.clicked.connect(self.run_hpc)
        layout.addWidget(self.btn_hpc)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log)

        self.setLayout(layout)

    # --------------------------------------------------------------- engine
    def _refresh_engine_display(self):
        path, source, err = resolve_engine()
        if path:
            self.le_engine.setText(path)
            self.le_engine.setToolTip(f"Resolved from {source}")
            self.le_engine.setStyleSheet("")
        else:
            self.le_engine.clear()
            self.le_engine.setToolTip(err or "Engine not found")
            self.le_engine.setStyleSheet("color: #b00020;")

    def _browse_engine(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select hyperspatial_cli engine binary", os.path.expanduser("~"))
        if path:
            QSettings().setValue(ENGINE_SETTINGS_KEY, path)
            self._refresh_engine_display()
    # ------------------------------------------------------------------ run
    def run_hpc(self):
        layer = self.cb_input.currentLayer()
        if not layer or not self.le_output.text():
            QMessageBox.warning(self, "Warning", "Please select an input layer and specify an output directory.")
            return

        input_path = layer.dataProvider().dataSourceUri()
        if "|" in input_path:
            input_path = input_path.split("|")[0]

        self.txt_log.clear()
        self.btn_hpc.setEnabled(False)

        # Resolve the EXTERNAL engine. Binary-free plugin: if it is not set in
        # QSettings, on PATH, or bundled (dev only), explain how to provide it and
        # re-enable the button instead of crashing.
        cli, source, err = resolve_engine()
        if not cli:
            QMessageBox.critical(self, "Engine not found", err)
            self.hpc_log.emit(f"ERROR: {err}")
            self._refresh_engine_display()
            self.btn_hpc.setEnabled(True)
            return
        self.hpc_log.emit(f"Using engine ({source}): {cli}")

        # Best-effort execute bit (ZIP extraction / a fresh build can drop it on
        # macOS/Linux; harmless no-op on Windows).
        try:
            os.chmod(cli, os.stat(cli).st_mode | stat.S_IEXEC)
        except OSError as e:
            self.hpc_log.emit(f"Warning: could not set execute bit on binary: {e}")

        cmd = [
            cli,
            "--input", input_path,
            "--output", self.le_output.text(),
            "--threads", str(self.spin_threads.value()),
            "--mode", self.cb_backend.currentData(),
        ]

        is_vrt_mode = self.radio_out_vrt.isChecked()
        # Capture checkbox state HERE (GUI thread). Qt widgets must not be read
        # from the worker thread, and the fixed band order must not depend on
        # which boxes happen to be ticked.
        selected = [
            self.chk_ndvi.isChecked(), self.chk_ndwi.isChecked(),
            self.chk_ndbi.isChecked(), self.chk_savi.isChecked(),
            self.chk_evi.isChecked(),  self.chk_ndmi.isChecked(),
            self.chk_msavi.isChecked(),
        ]
        threading.Thread(target=self._hpc_task,
                         args=(cmd, self.le_output.text(), is_vrt_mode, selected),
                         daemon=True).start()
    def _hpc_task(self, cmd, out_dir, is_vrt_mode, selected):
        try:
            self.hpc_log.emit(f"Executing: {' '.join(cmd)}\n")
            # On Windows, launching a console .exe from the GUI would flash a
            # console window; CREATE_NO_WINDOW suppresses it. The flag doesn't
            # exist on POSIX, so it's only added on win32.
            popen_kw = {}
            if sys.platform.startswith('win'):
                popen_kw['creationflags'] = subprocess.CREATE_NO_WINDOW
            # env=engine_env(): prepend QGIS's own GDAL dir so the external engine
            # links the exact GDAL that QGIS uses.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=engine_env(), **popen_kw)
            for line in iter(proc.stdout.readline, ''):
                self.hpc_log.emit(line.strip())
            proc.stdout.close()
            proc.wait()

            # Post-processing merge & cleanup (background thread).
            tiles = sorted(glob.glob(os.path.join(out_dir, "tile_*.tif")))
            if tiles:
                self.hpc_log.emit("\nMerging output tiles into single-index layers...")
                ds = gdal.Open(tiles[0])
                if ds is None:
                    self.hpc_log.emit(f"ERROR: GDAL could not open {tiles[0]}; skipping VRT merge.")
                    return
                n = ds.RasterCount
                ds = None

                # Index bands are the trailing _N_INDEX_BANDS bands, in the FIXED
                # order NDVI,NDWI,NDBI,SAVI,EVI,NDMI,MSAVI. Each layer's band
                # number is first_index_band + its fixed position and must NOT
                # depend on which checkboxes are ticked - otherwise unchecking an
                # earlier index shifts every later layer onto the wrong band.
                first_index_band = n - _N_INDEX_BANDS + 1
                index_names = ["ndvi_layer", "ndwi_layer", "ndbi_layer",
                               "savi_layer", "evi_layer", "ndmi_layer", "msavi_layer"]
                indices = [(index_names[i], first_index_band + i)
                           for i in range(_N_INDEX_BANDS) if selected[i]]

                for name, band_idx in indices:
                    vrt_path = os.path.join(out_dir, f"{name}.vrt")
                    tif_path = os.path.join(out_dir, f"{name}.tif")
                    ds_vrt = gdal.BuildVRT(vrt_path, tiles,
                                           options=gdal.BuildVRTOptions(bandList=[band_idx]))
                    if ds_vrt is None:
                        self.hpc_log.emit(f"WARNING: could not build {name}.vrt (band {band_idx}).")
                        continue
                    ds_vrt.FlushCache()
                    ds_vrt = None

                    if not is_vrt_mode:
                        self.hpc_log.emit(f"Merging {name}.tif...")
                        gdal.Translate(tif_path, vrt_path, options=gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES']))
                        os.remove(vrt_path)

                if not is_vrt_mode:
                    self.hpc_log.emit("Cleaning up temporary chunk files...")
                    for tile in tiles:
                        try:
                            os.remove(tile)
                        except OSError:
                            pass
        except Exception as e:
            self.hpc_log.emit(f"Process Failed: {e}")
        finally:
            # Always emit so the UI re-enables the button and (on success) loads
            # the results - even if the C++ step or the GDAL merge raised.
            self.hpc_finished.emit(out_dir)

    # ------------------------------------------------------------ GUI slots
    def append_hpc_log(self, msg):
        self.txt_log.append(msg)

    def on_hpc_finished(self, out_dir):
        # try/finally so the RUN button ALWAYS re-enables, even if the canvas
        # load below raises for any layer.
        try:
            self.hpc_log.emit("\n--- HPC COMPLETE ---")
            if self.chk_canvas.isChecked():
                self.hpc_log.emit("Loading computed index layers into canvas...")

                layers_to_load = []
                is_vrt = self.radio_out_vrt.isChecked()
                ext = ".vrt" if is_vrt else ".tif"

                if self.chk_ndvi.isChecked(): layers_to_load.append((os.path.join(out_dir, f"ndvi_layer{ext}"), "Computed NDVI"))
                if self.chk_ndwi.isChecked(): layers_to_load.append((os.path.join(out_dir, f"ndwi_layer{ext}"), "Computed NDWI"))
                if self.chk_ndbi.isChecked(): layers_to_load.append((os.path.join(out_dir, f"ndbi_layer{ext}"), "Computed NDBI"))
                if self.chk_savi.isChecked(): layers_to_load.append((os.path.join(out_dir, f"savi_layer{ext}"), "Computed SAVI"))
                if self.chk_evi.isChecked(): layers_to_load.append((os.path.join(out_dir, f"evi_layer{ext}"), "Computed EVI"))
                if self.chk_ndmi.isChecked(): layers_to_load.append((os.path.join(out_dir, f"ndmi_layer{ext}"), "Computed NDMI"))
                if self.chk_msavi.isChecked(): layers_to_load.append((os.path.join(out_dir, f"msavi_layer{ext}"), "Computed MSAVI"))

                for path, name in layers_to_load:
                    if not os.path.exists(path):
                        continue
                    # iface.addRasterLayer guarantees the layer lands in the layer
                    # tree (legend) + canvas, not just the project registry.
                    layer = self.iface.addRasterLayer(path, name)
                    if layer and layer.isValid():
                        from qgis.core import QgsContrastEnhancement
                        renderer = layer.renderer()
                        if renderer:
                            enhancement = QgsContrastEnhancement(layer.dataProvider().dataType(1))
                            enhancement.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum)
                            # Indices like NDVI typically range from -1 to 1.
                            enhancement.setMinimumValue(-1.0)
                            enhancement.setMaximumValue(1.0)
                            renderer.setContrastEnhancement(enhancement)

                        layer.triggerRepaint()
                        self.iface.mapCanvas().refresh()
                        self.hpc_log.emit(f"Loaded {name} to map canvas.")
        finally:
            self.btn_hpc.setEnabled(True)
