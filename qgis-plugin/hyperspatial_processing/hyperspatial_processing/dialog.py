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
import ssl
import glob
import stat
import math
import zipfile
import contextlib
import tempfile
import subprocess  # nosec B404 - runs the external HPC engine with a fixed argv, no shell
import threading
import multiprocessing
import urllib.error
import urllib.request

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QCheckBox, QComboBox, QTextEdit, QFileDialog,
    QRadioButton, QSpinBox, QDoubleSpinBox, QStackedWidget, QWidget
)
from qgis.PyQt.QtCore import pyqtSignal, QSettings
from osgeo import gdal

from .engine_locator import (
    resolve_engine, engine_env, ENGINE_SETTINGS_KEY, _platform_exe,
)

# Prebuilt engine releases live on GitHub Releases; "Download engine" fetches the
# asset matching the host platform so a store-installed (binary-free) plugin can
# obtain a working engine in one click.
_ENGINE_RELEASE_BASE = (
    "https://github.com/aaron43210/HYPERSPECTRAL_DATA_DISTRIBUTED_SYSTEM"
    "/releases/latest/download"
)


def _engine_asset_for_host():
    """Return ``(os_name, arch, asset_name, url)`` for the running host.

    Mapping: darwin -> macos/arm64, win32 -> windows/x64, else linux/x64.
    """
    host = sys.platform
    if host == 'darwin':
        os_name, arch = 'macos', 'arm64'
    elif host.startswith('win'):
        os_name, arch = 'windows', 'x64'
    else:
        os_name, arch = 'linux', 'x64'
    asset = f"engine-{os_name}-{arch}.zip"
    return os_name, arch, asset, f"{_ENGINE_RELEASE_BASE}/{asset}"


def _make_ssl_context():
    """Verified TLS context: prefer certifi's CA bundle, else the system default.

    Mirrors services/enmap_client.py: macOS Python builds often ship no root CA
    bundle, so certifi is preferred; either way certificate verification stays ON.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class HyperspectralProcessingDialog(QDialog):
    """Single-purpose dialog that runs the C++ engine and builds index layers."""

    # Thread-safe channels from the worker thread back to the GUI thread.
    hpc_log = pyqtSignal(str)
    hpc_finished = pyqtSignal(str)
    # Engine-download worker -> GUI: carries success flag so the slot re-enables
    # the Download button and refreshes the engine display on the GUI thread.
    engine_download_finished = pyqtSignal(bool)

    def __init__(self, iface, parent=None):
        super(HyperspectralProcessingDialog, self).__init__(parent)
        self.iface = iface
        self.setWindowTitle('HyperSpectral Processer')
        self.resize(820, 560)

        # SAM reference spectrum captured by clicking a pixel on the map canvas.
        # Populated on the GUI thread by _on_sam_canvas_clicked; read in run_hpc.
        self._sam_spectrum = None
        self._sam_rc = None
        self._sam_pick_tool = None
        # Operation + SAM threshold captured on the GUI thread at RUN time so the
        # finished-slot can pick renderers without re-reading operation widgets.
        self._last_operation = "indices"
        self._sam_threshold = None

        self.init_hpc_ui()

        self.hpc_log.connect(self.append_hpc_log)
        self.hpc_finished.connect(self.on_hpc_finished)
        self.engine_download_finished.connect(self.on_engine_download_finished)

        self._refresh_engine_display()

    # ------------------------------------------------------------------- UI
    def init_hpc_ui(self):
        layout = QVBoxLayout()

        from qgis.gui import QgsMapLayerComboBox
        from qgis.core import QgsMapLayerProxyModel

        # Input raster layer.
        h1 = QHBoxLayout()
        self.cb_input = QgsMapLayerComboBox()
        self.cb_input.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
        h1.addWidget(QLabel("Input Layer:"))
        h1.addWidget(self.cb_input)
        layout.addLayout(h1)

        # Output directory.
        h2 = QHBoxLayout()
        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output Directory (leave blank for QGIS temporary scratch space)")
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
        self.btn_engine_download = QPushButton("Download engine")
        self.btn_engine_download.setToolTip(
            "Download the prebuilt engine for your platform from GitHub Releases\n"
            "and install it so this plugin can run without building from source.")
        self.btn_engine_download.clicked.connect(self._download_engine)
        he.addWidget(QLabel("Engine:"))
        he.addWidget(self.le_engine)
        he.addWidget(btn_engine)
        he.addWidget(self.btn_engine_download)
        layout.addLayout(he)

        # ---- Operation selector + per-operation page stack ------------------
        # One dialog, no clutter: a combo chooses the operation and the stacked
        # widget shows only that operation's controls. The SHARED controls
        # (input, output, engine, format, cores, backend, RUN, log) stay OUTSIDE
        # the stack so they apply to every operation.
        hop = QHBoxLayout()
        self.cb_operation = QComboBox()
        self.cb_operation.addItem("Spectral Indices", "indices")
        self.cb_operation.addItem("PCA (false-colour composite)", "pca")
        self.cb_operation.addItem("Target Detection (SAM)", "sam")
        hop.addWidget(QLabel("Operation:"))
        hop.addWidget(self.cb_operation)
        hop.addStretch()
        layout.addLayout(hop)

        self.stack_op = QStackedWidget()

        # --- Page 0: Spectral indices (fixed order NDVI..MSAVI) --------------
        page_indices = QWidget()
        pi = QVBoxLayout(page_indices)
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
        opt.addStretch()
        pi.addWidget(QLabel("Spectral indices to compute (fixed band order):"))
        pi.addLayout(opt)
        pi.addStretch()
        self.stack_op.addWidget(page_indices)

        # --- Page 1: PCA -----------------------------------------------------
        page_pca = QWidget()
        pp = QVBoxLayout(page_pca)
        hpca = QHBoxLayout()
        hpca.addWidget(QLabel("Components (PC bands):"))
        self.spin_pca_components = QSpinBox()
        self.spin_pca_components.setRange(1, 224)
        self.spin_pca_components.setValue(3)
        self.spin_pca_components.setToolTip(
            "Number of principal components (PC1..PCk) the engine writes to pca.tif.\n"
            "3 gives an RGB false-colour composite out of the box.")
        hpca.addWidget(self.spin_pca_components)
        hpca.addStretch()
        pp.addLayout(hpca)
        self.chk_pca_correlation = QCheckBox("Use correlation matrix (advanced)")
        self.chk_pca_correlation.setChecked(False)
        self.chk_pca_correlation.setToolTip(
            "Off (default): covariance matrix. On: correlation matrix, which\n"
            "normalises per-band variance before decomposition.")
        pp.addWidget(self.chk_pca_correlation)
        pp.addStretch()
        self.stack_op.addWidget(page_pca)

        # --- Page 2: Target Detection (SAM) ----------------------------------
        page_sam = QWidget()
        ps = QVBoxLayout(page_sam)
        hsam1 = QHBoxLayout()
        self.btn_sam_pick = QPushButton("Pick reference pixel")
        self.btn_sam_pick.setToolTip(
            "Click, then click a pixel on the map canvas. All band values at that\n"
            "pixel become the SAM reference spectrum.")
        self.btn_sam_pick.clicked.connect(self._activate_sam_picker)
        self.lbl_sam_ref = QLabel("No reference captured.")
        hsam1.addWidget(self.btn_sam_pick)
        hsam1.addWidget(self.lbl_sam_ref)
        hsam1.addStretch()
        ps.addLayout(hsam1)

        hsam2 = QHBoxLayout()
        hsam2.addWidget(QLabel("Max angle:"))
        self.spin_sam_angle = QDoubleSpinBox()
        self.spin_sam_angle.setRange(0.0, 1.57)
        self.spin_sam_angle.setValue(0.10)
        self.spin_sam_angle.setSingleStep(0.01)
        self.spin_sam_angle.setSuffix(" rad")
        self.spin_sam_angle.valueChanged.connect(self._update_sam_angle_label)
        hsam2.addWidget(self.spin_sam_angle)
        self.lbl_sam_degrees = QLabel("")
        hsam2.addWidget(self.lbl_sam_degrees)
        hsam2.addStretch()
        ps.addLayout(hsam2)
        self._update_sam_angle_label(self.spin_sam_angle.value())

        self.chk_sam_mask = QCheckBox("Write detection mask (applies Max angle threshold)")
        self.chk_sam_mask.setChecked(True)
        self.chk_sam_mask.setToolTip(
            "On: pass --sam-threshold so the engine also writes a 0/1 sam_mask.tif.")
        ps.addWidget(self.chk_sam_mask)
        self.chk_sam_similarity = QCheckBox("Write similarity band")
        self.chk_sam_similarity.setChecked(False)
        self.chk_sam_similarity.setToolTip(
            "On: pass --sam-emit-similarity so the engine also writes sam_similarity.tif.")
        ps.addWidget(self.chk_sam_similarity)
        ps.addStretch()
        self.stack_op.addWidget(page_sam)

        layout.addWidget(self.stack_op)
        self.cb_operation.currentIndexChanged.connect(self.stack_op.setCurrentIndex)
        self.stack_op.setCurrentIndex(0)
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

        self.btn_hpc = QPushButton("RUN")
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

    # ------------------------------------------------------------ engine download
    def _download_engine(self):
        """One-click fetch+install of the prebuilt engine for the host platform.

        Resolves the release asset URL and target bin/<plat> dir on the GUI
        thread, then streams the download + install on a background thread so the
        UI never blocks (AGENTS.md). Widgets are only touched here / in the
        finished slot, never from the worker.
        """
        plat, exe = _platform_exe()
        os_name, arch, asset, url = _engine_asset_for_host()
        bin_dir = os.path.join(os.path.dirname(__file__), 'bin', plat)

        self.btn_engine_download.setEnabled(False)
        self.hpc_log.emit(f"Downloading engine for {os_name}/{arch}...")
        self.hpc_log.emit(f"Asset: {asset}")
        self.hpc_log.emit(f"URL: {url}")
        threading.Thread(target=self._download_engine_task,
                         args=(url, bin_dir, exe, plat),
                         daemon=True).start()

    def _download_engine_task(self, url, bin_dir, exe, plat):
        """Worker: download the release zip over verified TLS and install it.

        Streams to a temp file (constant RAM), extracts into bin/<plat>, sets the
        execute bit, and clears the macOS Gatekeeper quarantine flag. Emits only
        signals back to the GUI thread; never touches widgets directly.
        """
        success = False
        tmp_path = None
        try:
            ctx = _make_ssl_context()
            req = urllib.request.Request(
                url, headers={"User-Agent": "hyperspatial-processing-plugin"})
            with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:  # nosec B310 - fixed https GitHub release URL
                total = int(resp.headers.get("Content-Length", 0) or 0)
                fd, tmp_path = tempfile.mkstemp(prefix="hyperspatial_engine_",
                                                suffix=".zip")
                downloaded = 0
                next_report = 10
                chunk_size = 1 << 20  # 1 MB
                with os.fdopen(fd, "wb") as f:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        if total:
                            pct = int(downloaded * 100 / total)
                            if pct >= next_report:
                                self.hpc_log.emit(
                                    f"Downloaded {pct}% "
                                    f"({downloaded}/{total} bytes)")
                                next_report = pct - (pct % 10) + 10
                self.hpc_log.emit(f"Downloaded {downloaded} bytes; extracting...")

            os.makedirs(bin_dir, exist_ok=True)
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(bin_dir)  # nosec B202 - trusted first-party release asset

            # Locate the engine binary: expected directly in bin/<plat>, but tolerate
            # a zip that nests it one or more levels deep.
            engine_path = os.path.join(bin_dir, exe)
            if not os.path.isfile(engine_path):
                engine_path = None
                for root, _dirs, files in os.walk(bin_dir):
                    if exe in files:
                        engine_path = os.path.join(root, exe)
                        break

            if not engine_path:
                self.hpc_log.emit(
                    f"ERROR: extracted archive did not contain '{exe}'. "
                    "The release asset may be for a different platform.")
                return

            # Best-effort execute bit (ZIPs drop it on macOS/Linux).
            with contextlib.suppress(OSError):
                os.chmod(engine_path, os.stat(engine_path).st_mode | stat.S_IEXEC)

            # macOS Gatekeeper quarantines downloaded binaries; strip the flag so
            # the engine can launch. Non-fatal if xattr is missing or fails.
            if sys.platform == 'darwin':
                try:
                    subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                        ["xattr", "-dr", "com.apple.quarantine", bin_dir],
                        check=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    self.hpc_log.emit(
                        f"Note: could not clear macOS quarantine flag "
                        f"(non-fatal): {e}")

            # If the binary landed somewhere resolve_engine() would not look
            # (nested zip), persist its exact path so RUN still finds it.
            expected = os.path.join(bin_dir, exe)
            if os.path.abspath(engine_path) != os.path.abspath(expected):
                with contextlib.suppress(Exception):
                    QSettings().setValue(ENGINE_SETTINGS_KEY, engine_path)
                self.hpc_log.emit(f"Engine installed to {engine_path}")
            else:
                self.hpc_log.emit(f"Engine installed to bin/{plat}/")
            success = True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                self.hpc_log.emit(
                    "ERROR: no prebuilt engine release asset for your platform "
                    "yet (HTTP 404). Build from source or use Browse instead.")
            else:
                self.hpc_log.emit(
                    f"ERROR: download failed with HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            self.hpc_log.emit(f"ERROR: network error during download: {e.reason}")
        except Exception as e:
            self.hpc_log.emit(f"ERROR: engine download/install failed: {e}")
        finally:
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            self.engine_download_finished.emit(success)

    def on_engine_download_finished(self, success):
        """GUI-thread slot: re-enable the Download button and refresh the engine row."""
        try:
            self._refresh_engine_display()
            if success:
                self.hpc_log.emit("Engine ready - the RUN button can now use it.")
        finally:
            self.btn_engine_download.setEnabled(True)


    # --------------------------------------------------------------- SAM pick
    def _update_sam_angle_label(self, value):
        """Live degree equivalent for the Max-angle (radians) spin box."""
        try:
            self.lbl_sam_degrees.setText(f"≈ {math.degrees(value):.1f}°")
        except Exception:
            self.lbl_sam_degrees.setText("")

    def _activate_sam_picker(self):
        """Arm a QgsMapToolEmitPoint so the next canvas click captures a spectrum."""
        layer = self.cb_input.currentLayer()
        if not layer:
            QMessageBox.warning(self, "Warning",
                                "Select an input raster layer before picking a reference pixel.")
            return
        try:
            from qgis.gui import QgsMapToolEmitPoint
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Map tool unavailable: {e}")
            return
        self._sam_pick_tool = QgsMapToolEmitPoint(self.iface.mapCanvas())
        self._sam_pick_tool.canvasClicked.connect(self._on_sam_canvas_clicked)
        self.iface.mapCanvas().setMapTool(self._sam_pick_tool)
        self.hpc_log.emit("Pick reference pixel: click a point on the map canvas...")

    def _on_sam_canvas_clicked(self, point, button):
        """Read every band value at the clicked pixel into the SAM reference spectrum."""
        try:
            layer = self.cb_input.currentLayer()
            if not layer:
                return
            from qgis.core import (QgsRaster, QgsProject, QgsPointXY,
                                   QgsCoordinateTransform)
            # Canvas CRS -> layer CRS so the identify hits the right pixel.
            pt = QgsPointXY(point)
            try:
                canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
                layer_crs = layer.crs()
                if canvas_crs != layer_crs:
                    xform = QgsCoordinateTransform(canvas_crs, layer_crs,
                                                   QgsProject.instance())
                    pt = xform.transform(pt)
            except Exception:
                pt = QgsPointXY(point)

            provider = layer.dataProvider()
            res = provider.identify(pt, QgsRaster.IdentifyFormat.IdentifyFormatValue)
            if not res or not res.isValid():
                self.hpc_log.emit("Could not read pixel values at that location (outside the raster?).")
                return
            results = res.results()  # {band_number: value}, 1-based band keys
            # Per-band nodata comes back as None/NaN. The engine treats nodata as
            # 0.0 (NoDataHandler) and SAM only ignores a pixel when ALL bands are
            # nodata. So map None/NaN -> 0.0 and reject only a fully-nodata pixel
            # (the scene's black border) — NOT a valid pixel that happens to have
            # a few bad/nodata bands, which would otherwise reject most real pixels.
            raw = [results[b] for b in sorted(results.keys())]
            spectrum = [0.0 if (v is None or v != v) else float(v) for v in raw]
            if not spectrum or all(v == 0.0 for v in spectrum):
                self.hpc_log.emit("That pixel is all-nodata (likely the black border) — "
                                  "click on the actual image.")
                return

            # Best-effort row/col for the confirmation message only (the spectrum
            # itself is sent to the engine via --sam-spectrum).
            try:
                extent = layer.extent()
                col = int((pt.x() - extent.xMinimum()) / layer.rasterUnitsPerPixelX())
                row = int((extent.yMaximum() - pt.y()) / layer.rasterUnitsPerPixelY())
            except Exception:
                row, col = -1, -1

            self._sam_spectrum = [float(v) for v in spectrum]
            self._sam_rc = (row, col)
            self.lbl_sam_ref.setText(f"Reference: {len(self._sam_spectrum)} bands @ row {row}, col {col}")
            self.hpc_log.emit(
                f"Reference captured: {len(self._sam_spectrum)} bands at row {row}, col {col}.")
        except Exception as e:
            self.hpc_log.emit(f"Reference capture failed: {e}")
        finally:
            with contextlib.suppress(Exception):
                if self._sam_pick_tool is not None:
                    self.iface.mapCanvas().unsetMapTool(self._sam_pick_tool)
    # ------------------------------------------------------------------ run
    def run_hpc(self):
        layer = self.cb_input.currentLayer()
        if not layer:
            QMessageBox.warning(self, "Warning", "Please select an input raster layer.")
            return

        input_path = layer.dataProvider().dataSourceUri()
        if "|" in input_path:
            input_path = input_path.split("|")[0]

        self.txt_log.clear()
        self.btn_hpc.setEnabled(False)

        # Output folder: use what the user typed, or fall back to a private
        # subfolder of QGIS's own temporary scratch space. QgsProcessingUtils
        # .tempFolder() honours the user's QGIS temp setting and is managed by
        # QGIS; mkdtemp gives each run its own subfolder so the engine's
        # clear-prior-outputs step can never touch unrelated files. Nothing is
        # deleted while this run's layers are open — QGIS reclaims its temp space
        # on its own terms.
        out_dir = self.le_output.text().strip()
        if not out_dir:
            try:
                from qgis.core import QgsProcessingUtils
                base = QgsProcessingUtils.tempFolder()
            except Exception:
                base = tempfile.gettempdir()
            out_dir = tempfile.mkdtemp(prefix="hyperspatial_", dir=base)
            self.le_output.setText(out_dir)
            self.hpc_log.emit(f"No output folder set - using QGIS temporary scratch: {out_dir}")

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

        # Capture ALL widget state on the GUI thread BEFORE starting the worker.
        operation = self.cb_operation.currentData()
        self._last_operation = operation
        self._sam_threshold = None

        cmd = [
            cli,
            "--input", input_path,
            "--output", self.le_output.text(),
            "--threads", str(self.spin_threads.value()),
            "--mode", self.cb_backend.currentData(),
            "--operation", operation,
        ]

        if operation == "pca":
            cmd += ["--pca-components", str(self.spin_pca_components.value())]
            if self.chk_pca_correlation.isChecked():
                cmd += ["--pca-matrix", "correlation"]

        elif operation == "sam":
            if not self._sam_spectrum:
                QMessageBox.warning(
                    self, "No reference spectrum",
                    "Pick a reference pixel on the map before running SAM target detection.")
                self.btn_hpc.setEnabled(True)
                return
            cmd += ["--sam-spectrum", ",".join(repr(v) for v in self._sam_spectrum)]
            if self.chk_sam_mask.isChecked():
                self._sam_threshold = float(self.spin_sam_angle.value())
                cmd += ["--sam-threshold", str(self._sam_threshold)]
            if self.chk_sam_similarity.isChecked():
                cmd += ["--sam-emit-similarity"]

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
                         args=(cmd, self.le_output.text(), is_vrt_mode, selected, operation),
                         daemon=True).start()

    def _hpc_task(self, cmd, out_dir, is_vrt_mode, selected, operation):
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
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # nosec B603 - cmd is the resolved engine path + fixed flags, no shell
                                    text=True, env=engine_env(), **popen_kw)
            for line in iter(proc.stdout.readline, ''):
                self.hpc_log.emit(line.strip())
            proc.stdout.close()
            proc.wait()

            # Post-processing merge & cleanup (background thread). Only spectral
            # indices are written as per-tile chunks that need VRT merging; PCA
            # and SAM write their final GeoTIFFs (pca.tif / sam_angle.tif ...)
            # directly, so they skip this block entirely.
            tiles = sorted(glob.glob(os.path.join(out_dir, "tile_*.tif")))
            if operation == "indices" and tiles:
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
                        with contextlib.suppress(OSError):
                            os.remove(tile)
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
            if not self.chk_canvas.isChecked():
                return

            operation = self._last_operation
            if operation == "pca":
                self._load_pca_outputs(out_dir)
            elif operation == "sam":
                self._load_sam_outputs(out_dir)
            else:
                self._load_index_outputs(out_dir)
        finally:
            self.btn_hpc.setEnabled(True)

    def _load_index_outputs(self, out_dir):
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
                if renderer and hasattr(renderer, "setContrastEnhancement"):
                    enhancement = QgsContrastEnhancement(layer.dataProvider().dataType(1))
                    enhancement.setContrastEnhancementAlgorithm(QgsContrastEnhancement.ContrastEnhancementAlgorithm.StretchToMinimumMaximum)
                    # Indices like NDVI typically range from -1 to 1.
                    enhancement.setMinimumValue(-1.0)
                    enhancement.setMaximumValue(1.0)
                    renderer.setContrastEnhancement(enhancement)

                layer.triggerRepaint()
                self.iface.mapCanvas().refresh()
                self.hpc_log.emit(f"Loaded {name} to map canvas.")

    def _load_pca_outputs(self, out_dir):
        """Load pca.tif as a 3-band false-colour composite (PC1/PC2/PC3 -> R/G/B)."""
        eigen = os.path.join(out_dir, "pca_eigen.csv")
        if os.path.exists(eigen):
            self.hpc_log.emit(f"Eigenvalues/vectors written to {eigen}")

        path = os.path.join(out_dir, "pca.tif")
        if not os.path.exists(path):
            self.hpc_log.emit("PCA finished but pca.tif was not found in the output folder.")
            return
        layer = self.iface.addRasterLayer(path, "PCA Composite")
        if not (layer and layer.isValid()):
            self.hpc_log.emit("PCA output pca.tif could not be loaded as a raster layer.")
            return
        try:
            from qgis.core import QgsMultiBandColorRenderer, QgsContrastEnhancement
            provider = layer.dataProvider()
            nbands = provider.bandCount()
            r = 1
            g = 2 if nbands >= 2 else 1
            b = 3 if nbands >= 3 else g
            renderer = QgsMultiBandColorRenderer(provider, r, g, b)
            for band, setter in ((r, renderer.setRedContrastEnhancement),
                                 (g, renderer.setGreenContrastEnhancement),
                                 (b, renderer.setBlueContrastEnhancement)):
                try:
                    ce = QgsContrastEnhancement(provider.dataType(band))
                    ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.ContrastEnhancementAlgorithm.StretchToMinimumMaximum)
                    mn, mx = self._band_min_max(provider, band)
                    ce.setMinimumValue(mn)
                    ce.setMaximumValue(mx)
                    setter(ce)
                except Exception as e:
                    self.hpc_log.emit(f"PCA band {band} stretch skipped: {e}")
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            self.iface.mapCanvas().refresh()
            self.hpc_log.emit(f"Loaded PCA false-colour composite (PC{r}/PC{g}/PC{b} -> R/G/B).")
        except Exception as e:
            self.hpc_log.emit(f"PCA renderer skipped (layer still loaded): {e}")

    def _load_sam_outputs(self, out_dir):
        """Load sam_angle.tif (inverted ramp) plus optional mask / similarity bands."""
        path = os.path.join(out_dir, "sam_angle.tif")
        if not os.path.exists(path):
            self.hpc_log.emit("SAM finished but sam_angle.tif was not found in the output folder.")
        else:
            layer = self.iface.addRasterLayer(path, "SAM Angle (rad)")
            if layer and layer.isValid():
                try:
                    self._apply_sam_angle_renderer(layer)
                    layer.triggerRepaint()
                    self.iface.mapCanvas().refresh()
                    self.hpc_log.emit("Loaded SAM angle layer (small angle = strong match).")
                except Exception as e:
                    self.hpc_log.emit(f"SAM angle renderer skipped (layer still loaded): {e}")

        # Detection mask (only present when a threshold was requested).
        mask_path = os.path.join(out_dir, "sam_mask.tif")
        if os.path.exists(mask_path):
            mlayer = self.iface.addRasterLayer(mask_path, "SAM Detection Mask")
            if mlayer and mlayer.isValid():
                try:
                    from qgis.core import QgsPalettedRasterRenderer
                    from qgis.PyQt.QtGui import QColor
                    classes = [
                        QgsPalettedRasterRenderer.Class(0, QColor(0, 0, 0, 0), "No match"),
                        QgsPalettedRasterRenderer.Class(1, QColor(255, 0, 0), "Detected"),
                    ]
                    mlayer.setRenderer(QgsPalettedRasterRenderer(mlayer.dataProvider(), 1, classes))
                    mlayer.triggerRepaint()
                    self.hpc_log.emit("Loaded SAM detection mask (sam_mask.tif).")
                except Exception as e:
                    self.hpc_log.emit(f"SAM mask renderer skipped (layer still loaded): {e}")

        # Optional per-pixel cosine similarity band.
        sim_path = os.path.join(out_dir, "sam_similarity.tif")
        if os.path.exists(sim_path):
            slayer = self.iface.addRasterLayer(sim_path, "SAM Similarity")
            if slayer and slayer.isValid():
                self.hpc_log.emit("Loaded SAM similarity band (sam_similarity.tif).")

    def _band_min_max(self, provider, band):
        """~2nd/98th percentile stretch bounds, falling back to full band stats."""
        with contextlib.suppress(Exception):
            cut = provider.cumulativeCut(band, 0.02, 0.98)
            mn, mx = cut[0], cut[1]
            if mn is not None and mx is not None and mx > mn:
                return mn, mx
        stats = provider.bandStatistics(band)
        return stats.minimumValue, stats.maximumValue

    def _apply_sam_angle_renderer(self, layer):
        """Singleband pseudocolor with an INVERTED ramp: small angle = hot match."""
        from qgis.core import (QgsSingleBandPseudoColorRenderer, QgsRasterShader,
                               QgsColorRampShader)
        from qgis.PyQt.QtGui import QColor
        provider = layer.dataProvider()
        stats = provider.bandStatistics(1)
        # nodata is -1.0; clamp the visible minimum to a real angle (>= 0).
        mn = max(0.0, stats.minimumValue if stats.minimumValue is not None else 0.0)
        data_max = stats.maximumValue if stats.maximumValue is not None else 1.0
        # Clamp the ramp max to the chosen threshold if the user set one.
        mx = self._sam_threshold if self._sam_threshold else data_max
        if not (mx and mx > mn):
            mx = data_max if data_max and data_max > mn else mn + 1.0
        mid = mn + (mx - mn) * 0.5

        shader = QgsRasterShader()
        ramp = QgsColorRampShader()
        ramp.setColorRampType(QgsColorRampShader.Type.Interpolated)
        # Inverted: strongest match (smallest angle) is hot, large angle is cool.
        items = [
            QgsColorRampShader.ColorRampItem(mn, QColor(215, 25, 28), f"{mn:.3f} (strong)"),
            QgsColorRampShader.ColorRampItem(mid, QColor(255, 255, 191), f"{mid:.3f}"),
            QgsColorRampShader.ColorRampItem(mx, QColor(44, 123, 182), f"{mx:.3f} (weak)"),
        ]
        ramp.setColorRampItemList(items)
        shader.setRasterShaderFunction(ramp)
        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        if hasattr(renderer, "setClassificationMin"):
            renderer.setClassificationMin(mn)
            renderer.setClassificationMax(mx)
        layer.setRenderer(renderer)
