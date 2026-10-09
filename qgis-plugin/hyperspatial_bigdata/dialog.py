# -*- coding: utf-8 -*-
# Number of spectral index bands the C++ engine appends to every tile GeoTIFF,
# in fixed order: NDVI, NDWI, NDBI, SAVI, EVI, NDMI, MSAVI.
_N_INDEX_BANDS = 7
import os
import glob
import subprocess
import threading
import multiprocessing
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QProgressBar, QMessageBox, QHeaderView,
    QTabWidget, QWidget, QCheckBox, QComboBox, QTextEdit, QFileDialog,
    QRadioButton, QSpinBox, QDoubleSpinBox
)
from qgis.PyQt.QtCore import pyqtSignal, Qt
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from osgeo import gdal

from .services.enmap_client import search_scenes, download_scene

class HyperSpatialDialog(QDialog):
    
    download_progress = pyqtSignal(int, int)
    download_finished = pyqtSignal(str)
    download_error = pyqtSignal(str)
    
    hpc_log = pyqtSignal(str)
    hpc_finished = pyqtSignal(str)

    spark_log = pyqtSignal(str)
    spark_finished = pyqtSignal()

    def __init__(self, iface, parent=None):
        super(HyperSpatialDialog, self).__init__(parent)
        self.iface = iface
        self.setWindowTitle('HyperSpatial Big Data')
        self.resize(850, 600)
        
        self.tabs = QTabWidget()
        self.tab_downloader = QWidget()
        self.tab_hpc = QWidget()
        self.tab_analytics = QWidget()
        
        self.tabs.addTab(self.tab_downloader, "STAC Downloader")
        self.tabs.addTab(self.tab_hpc, "HPC Preprocessing")
        self.tabs.addTab(self.tab_analytics, "Sedona Analytics")
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        
        self.init_downloader_ui()
        self.init_hpc_ui()
        self.init_analytics_ui()
        
        # Connect signals
        self.download_progress.connect(self.update_progress)
        self.download_finished.connect(self.on_download_finished)
        self.download_error.connect(self.on_download_error)
        
        self.hpc_log.connect(self.append_hpc_log)
        self.hpc_finished.connect(self.on_hpc_finished)
        self.spark_log.connect(self.append_spark_log)
        self.spark_finished.connect(self.on_spark_finished)

    def append_spark_log(self, msg):
        self.txt_spark_log.append(msg)

    def on_spark_finished(self):
        # Re-enable the button from the GUI thread (safe: driven by a signal).
        self.btn_spark.setEnabled(True)

    # =========================================================================
    # STAC DOWNLOADER TAB
    # =========================================================================
    def init_downloader_ui(self):
        layout = QVBoxLayout()

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("BBox (W,S,E,N):"))
        self.le_bbox = QLineEdit("10.0,47.0,11.0,48.0")
        h_layout.addWidget(self.le_bbox)
        
        h_layout.addWidget(QLabel("Start:"))
        self.le_start = QLineEdit("2022-01-01")
        h_layout.addWidget(self.le_start)
        
        h_layout.addWidget(QLabel("End:"))
        self.le_end = QLineEdit("2023-01-01")
        h_layout.addWidget(self.le_end)

        layout.addLayout(h_layout)

        t_layout = QHBoxLayout()
        t_layout.addWidget(QLabel("DLR EOWEB Token (Optional):"))
        self.le_token = QLineEdit()
        self.le_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.le_token.setPlaceholderText("Bearer token for restricted assets")
        t_layout.addWidget(self.le_token)

        self.btn_search = QPushButton("Search EnMAP")
        self.btn_search.clicked.connect(self.search)
        t_layout.addWidget(self.btn_search)
        layout.addLayout(t_layout)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Scene ID', 'Date', 'Cloud Cover', 'URL'])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        d_layout = QHBoxLayout()
        self.btn_download = QPushButton("Download Selected")
        self.btn_download.clicked.connect(self.start_download)
        self.btn_download.setEnabled(False)
        self.table.itemSelectionChanged.connect(lambda: self.btn_download.setEnabled(True))
        d_layout.addWidget(self.btn_download)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        d_layout.addWidget(self.progress_bar)
        layout.addLayout(d_layout)

        self.tab_downloader.setLayout(layout)

    def search(self):
        self.table.setRowCount(0)
        try:
            # Parse the bbox INSIDE the try: a malformed "W,S,E,N" string would
            # otherwise raise uncaught out of this slot.
            parts = [p.strip() for p in self.le_bbox.text().split(',')]
            if len(parts) != 4:
                raise ValueError("BBox must be four comma-separated numbers: W,S,E,N")
            bbox = [float(x) for x in parts]
            token = self.le_token.text().strip() or None
            results = search_scenes(bbox, self.le_start.text(), self.le_end.text(), limit=50, token=token)
            if not results:
                QMessageBox.information(self, "No Results", "No EnMAP scenes matched your search.")
                return
            self.table.setRowCount(len(results))
            for i, res in enumerate(results):
                self.table.setItem(i, 0, QTableWidgetItem(res['id']))
                self.table.setItem(i, 1, QTableWidgetItem(res['date']))
                self.table.setItem(i, 2, QTableWidgetItem(str(res['cloud_cover'])))
                self.table.setItem(i, 3, QTableWidgetItem(res['download_url']))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed: {str(e)}")

    def start_download(self):
        row = self.table.currentRow()
        if row < 0: return
        scene_id = self.table.item(row, 0).text()
        url = self.table.item(row, 3).text()
        # Universal save location: default to the user's Downloads folder
        base_dir = os.path.join(os.path.expanduser("~"), "Downloads", "hyperspatial_data")
        os.makedirs(base_dir, exist_ok=True)
        self.btn_download.setEnabled(False)
        self.progress_bar.setValue(0)
        token = self.le_token.text().strip() or None
        threading.Thread(target=self._download_task, args=(scene_id, url, base_dir, token), daemon=True).start()

    def _download_task(self, scene_id, url, output_dir, token):
        try:
            def cb(downloaded, total):
                if total > 0: self.download_progress.emit(int((downloaded / total) * 100), total)
            path = download_scene(scene_id, url, output_dir, token=token, progress_callback=cb)
            self.download_finished.emit(path)
        except Exception as e:
            self.download_error.emit(str(e))

    def update_progress(self, pct, total): self.progress_bar.setValue(pct)
    def on_download_finished(self, path):
        QMessageBox.information(self, "Success", f"Downloaded to:\n{path}")
        self.btn_download.setEnabled(True)
        self.progress_bar.setValue(100)
    def on_download_error(self, err):
        QMessageBox.critical(self, "Error", err)
        self.btn_download.setEnabled(True)

    # =========================================================================
    # HPC PREPROCESSING TAB
    # =========================================================================
    def init_hpc_ui(self):
        layout = QVBoxLayout()
        
        from qgis.gui import QgsMapLayerComboBox
        from qgis.core import QgsMapLayerProxyModel
        
        # Input Layer Dropdown
        h1 = QHBoxLayout()
        self.cb_input = QgsMapLayerComboBox()
        self.cb_input.setFilters(QgsMapLayerProxyModel.RasterLayer)
        h1.addWidget(QLabel("Input Layer:"))
        h1.addWidget(self.cb_input)
        layout.addLayout(h1)
        
        h2 = QHBoxLayout()
        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output Directory for Tiles and VRT")
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(lambda: self.le_output.setText(QFileDialog.getExistingDirectory(self, "Select Output Folder")))
        h2.addWidget(QLabel("Output:"))
        h2.addWidget(self.le_output)
        h2.addWidget(btn_out)
        layout.addLayout(h2)

        # Options
        opt = QHBoxLayout()
        self.chk_ndvi = QCheckBox("NDVI")
        self.chk_ndvi.setChecked(True)
        self.chk_ndwi = QCheckBox("NDWI")
        self.chk_ndwi.setChecked(True)
        self.chk_ndbi = QCheckBox("NDBI")
        self.chk_ndbi.setChecked(True)
        self.chk_savi = QCheckBox("SAVI")
        self.chk_savi.setChecked(True)
        self.chk_evi = QCheckBox("EVI")
        self.chk_evi.setChecked(True)
        self.chk_ndmi = QCheckBox("NDMI")
        self.chk_ndmi.setChecked(True)
        self.chk_msavi = QCheckBox("MSAVI")
        self.chk_msavi.setChecked(True)
        
        opt.addWidget(self.chk_ndvi)
        opt.addWidget(self.chk_ndwi)
        opt.addWidget(self.chk_ndbi)
        opt.addWidget(self.chk_savi)
        opt.addWidget(self.chk_evi)
        opt.addWidget(self.chk_ndmi)
        opt.addWidget(self.chk_msavi)
        layout.addLayout(opt)
        
        # Output Format
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
        
        # Thread/Core Selection
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("CPU Cores (0 = Auto-detect max):"))
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(0, multiprocessing.cpu_count())
        self.spin_threads.setValue(0)
        h3.addWidget(self.spin_threads)

        # Compute backend. The C++ engine's --mode selects the index backend:
        #   openmp = multi-core CPU (default); gpu = cross-vendor OpenCL
        #   (Apple / NVIDIA / Intel / AMD). GPU transparently falls back to CPU
        #   if no OpenCL GPU is available, so it's always safe to pick.
        h3.addSpacing(16)
        h3.addWidget(QLabel("Backend:"))
        self.cb_backend = QComboBox()
        self.cb_backend.addItem("CPU (OpenMP)", "openmp")
        self.cb_backend.addItem("GPU (OpenCL: Apple/NVIDIA/Intel/AMD)", "gpu")
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
        
        self.tab_hpc.setLayout(layout)

    def run_hpc(self):
        layer = self.cb_input.currentLayer()
        if not layer or not self.le_output.text():
            QMessageBox.warning(self, "Warning", "Please select an input layer and specify an output directory.")
            return
            
        input_path = layer.dataProvider().dataSourceUri()
        if "|" in input_path: input_path = input_path.split("|")[0]
            
        self.txt_log.clear()
        self.btn_hpc.setEnabled(False)
        
        # Locate C++ binary bundled within the plugin
        cli = os.path.join(os.path.dirname(__file__), 'bin', 'hyperspatial_cli')
        
        if not os.path.exists(cli):
            self.hpc_log.emit("ERROR: Could not find bundled hyperspatial_cli binary.")
            self.btn_hpc.setEnabled(True)
            return
            
        # Fix for macOS/Linux [Errno 13] Permission denied after extracting from ZIP.
        # Best-effort: a read-only install location shouldn't abort the run.
        import stat
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
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(proc.stdout.readline, ''):
                self.hpc_log.emit(line.strip())
            proc.stdout.close()
            proc.wait()

            # Post-Processing Merge & Cleanup (Background Thread)
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
                # depend on which checkboxes are ticked — otherwise unchecking an
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
            # the results — even if the C++ step or the GDAL merge raised.
            self.hpc_finished.emit(out_dir)

    def append_hpc_log(self, msg):
        self.txt_log.append(msg)
        
    def on_hpc_finished(self, out_dir):
        self.hpc_log.emit("\n--- HPC COMPLETE ---")
        if self.chk_canvas.isChecked():
            self.hpc_log.emit("Loading merged TIFFs into canvas...")
            
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
                # BUG FIX: Use iface.addRasterLayer to guarantee the layer is added to the 
                # layer tree (legend) and canvas, rather than just the background project registry.
                layer = self.iface.addRasterLayer(path, name)
                if layer and layer.isValid():
                    from qgis.core import QgsContrastEnhancement
                    renderer = layer.renderer()
                    if renderer:
                        enhancement = QgsContrastEnhancement(layer.dataProvider().dataType(1))
                        enhancement.setContrastEnhancementAlgorithm(QgsContrastEnhancement.StretchToMinimumMaximum)
                        # Indices like NDVI typically range from -1 to 1, or 0 to 1 if normalized
                        enhancement.setMinimumValue(-1.0)
                        enhancement.setMaximumValue(1.0)
                        renderer.setContrastEnhancement(enhancement)
                    
                    layer.triggerRepaint()
                    self.iface.mapCanvas().refresh()
                    self.hpc_log.emit(f"Loaded {name} to map canvas.")
                    
        # Seamless UX: point the Sedona Analytics tab at this run's tiles folder.
        # Analytics reads the per-tile GeoTIFFs directly, so it needs the folder
        # that holds tile_r*.tif. Those exist in VRT mode; merged-TIFF mode
        # deletes them, so warn in that case.
        if self.le_output.text():
            out_dir = self.le_output.text()
            self.le_rasters.setText(out_dir)
            import glob as _glob
            if _glob.glob(os.path.join(out_dir, "tile_r*.tif")):
                self.hpc_log.emit(f"Analytics tiles directory set to: {out_dir}")
            else:
                self.hpc_log.emit("Note: no tile_r*.tif remain (merged-TIFF mode). "
                                  "Re-run in Virtual Raster (.vrt) mode to use Sedona analytics.")

        self.btn_hpc.setEnabled(True)

    # =========================================================================
    # SEDONA ANALYTICS TAB
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

        self.btn_spark = QPushButton("RUN SPARK JOB")
        self.btn_spark.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_spark.clicked.connect(self.run_analytics)
        layout.addWidget(self.btn_spark)

        self.txt_spark_log = QTextEdit()
        self.txt_spark_log.setReadOnly(True)
        layout.addWidget(self.txt_spark_log)

        self.tab_analytics.setLayout(layout)

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
        repo_root = os.path.dirname(__file__)
        jobs_dir = os.path.join(repo_root, 'spark-sedona', 'jobs')
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

        # Resolve spark-submit portably. When QGIS is launched from a terminal
        # with the conda/Homebrew env active it is already on PATH; when launched
        # from the GUI it is not, so fall back to the user's login shell (which
        # sources their profile and PATH). zsh is no longer hard-coded.
        import shutil, shlex
        spark_submit = shutil.which("spark-submit")
        if spark_submit:
            cmd = [spark_submit] + spark_args[1:]
        else:
            cmd_str = " ".join(shlex.quote(a) for a in spark_args)
            shell = os.environ.get("SHELL") or "/bin/bash"
            cmd = [shell, "-l", "-c", cmd_str]

        threading.Thread(target=self._spark_task, args=(cmd, out_dir), daemon=True).start()

    def _spark_task(self, cmd, out_dir):
        try:
            self.spark_log.emit(f"Executing: {' '.join(cmd)}\n")
            # CRITICAL FIX: Unset HADOOP_CONF_DIR to prevent macOS crashes
            env = os.environ.copy()
            env["HADOOP_CONF_DIR"] = ""

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
