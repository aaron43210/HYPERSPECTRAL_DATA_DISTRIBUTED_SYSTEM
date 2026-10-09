# -*- coding: utf-8 -*-
"""
EnMAP STAC downloader dialog.

Single-purpose QDialog extracted from the "STAC Downloader" tab of the
HyperSpatial Big Data monolith. It searches the DLR EnMAP L2A STAC catalogue
and downloads a selected scene.

All network work (search + download) runs in a background threading.Thread and
marshals its results back to the Qt GUI thread via pyqtSignal, so the QGIS UI
thread is never blocked.
"""
import os
import threading

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QProgressBar, QMessageBox, QHeaderView,
)
from qgis.PyQt.QtCore import pyqtSignal

from .services.enmap_client import search_scenes, download_scene


class EnmapDownloaderDialog(QDialog):

    download_progress = pyqtSignal(int, int)
    download_finished = pyqtSignal(str)
    download_error = pyqtSignal(str)

    def __init__(self, iface, parent=None):
        super(EnmapDownloaderDialog, self).__init__(parent)
        self.iface = iface
        self.setWindowTitle('HyperSpatial EnMAP Downloader')
        self.resize(850, 500)

        self.init_downloader_ui()

        # Connect worker-thread signals to GUI-thread slots.
        self.download_progress.connect(self.update_progress)
        self.download_finished.connect(self.on_download_finished)
        self.download_error.connect(self.on_download_error)

    # =========================================================================
    # STAC DOWNLOADER UI
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

        self.setLayout(layout)

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
