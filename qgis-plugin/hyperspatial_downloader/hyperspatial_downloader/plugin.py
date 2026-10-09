# -*- coding: utf-8 -*-
"""
HyperSpatial EnMAP Downloader - QGIS plugin hook.

A single-purpose plugin that adds one toolbar icon + one Web-menu action which
opens the EnMAP STAC downloader dialog. The dialog is constructed lazily on
first use and shown non-modally so the QGIS UI is never blocked.
"""
import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction


class HyperspatialDownloaderPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.menu = 'HyperSpatial EnMAP Downloader'
        self.actions = []
        self.dialog = None

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, 'icons', 'icon.png')
        action = QAction(QIcon(icon_path),
                         'HyperSpatial EnMAP Downloader',
                         self.iface.mainWindow())
        action.triggered.connect(self.run)
        action.setStatusTip('Search and download EnMAP L2A scenes from the DLR STAC catalogue')

        # category=Web -> live under the QGIS "Web" menu.
        self.iface.addToolBarIcon(action)
        self.iface.addPluginToWebMenu(self.menu, action)
        self.actions.append(action)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginWebMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        self.actions = []

    def run(self):
        # Lazily construct the dialog on first use, then reuse it.
        if self.dialog is None:
            from .dialog import EnmapDownloaderDialog
            self.dialog = EnmapDownloaderDialog(self.iface, self.iface.mainWindow())

        # Show modeless so the user can keep working in QGIS while the download
        # runs in a background thread. exec() would block the main window.
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
