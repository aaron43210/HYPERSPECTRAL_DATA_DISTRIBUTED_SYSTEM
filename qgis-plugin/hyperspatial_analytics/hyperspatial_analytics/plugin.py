# -*- coding: utf-8 -*-
"""QGIS plugin hook for HyperSpatial Sedona Analytics.

Standalone, single-purpose plugin: it adds one toolbar icon + menu action that
opens the Sedona Analytics dialog. All heavy work (spark-submit) runs in a
background thread inside the dialog, so the QGIS UI thread never blocks.
"""
import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .dialog import SedonaAnalyticsDialog


class HyperspatialAnalyticsPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = 'HyperSpatial Sedona Analytics'
        # Constructed lazily on first run() so plugin load stays cheap.
        self.dialog = None

    def add_action(self, icon_path, text, callback, enabled_flag=True,
                   add_to_menu=True, add_to_toolbar=True,
                   status_tip=None, whats_this=None, parent=None):

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)
        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.iface.addToolBarIcon(action)
        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, 'icons', 'icon.png')
        self.add_action(
            icon_path,
            text='HyperSpatial Sedona Analytics',
            callback=self.run,
            parent=self.iface.mainWindow()
        )

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)

    def run(self):
        # Lazily construct the dialog on first use.
        if self.dialog is None:
            self.dialog = SedonaAnalyticsDialog(self.iface)

        # Show modeless so the user can keep working in QGIS while the Spark job
        # runs in a background thread. exec() would spin a modal event loop that
        # blocks QGIS's main window, defeating the "NEVER BLOCK THE UI" design.
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
