# -*- coding: utf-8 -*-
"""QGIS plugin entry point for standalone hyperspectral index processing.

Single-purpose plugin extracted from the HyperSpatial Big Data monolith's
"HPC Preprocessing" tab. It adds one toolbar icon + one menu action that opens a
modeless dialog; all heavy work (the external C++ engine) runs off the UI thread
inside the dialog (see dialog.py).
"""
import os

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .dialog import HyperspectralProcessingDialog


class HyperspectralProcessingPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)

        # QSettings may return None when the locale is unset; guard it so the
        # plugin still loads (None[0:2] would raise and abort initialisation).
        locale = str(QSettings().value('locale/userLocale') or 'en')[0:2]
        locale_path = os.path.join(
            self.plugin_dir, 'i18n', f'HyperspectralProcessing_{locale}.qm')
        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        self.actions = []
        self.menu = 'HyperSpectral Processer'
        self.first_start = None
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
            text='HyperSpectral Processer',
            callback=self.run,
            parent=self.iface.mainWindow()
        )
        self.first_start = True

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        self.actions = []
        # Release the dialog too, so reloading/uninstalling the plugin leaves
        # no orphaned window or SAM map tool behind.
        if self.dialog is not None:
            try:
                tool = getattr(self.dialog, "_sam_pick_tool", None)
                if tool is not None and self.iface.mapCanvas().mapTool() is tool:
                    self.iface.mapCanvas().unsetMapTool(tool)
                self.dialog.close()
                self.dialog.deleteLater()
            except Exception as e:  # never let unload fail, but leave a trace
                from qgis.core import QgsMessageLog, Qgis
                QgsMessageLog.logMessage(f"Dialog cleanup on unload failed: {e}",
                                         "HyperSpectral Processer", Qgis.MessageLevel.Warning)
            self.dialog = None
        self.first_start = True

    def run(self):
        # Lazily build the dialog on first use.
        if self.first_start:
            self.first_start = False
            self.dialog = HyperspectralProcessingDialog(self.iface)

        # Show modeless so the user can keep working in QGIS (pan/zoom, inspect
        # the layers the plugin adds) while the C++ engine runs in a background
        # thread. exec() would spin a modal loop that blocks QGIS's main window,
        # defeating the "NEVER BLOCK THE UI" design.
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
