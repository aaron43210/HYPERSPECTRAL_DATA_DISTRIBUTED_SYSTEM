# -*- coding: utf-8 -*-
import os
from qgis.core import QgsApplication
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .dialog import HyperSpatialDialog

class HyperSpatialBigData:

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        
        # QSettings may return None when the locale is unset; guard it so the
        # plugin still loads (None[0:2] would raise and abort initialisation).
        locale = str(QSettings().value('locale/userLocale') or 'en')[0:2]
        locale_path = os.path.join(self.plugin_dir, 'i18n', f'HyperSpatialBigData_{locale}.qm')
        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        self.actions = []
        self.menu = 'HyperSpatial Big Data'
        self.first_start = None
        self.dialog = None
        self.provider = None

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
        # Initialize the Unified UI Tool
        icon_path = os.path.join(self.plugin_dir, 'icons', 'icon.png')
        self.add_action(
            icon_path,
            text='HyperSpatial Big Data Processing',
            callback=self.run,
            parent=self.iface.mainWindow()
        )
        self.first_start = True

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu('HyperSpatial Big Data', action)
            self.iface.removeToolBarIcon(action)


    def run(self):
        if self.first_start:
            self.first_start = False
            self.dialog = HyperSpatialDialog(self.iface)

        # Show modeless so the user can keep working in QGIS (pan/zoom, inspect
        # the layers the plugin adds to the canvas) while downloads, the C++
        # engine and Spark run in background threads. exec() would spin a modal
        # event loop that blocks QGIS's main window — defeating the whole
        # "NEVER BLOCK THE UI" design.
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
