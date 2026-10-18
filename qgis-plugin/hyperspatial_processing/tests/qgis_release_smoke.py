"""Release smoke test for the HyperSpectral Processer plugin, run inside QGIS's Python.

Installs the packaged ZIP into a temporary plugins folder, loads it through
classFactory exactly like QGIS does, runs initGui, opens the dialog, runs the
three operations (indices, PCA, SAM) end to end with the real engine, checks the
layers that land on the canvas, the GUI-thread responsiveness, and unload().
Also validates metadata.txt against the plugins.qgis.org required fields.

  A=/Applications/QGIS-<ver>.app/Contents
  PROJ_DATA=$A/Resources/qgis/proj QT_QPA_PLATFORM=offscreen \
  PYTHONPATH=$A/Resources/python3.12:$A/Resources/python3.12/lib-dynload:$A/Resources/python3.12/site-packages \
  QGIS_PREFIX_PATH=$A/MacOS PATH=<dir containing hyperspatial_cli>:$PATH \
  $A/MacOS/python3.12 qgis_release_smoke.py --zip ../hyperspatial_processing.zip --cube <SPECTRAL_IMAGE.BSQ>
"""
import argparse, configparser, os, sys, tempfile, time, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--cube", required=True)
ap.add_argument("--timeout", type=float, default=300)
a = ap.parse_args()

from qgis.core import QgsApplication, QgsProject, QgsRasterLayer
from qgis.gui import QgsMapCanvas
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QMainWindow, QToolBar, QMenu

app = QgsApplication([], True)
app.initQgis()
QCoreApplication.setOrganizationName("HyperSpatialReleaseTest")
QCoreApplication.setApplicationName("HyperSpatialReleaseTest")
results = []
check = lambda name, ok, info="": results.append((name, bool(ok), info))

# ---- install from the ZIP ------------------------------------------------------
plugins = tempfile.mkdtemp(prefix="qgis_plugins_")
with zipfile.ZipFile(a.zip) as z:
    roots = {n.split("/")[0] for n in z.namelist()}
    check("ZIP has exactly one root folder 'hyperspatial_processing'", roots == {"hyperspatial_processing"}, str(roots))
    z.extractall(plugins)
sys.path.insert(0, plugins)

# ---- metadata ----------------------------------------------------------------------
cp = configparser.ConfigParser(interpolation=None)
cp.read(os.path.join(plugins, "hyperspatial_processing", "metadata.txt"))
g = cp["general"]
required = ["name", "qgisMinimumVersion", "description", "about", "version", "author", "email", "repository"]
missing = [k for k in required if not g.get(k, "").strip()]
check("metadata: required fields present", not missing, ", ".join(missing) or "all present")
check("metadata: stable (experimental=False, deprecated=False)",
      g.get("experimental") == "False" and g.get("deprecated") == "False",
      f"experimental={g.get('experimental')} deprecated={g.get('deprecated')}")
check("metadata: version has a changelog entry", g.get("version") and g.get("version") in g.get("changelog", ""),
      g.get("version"))
icon = os.path.join(plugins, "hyperspatial_processing", g.get("icon", ""))
check("metadata: icon exists and is small", os.path.isfile(icon) and os.path.getsize(icon) < 100_000,
      f"{os.path.getsize(icon) // 1024} KB" if os.path.isfile(icon) else "missing")


# ---- minimal QGIS interface ---------------------------------------------------------
class Iface:
    def __init__(self):
        self.win = QMainWindow(); self.canvas = QgsMapCanvas(self.win)
        self.toolbar = QToolBar(); self.menus = {}
    def mainWindow(self): return self.win
    def mapCanvas(self): return self.canvas
    def addToolBarIcon(self, action): self.toolbar.addAction(action)
    def removeToolBarIcon(self, action): self.toolbar.removeAction(action)
    def addPluginToMenu(self, name, action): self.menus.setdefault(name, []).append(action)
    def removePluginMenu(self, name, action):
        if action in self.menus.get(name, []): self.menus[name].remove(action)
    def addRasterLayer(self, path, name):
        lyr = QgsRasterLayer(path, name)
        if lyr.isValid(): QgsProject.instance().addMapLayer(lyr)
        return lyr
    def messageBar(self): return None


iface = Iface()
import hyperspatial_processing
plugin = hyperspatial_processing.classFactory(iface)
plugin.initGui()
n_actions = len(iface.toolbar.actions())
check("initGui adds a toolbar action and a menu entry", n_actions >= 1 and any(iface.menus.values()),
      f"{n_actions} toolbar action(s)")
plugin.run()
dlg = plugin.dialog
check("dialog opens", dlg is not None and dlg.isVisible())
check("engine resolved", "not found" not in dlg.le_engine.text(), dlg.le_engine.text()[-60:])

cube = QgsRasterLayer(a.cube, "EnMAP cube"); QgsProject.instance().addMapLayer(cube)
dlg.cb_input.setLayer(cube)


def run_op(op_index, prep=None):
    out = tempfile.mkdtemp(prefix="hs_release_")
    dlg.le_output.setText(out)
    dlg.cb_operation.setCurrentIndex(op_index)
    if prep: prep()
    before = set(QgsProject.instance().mapLayers())
    t0 = time.time(); last = time.time(); worst = 0.0
    dlg.run_hpc()
    time.sleep(0.2)
    while time.time() - t0 < a.timeout:
        app.processEvents(); now = time.time(); worst = max(worst, now - last); last = now
        if dlg.btn_hpc.isEnabled():
            break
        time.sleep(0.05)
    for _ in range(10): app.processEvents()
    new = [l.name() for k, l in QgsProject.instance().mapLayers().items() if k not in before]
    return dlg.txt_log.toPlainText(), new, time.time() - t0, worst


log, new, dt, stall = run_op(0)
check("Indices: engine RESULT line and 7 index layers", "operation=indices" in log and
      sum(n.startswith("Computed") for n in new) == 7, f"{dt:.1f} s, layers: {len(new)}")
check("Indices: GUI never stalled > 1 s", stall < 1.0, f"{stall:.2f} s")

log, new, dt, stall = run_op(1)
check("PCA: engine RESULT line and PCA composite layer", "operation=pca" in log and "PCA Composite" in new,
      f"{dt:.1f} s")

# Reference spectrum = a lake pixel (row 1082, col 738 on the EnMAP test scene),
# nodata bands sent as 0.0 exactly like the plugin's canvas picker does.
from osgeo import gdal
_ds = gdal.Open(a.cube)
_nd = _ds.GetRasterBand(1).GetNoDataValue()
ref = []
for b in range(1, _ds.RasterCount + 1):
    v = float(_ds.GetRasterBand(b).ReadAsArray(738, 1082, 1, 1)[0, 0])
    ref.append(0.0 if v == _nd else v)
_ds = None
def sam_prep():
    dlg._sam_spectrum = ref; dlg._sam_rc = (1082, 738)
    dlg.chk_sam_mask.setChecked(True); dlg.chk_sam_similarity.setChecked(True)
log, new, dt, stall = run_op(2, sam_prep)
check("SAM: angle, mask and similarity layers", "operation=sam" in log and
      {"SAM Angle (rad)", "SAM Detection Mask", "SAM Similarity"} <= set(new), f"{dt:.1f} s, {new}")
check("no ERROR / Traceback in any run log", "Traceback" not in log and "ERROR" not in log)

plugin.unload()
for _ in range(5): app.processEvents()
check("unload removes toolbar action, menu entry and dialog",
      not iface.toolbar.actions() and not any(iface.menus.values()) and plugin.dialog is None)

print("\nResults:")
for name, ok, info in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {info}" if info else ""))
app.exitQgis()
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
