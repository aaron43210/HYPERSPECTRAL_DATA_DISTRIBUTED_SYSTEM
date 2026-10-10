# -*- coding: utf-8 -*-
"""Locate the external C++/OpenMP/OpenCL hyperspectral engine at run time.

This plugin ships BINARY-FREE: the compiled ``hyperspatial_cli`` engine is NOT
inside the published zip. It is resolved at run time, in priority order:

    1. An explicit path saved in QSettings('hyperspatial_processing/engine_path')
       (set via the dialog's Engine "Browse..." button).
    2. ``hyperspatial_cli`` on the system PATH (e.g. a conda/Homebrew install or
       a ``make install`` of hpc-engine).
    3. A bundled copy under bin/<platform>/ (or a flat bin/). These exist only in
       internal/dev builds and are absent from the submitted plugin zip.

``engine_env()`` returns an environment that prepends QGIS's own GDAL library
directory so the external engine links against the exact GDAL QGIS uses.
"""
import os
import sys
import platform
import shutil
import contextlib

from qgis.PyQt.QtCore import QSettings

ENGINE_SETTINGS_KEY = 'hyperspatial_processing/engine_path'


def _platform_exe():
    """Return ``(platform_subdir, executable_name)`` for the host OS."""
    if sys.platform.startswith('win'):
        return 'windows', 'hyperspatial_cli.exe'
    if sys.platform == 'darwin':
        return 'macos', 'hyperspatial_cli'
    return 'linux', 'hyperspatial_cli'


def resolve_engine():
    """Find the external C++ engine binary.

    Returns a 3-tuple ``(path, source, error)``:
      * success -> (absolute_path, human_source_label, None)
      * failure -> (None, None, help_message)
    """
    plat, exe = _platform_exe()

    # (1) Explicit path saved by the user via the Engine "Browse..." button.
    try:
        saved = QSettings().value(ENGINE_SETTINGS_KEY)
    except Exception:
        saved = None
    if saved:
        saved = str(saved)
        if os.path.isfile(saved):
            return saved, 'configured path (QSettings)', None

    # (2) On the system PATH (conda / Homebrew / `make install`).
    found = shutil.which(exe)
    if found:
        return found, 'system PATH', None

    # (3) Bundled next to the plugin (internal/dev builds only; absent in the zip).
    bin_dir = os.path.join(os.path.dirname(__file__), 'bin')
    for candidate in (os.path.join(bin_dir, plat, exe),
                      os.path.join(bin_dir, exe)):
        if os.path.isfile(candidate):
            return candidate, 'bundled binary', None

    error = (  # nosec B608 - plain English user-facing message, not an SQL query
        "Hyperspectral C++ engine not found.\n\n"
        "This plugin ships without the compiled engine. To enable processing, do "
        "ONE of the following:\n"
        f"  1. Click 'Browse...' on the Engine row and select the '{exe}' binary "
        "(the choice is remembered).\n"
        f"  2. Put '{exe}' on your system PATH.\n"
        "  3. Build it from source: see hpc-engine/ (CMake + GDAL + OpenMP/OpenCL) "
        f"and install the resulting '{exe}'.\n\n"
        f"(Host platform: {platform.system()} {platform.machine()}.)"
    )
    return None, None, error


def engine_env():
    """Return an ``os.environ`` copy with QGIS's GDAL dir on the loader path.

    The external engine links GDAL; prepending QGIS's own GDAL directory makes
    the engine resolve the exact library QGIS uses, avoiding version/symbol
    mismatches. Fully defensive: any failure returns the unmodified environment
    rather than raising into the UI.
    """
    try:
        env = os.environ.copy()
        lib_dirs = []

        # GDAL directory derived from the Python osgeo package location.
        with contextlib.suppress(Exception):
            from osgeo import gdal
            lib_dirs.append(os.path.dirname(gdal.__file__))

        # QGIS install prefix (its lib/ usually holds the GDAL dylib/so/dll).
        with contextlib.suppress(Exception):
            from qgis.core import QgsApplication
            prefix = QgsApplication.prefixPath()
            if prefix:
                lib_dirs.append(os.path.join(prefix, 'lib'))
                lib_dirs.append(prefix)

        lib_dirs = [d for d in lib_dirs if d and os.path.isdir(d)]
        if lib_dirs:
            if sys.platform == 'darwin':
                var = 'DYLD_FALLBACK_LIBRARY_PATH'
            elif sys.platform.startswith('win'):
                var = 'PATH'
            else:
                var = 'LD_LIBRARY_PATH'
            existing = env.get(var, '')
            parts = lib_dirs + ([existing] if existing else [])
            env[var] = os.pathsep.join(parts)
        return env
    except Exception:
        return os.environ.copy()
