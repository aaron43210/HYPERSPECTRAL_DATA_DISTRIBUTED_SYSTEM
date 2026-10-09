#!/usr/bin/env bash

# Packages the unified QGIS plugin (qgis-plugin/hyperspatial_bigdata) into a
# single zip for installation / upload to plugins.qgis.org.
#
# NOTE: the plugin REQUIRES its bundled C++ binary (bin/hyperspatial_cli) and the
# bundled spark-sedona/ jobs + data/ — the plugin runs them directly, so they are
# intentionally INCLUDED. Only caches and local scratch are excluded. The result
# is ~50 KB (the compiled binary is a small, single-arch Mach-O), not a "massive"
# zip. Do not "shrink" it by deleting bin/ or spark-sedona/ — that breaks the HPC
# and Analytics tabs.
#
# The bundled binary is produced by building hpc-engine (see hpc-engine/CMakeLists.txt)
# and copying the resulting hyperspatial_cli into qgis-plugin/hyperspatial_bigdata/bin/.
# Rebuild it on the target architecture; a binary built with -DNATIVE_ARCH=ON is
# NOT portable to other CPUs.
#
# CROSS-PLATFORM: the engine is a native per-OS executable (Windows .exe / Linux
# ELF / macOS Mach-O) and macOS + Linux even share the file name, so a single flat
# bin/ cannot hold all three. For a multi-OS plugin, lay the binaries out per OS:
#     bin/windows/hyperspatial_cli.exe
#     bin/linux/hyperspatial_cli
#     bin/macos/hyperspatial_cli
# The plugin (dialog.py::_resolve_cli) picks the one matching the host, and still
# accepts a flat bin/hyperspatial_cli[.exe] for a single-OS package.

PLUGIN_DIR="qgis-plugin/hyperspatial_bigdata"
ZIP_NAME="hyperspatial_bigdata.zip"

echo "Packaging QGIS Plugin..."

# Remove old zip if exists
rm -f $ZIP_NAME

# Create zip excluding __pycache__, logs, or any local test data
cd qgis-plugin
zip -r ../$ZIP_NAME hyperspatial_bigdata -x "*/__pycache__/*" -x "*/.DS_Store" -x "*/test/*"

cd ..
echo "Done! The plugin is ready for upload: $ZIP_NAME"
echo "File size:"
du -h $ZIP_NAME
