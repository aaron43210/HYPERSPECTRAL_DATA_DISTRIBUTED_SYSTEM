#!/usr/bin/env bash

# Packages the standalone HyperSpatial EnMAP Downloader QGIS plugin
# (hyperspatial_downloader) into a single zip for installation / upload to
# plugins.qgis.org.
#
# This is a lightweight, download-only plugin: it contains just Python, the
# bundled STAC client (services/) and an icon -- no compiled binaries and no
# spark-sedona jobs -- so the resulting zip is tiny.

# Run from the directory this script lives in (the outer plugin dir), so the
# package folder is zipped with the correct top-level name regardless of cwd.
cd "$(dirname "$0")"

PLUGIN_DIR="hyperspatial_downloader"
ZIP_NAME="hyperspatial_downloader.zip"

echo "Packaging QGIS Plugin..."

# Remove old zip if exists
rm -f "$ZIP_NAME"

# Create zip excluding Python caches and macOS metadata
zip -r "$ZIP_NAME" "$PLUGIN_DIR" -x "*/__pycache__/*" -x "*/.DS_Store"

echo "Done! The plugin is ready for upload: $ZIP_NAME"
echo "File size:"
du -h "$ZIP_NAME"
