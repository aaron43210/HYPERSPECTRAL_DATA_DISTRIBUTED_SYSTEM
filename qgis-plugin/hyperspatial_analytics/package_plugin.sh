#!/usr/bin/env bash
#
# Packages the standalone "HyperSpatial Sedona Analytics" QGIS plugin into a
# single zip for installation / upload to plugins.qgis.org.
#
# The plugin bundles its Spark/Sedona jobs under spark-sedona/ (the plugin runs
# that bundled copy directly), so they are intentionally INCLUDED. Only caches
# and OS cruft are excluded. Apache Spark itself is a user-provided runtime
# dependency and is NOT bundled.
set -euo pipefail

# Run from this script's own directory (the outer plugin dir) so the zip layout
# is correct no matter where it is invoked from.
cd "$(dirname "$0")"

PLUGIN_DIR="hyperspatial_analytics"
ZIP_NAME="hyperspatial_analytics.zip"

echo "Packaging QGIS Plugin..."

# Remove old zip if it exists.
rm -f "$ZIP_NAME"

# Create zip excluding caches and OS cruft.
zip -r "$ZIP_NAME" "$PLUGIN_DIR" \
    -x "*/__pycache__/*" \
    -x "*/.DS_Store" \
    -x "*/test/*"

echo "Done! The plugin is ready for upload: $ZIP_NAME"
echo "File size:"
du -h "$ZIP_NAME"
