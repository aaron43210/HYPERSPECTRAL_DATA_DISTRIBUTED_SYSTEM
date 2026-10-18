#!/usr/bin/env bash
#
# Packages the standalone "Hyperspectral Index Processing" QGIS plugin into a
# single zip for installation / upload to plugins.qgis.org.
#
# BINARY-FREE: unlike the HyperSpatial Big Data monolith, this plugin does NOT
# bundle the compiled C++ engine. The engine (hyperspatial_cli) is resolved at
# run time from QSettings, the system PATH, or - dev builds only - a bin/ folder.
# The published zip intentionally EXCLUDES bin/ so it stays lightweight and
# submission-compliant. The user supplies the engine via the dialog's Engine
# "Browse..." button, a PATH install, or by building hpc-engine from source.
#
# A dev may drop bin/<os-arch>/hyperspatial_cli into the package for local
# testing; the -x "*/bin/*" rule keeps it out of the shipped zip regardless.
set -euo pipefail

# Run from this script's own directory (the outer plugin dir) so the zip layout
# is correct no matter where it is invoked from.
cd "$(dirname "$0")"

PLUGIN_DIR="hyperspatial_processing"
ZIP_NAME="hyperspatial_processing.zip"

echo "Packaging QGIS Plugin (binary-free)..."

# Remove old zip if it exists.
rm -f "$ZIP_NAME"

# Create zip excluding caches, OS cruft, the (dev-only) engine binary and tests.
zip -r "$ZIP_NAME" "$PLUGIN_DIR" \
    -x "*/__pycache__/*" \
    -x "*/.DS_Store" \
    -x "*/bin/*" \
    -x "*/test/*" \
    -x "*.pyc" \
    -x "*/.pytest_cache/*"

echo "Done! The plugin is ready for upload: $ZIP_NAME"
echo "File size:"
du -h "$ZIP_NAME"
