# HyperSpectral Distributed System

A local-first pipeline for processing gigabyte-scale hyperspectral satellite
imagery (EnMAP L2A) into map layers and running distributed spatial analytics
over the results — entirely on local hardware, with no cloud dependencies. A
native C++17 engine does all per-pixel math tile-by-tile, two QGIS plugins
provide the user interface, and Apache Sedona runs distributed spatial SQL
directly over the engine's GeoTIFF outputs.

- GitHub: https://github.com/aaron43210/HYPERSPECTRAL_DATA_DISTRIBUTED_SYSTEM
- Author: Aaron R, Digital University Kerala
- License: GPL-2.0

## Repository layout

```
hpc-engine/                          C++17 preprocessor (CMakeLists.txt, include/, src/, src/gpu/ OpenCL, tests/)
qgis-plugin/
  hyperspatial_processing/           Primary QGIS plugin (submitted to plugins.qgis.org; ships binary-free)
  hyperspatial_analytics/            PySpark + Apache Sedona analytics plugin (local; not submitted)
spark-sedona/                        Source of truth for the Spark/Sedona jobs (jobs/, utils/, requirements.txt)
docs/                                architecture.md, connections.md
environment.yml                      conda env (Python 3.11, GDAL, PySpark, Sedona 1.5.1)
```

## The two plugins

### 1. HyperSpectral Processer (`qgis-plugin/hyperspatial_processing/`)

The primary plugin, submitted to plugins.qgis.org and shipped binary-free. A thin
PyQt5 dialog that drives the external C++17 engine `hyperspatial_cli`. A single
Operation selector offers:

- **Spectral indices** — NDVI, NDWI, NDBI, SAVI, EVI, NDMI, MSAVI.
- **PCA** — principal-component dimensionality reduction producing a false-colour
  PC composite.
- **SAM** — Spectral Angle Mapper target detection from a reference pixel picked
  on the map, with an optional detection mask and similarity band.

Compute runs on the GPU through OpenCL where available, with automatic multi-core
OpenMP CPU fallback that produces identical results. Outputs are per-tile
GeoTIFFs (`tile_r####_c####.tif`) plus a VRT mosaic (or a merged GeoTIFF), loaded
straight onto the QGIS canvas.

The engine is not inside the published zip. It is resolved at run time from a
saved path, the system `PATH`, or a bundled `bin/`, or fetched by the plugin's
"Download engine" button from the GitHub Release.

### 2. HyperSpatial Sedona Analytics (`qgis-plugin/hyperspatial_analytics/`)

A local plugin (not submitted to plugins.qgis.org). A PyQt5 dialog that runs
PySpark + Apache Sedona (`sedona-spark-shaded 1.5.1` + `geotools-wrapper
1.5.1-28.2`) raster jobs via `spark-submit` (the Spark install is located through
a Browse row). Jobs:

- **Q1 — AOI filter:** keep tiles intersecting an area of interest.
- **Q2 — spatial join:** per-field zonal statistics (NDVI/NDWI/NDBI) against a
  vector layer.
- **Q3 — NDVI filter:** threshold NDVI, with an optional per-pixel vegetation-mask
  GeoTIFF added to the canvas.
- **Q4 — Raw-cube NDVI:** computes NDVI per pixel straight from a raw BSQ/ENVI/
  GeoTIFF cube via Sedona `RS_MapAlgebra`, independent of the C++ engine
  (RED/NIR bands auto-detected from wavelength metadata).

Spark runs as a background subprocess so the QGIS UI is never blocked; query
outputs are written as CSV directories.

## The C++ engine (`hpc-engine/`)

C++17 with GDAL, OpenMP, and OpenCL (no MPI). Entry point is `src/main.cpp`,
selected with `--operation indices|pca|sam`. The engine processes a scene
tile-by-tile and never loads the whole cube into memory; each worker thread opens
its own fresh `GDALOpen` handle (GDAL datasets are not thread-safe).

### Build

Prerequisites: a C++17 compiler, GDAL, and OpenMP / OpenCL.
On macOS: `brew install gdal libomp cmake`.

```bash
cmake -S hpc-engine -B hpc-engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build hpc-engine/build -j
```

### Run directly

```bash
hpc-engine/build/hyperspatial_cli \
  --operation indices \
  --input  path/to/SPECTRAL_IMAGE.bsq \
  --output path/to/out_dir
```

## Installation

### QGIS plugins

Each plugin has its own `package_plugin.sh` that builds an installable zip:

```bash
# Processing plugin
cd qgis-plugin/hyperspatial_processing && ./package_plugin.sh

# Analytics plugin
cd qgis-plugin/hyperspatial_analytics && ./package_plugin.sh
```

Then in QGIS: **Plugins → Manage and Install Plugins → Install from ZIP** and
select the built zip. The processing plugin is also available from
plugins.qgis.org (currently flagged experimental); the analytics plugin is
install-from-ZIP only.

### Engine

The processing plugin ships without a binary. Build it with CMake (above) and
place it on the system `PATH` or configure its path in the plugin, or use the
plugin's **Download engine** button to fetch a prebuilt binary from the GitHub
Release.

## Analytics runtime setup

The analytics plugin and the standalone Spark jobs require:

- Python 3.11
- Java 8, 11, or 17
- Spark / PySpark 3.4
- apache-sedona 1.5.1 (`sedona-spark-shaded 1.5.1` + `geotools-wrapper 1.5.1-28.2`)

Create the environment from the pinned definition:

```bash
conda env create -f environment.yml
conda activate hyperspatial
```

(or `pip install -r spark-sedona/requirements.txt` into a Python 3.11 environment).

Run a Spark job directly:

```bash
HADOOP_CONF_DIR="" spark-submit spark-sedona/jobs/q3_ndvi_filter.py \
  --rasters path/to/out_dir --output path/to/out_dir/sedona_results/q3 --threshold 0.4
```

`HADOOP_CONF_DIR` must be empty on macOS; a populated value triggers local XML
parsing that crashes the JVM.

## Data contract

The C++ engine writes per-tile GeoTIFFs named `tile_r####_c####.tif`, each
containing **224 raw bands + 7 trailing index bands** (NDVI, NDWI, NDBI, SAVI,
EVI, NDMI, MSAVI, in that order). There is no `summary.csv`.

The GeoTIFFs are the bridge between the engine and the analytics layer: Sedona
reads the `tile_r*.tif` files directly via `RS_FromGeoTiff` — not through any CSV
index. Spark query results are written as CSV directories. GeoParquet / pyarrow
is deliberately avoided because it collides with the system `libgdal` on macOS;
data transfer stays GeoTIFF + CSV + WKT.

## License

Released under the GNU General Public License v2.0 (GPL-2.0).
