# HyperSpatial Big Data

A **local-first** pipeline for processing gigabyte-scale hyperspectral satellite
imagery (EnMAP L2A) into map layers and distributed spatial analytics — entirely
on local hardware, with **no cloud dependencies**.

It has three components that communicate only through files on disk and
subprocess stdout:

| Component | Tech | Role |
|-----------|------|------|
| **HPC Engine** (`hpc-engine/`) | C++17 · GDAL · OpenMP / OpenCL / MPI | Tiles a scene and computes 7 spectral indices per pixel |
| **QGIS Plugin** (`qgis-plugin/`) | PyQt5 · GDAL | Unified UI: download → preprocess → visualise → analyse |
| **Sedona Analytics** (`spark-sedona/`) | PySpark · Apache Sedona | Distributed per-pixel spatial SQL over the output rasters |

> **Full design docs:** [`docs/architecture.md`](docs/architecture.md) (function-by-function
> breakdown, data lifecycles, maths, concurrency/memory models, known limitations)
> and [`docs/connections.md`](docs/connections.md) (the rigid interface contracts).

---

## Repository layout

```
hpc-engine/              C++17 preprocessor (include/ headers, src/ sources, src/gpu/ OpenCL)
qgis-plugin/
  hyperspatial_bigdata/  the installable QGIS plugin
    bin/                 prebuilt hyperspatial_cli (required by the plugin)
    services/            STAC download + metadata client
    spark-sedona/        bundled copy of the Spark jobs the plugin runs
spark-sedona/            source of truth for the Spark/Sedona jobs
docs/                    architecture.md, connections.md
data/                    sample EnMAP scene (NOT committed — see .gitignore)
package_plugin.sh        builds the installable plugin zip
environment.yml          conda env (Python 3.11, GDAL, PySpark, Sedona 1.5.1)
```

---

## Spectral indices

For each tile the engine appends 7 index bands (fixed order): **NDVI, NDWI, NDBI,
SAVI, EVI, NDMI, MSAVI**. The NIR/Red/Green/SWIR/Blue bands are selected
automatically by **nearest centre wavelength** read from the raster metadata
(overridable with `--nir-band` etc.), so the indices are physically correct for
any cube — not a hard-coded band layout. See
[`docs/architecture.md` §1.4–1.5](docs/architecture.md).

---

## Quick start

### 1. Build the C++ engine

Prerequisites: a C++17 compiler, GDAL, and (optional) OpenMP / OpenCL / MPI.
On macOS: `brew install gdal libomp cmake`.

```bash
cmake -S hpc-engine -B hpc-engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build hpc-engine/build -j
cp hpc-engine/build/hyperspatial_cli qgis-plugin/hyperspatial_bigdata/bin/
```

The build defaults to a **portable** `-O3`. Only add `-DNATIVE_ARCH=ON` for a
machine-local build (a `-march=native` binary is not portable to other CPUs).

### 2. Set up the Python / Spark environment

```bash
conda env create -f environment.yml
conda activate hyperspatial
```

### 3. Package & install the QGIS plugin

```bash
./package_plugin.sh
```

Then in QGIS: **Plugins → Manage and Install Plugins → Install from ZIP →**
`hyperspatial_bigdata.zip`.

### 4. Use the plugin (three tabs)

1. **STAC Downloader** — search the DLR EnMAP STAC catalogue by bbox/date and
   download a scene (optional EOWEB bearer token for restricted assets).
2. **HPC Preprocessing** — pick the input raster + output directory, choose the
   CPU/GPU backend, and run. Use **Virtual Raster (.vrt)** output mode if you plan
   to run analytics (merged-TIFF mode deletes the per-tile GeoTIFFs). Index layers
   are added to the canvas automatically.
3. **Sedona Analytics** — point at the tiles directory, pick a query (AOI filter /
   spatial join / NDVI filter), supply an AOI GeoJSON where required, and run.

### Running the engine directly (CLI)

```bash
hpc-engine/build/hyperspatial_cli \
  --input  path/to/SPECTRAL_IMAGE.bsq \
  --output path/to/out_dir \
  --mode   openmp          # serial | openmp | gpu
```

Outputs `tile_r####_c####.tif` (N raw + 7 index bands) and `summary.csv` into the
output directory. Prints the resolved bands and a final
`RESULT mode=... tiles=... time_sec=...` line; exits non-zero if any tile fails.

### Running a Spark job directly

```bash
HADOOP_CONF_DIR="" spark-submit spark-sedona/jobs/q3_ndvi_filter.py \
  --rasters path/to/out_dir --output path/to/out_dir/sedona_results/q3 --threshold 0.4
```

`HADOOP_CONF_DIR` **must be empty** on macOS (a populated value crashes the JVM).

---

## Notes & constraints

- **No cloud** services, and no reading gigabyte files into Python memory — all
  per-pixel maths is in C++ (one tile per worker), all spatial joins in Sedona.
- **No GeoParquet / pyarrow** in the Spark JVM (fatal `libgdal` symbol collision on
  macOS); data transfer is GeoTIFF + CSV + WKT.
- The sample `data/` scene (~650 MB) is **git-ignored**; download your own via the
  plugin's STAC Downloader tab.
- Known limitations (per-tile normalisation seams, Sedona axis-order on some
  GeoTools builds, etc.) are catalogued in
  [`docs/architecture.md` §6](docs/architecture.md).

---

## License

Add your preferred license here (e.g. MIT) before publishing.
