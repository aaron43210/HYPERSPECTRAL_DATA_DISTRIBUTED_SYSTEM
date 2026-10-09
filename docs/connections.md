# HyperSpatial Big Data — Connections & Interface Boundaries

Every arrow in the pipeline is a **rigid contract**. Breaking one of these is how
the system silently produces empty/garbage output instead of crashing loudly.
Treat each boundary below as an API.

---

## 1. QGIS Plugin → C++ engine (subprocess + stdout)

* **Invocation:** `subprocess.Popen([cli, "--input", <path>, "--output", <dir>, "--threads", <n>, "--mode", <openmp|gpu>])`,
  `stderr=STDOUT`, line-buffered. Runs in a worker thread; stdout is streamed to
  the log via `hpc_log` signal. The HPC tab's **Backend** dropdown sets `--mode`
  (`openmp` CPU or `gpu` cross-vendor OpenCL); `gpu` self-falls-back to CPU if no
  OpenCL GPU is present, so it is always safe to pass.
* **Progress contract:** the plugin only *displays* stdout; it does not parse
  progress. The final line `RESULT mode=... tiles=... time_sec=...` exists for
  benchmark scraping — keep its shape stable.
* **Output contract:** the engine writes `tile_r####_c####.tif` and
  `summary.csv` into `--output`. The engine **clears its own prior outputs**
  (`summary.csv` + `tile_r*`) at startup so re-runs don't accumulate.

## 2. C++ engine → QGIS raster layers (VRT mosaic)

* Every output tile **must have an identical band count** (224 raw + 7 index).
  `gdalbuildvrt` silently **skips** any tile whose band count differs from the
  first, leaving holes — this is why `BadBandRemover` *masks* instead of erasing.
* The plugin reads `RasterCount` from `tiles[0]`, computes
  `first_index_band = RasterCount − 7 + 1`, and builds one VRT per index with
  `BuildVRTOptions(bandList=[first_index_band + k])` for fixed offset `k`.
  **The offset must not depend on the index checkboxes** (unchecking NDVI must
  not shift NDWI onto NDVI's band).
* Nodata is `0.0` on every band; the canvas stretch for index layers is −1…1.

## 3. C++ engine → Spark (the raster bridge)

* The analytics layer consumes the engine's **`tile_r*.tif` rasters directly**
  (Sedona `RS_FromGeoTiff` via the Spark `binaryFile` reader) — this is the data
  channel, not the old `summary.csv`. The tiles must therefore still exist when
  analytics runs: use the HPC tab's **VRT output mode** (merged-TIFF mode deletes
  the per-tile GeoTIFFs after mosaicking).
* Every tile keeps the uniform `224 + 7` band layout, so NDVI is band
  `RS_NumBands − 6` (= `first_index_band`), NDWI `−5`, …, MSAVI `−0`. Index band
  numbers are derived from `RS_NumBands`, never hard-coded.
* CRS: each raster carries its own SRID (`RS_SRID`). `RS_ZonalStats` reprojects
  the zone geometry into the raster CRS internally, so AOI/field polygons are
  passed in EPSG:4326 and need no manual transform. The intersect pre-filter uses
  `ST_Transform(ST_SetSRID(RS_Envelope(raster), RS_SRID(raster)), 'epsg:4326')`.
* Query outputs are written as **CSV directories**. Do **not** introduce
  GeoParquet/pyarrow: bundling `pyarrow` in the Spark JVM collides with the
  system `libgdal` on macOS and crashes the process. (`summary.csv` is still
  written by the engine as a lightweight tile index, but the jobs don't read it.)

## 4. QGIS Plugin → Spark job (subprocess argument contract)

`spark-submit <job>.py <args>`, with `HADOOP_CONF_DIR=""` in the child env
(unset it — a populated value triggers local XML parsing that crashes the JVM).
**The argument names are not interchangeable between jobs:**

| Job | Required arguments | Optional |
|-----|--------------------|----------|
| `q1_aoi_filter.py` | `--rasters <dir>` `--output <dir>` `--aoi <geojson>` | — |
| `q2_spatial_join.py` | `--rasters <dir>` `--agri <geojson>` `--output <dir>` | — |
| `q3_ndvi_filter.py` | `--rasters <dir>` `--output <dir>` | `--threshold <float>` (default 0.4) |
| `q4_partitioning_benchmark.py` | `--rasters <dir>` `--agri <geojson>` | `--csv-out <path>` |

> `--rasters` is the HPC output **directory** holding `tile_r*.tif`. Every job
> requires `--output`. A wrong/missing name makes argparse exit non-zero *before
> Spark starts* — the classic "the button does nothing" failure.

`--output` is a Spark **output directory** (part-files), written with
`mode("overwrite")`. The plugin places them under `<rasters_dir>/sedona_results/<job>/`.

### AOI / boundaries GeoJSON

`--aoi` / `--agri` take a GeoJSON **file path** (geometry, Feature, or
FeatureCollection — a collection is dissolved with `shapely.unary_union`),
EPSG:4326 (lon/lat) by spec. **Required** for Q1/Q2 — the plugin aborts with a
prompt if none is chosen (there is no bundled demo fallback any more).

## 5. QGIS Plugin → DLR STAC API (HTTP)

* `GET {STAC_URL}/items?bbox=w,s,e,n&datetime=start/end&limit=N`, parsed as STAC
  JSON. Download href is taken from `assets.data` → `SPECTRAL_IMAGE` → `image`.
* **TLS is verified** (certifi CA bundle preferred). The scene download sends the
  user's EOWEB bearer token in an `Authorization` header — it must never travel
  over an unverified connection.
* A `text/html` response on download means the server redirected to the login
  page: treated as an authentication failure, not saved as a `.tif`.

## 6. spark-sedona ↔ its two copies

There are **two** `spark-sedona/` trees: the top-level source and the copy
bundled inside the plugin (`qgis-plugin/hyperspatial_bigdata/spark-sedona/`).
**The plugin runs the bundled copy.** Any job fix must be applied to both, or the
packaging step must sync them, or they drift and the plugin runs stale logic.
