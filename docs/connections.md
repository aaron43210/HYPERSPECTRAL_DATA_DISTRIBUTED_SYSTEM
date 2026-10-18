# HyperSpatial Big Data — Connections & Interface Boundaries

Every arrow in the pipeline is a **rigid contract**. Breaking one of these is how
the system silently produces empty/garbage output instead of crashing loudly.
Treat each boundary below as an API.

---

## 1. `hyperspatial_processing` plugin → C++ engine (subprocess + stdout)

* **Invocation:** `subprocess.Popen([cli, "--input", <path>, "--output", <dir>, "--threads", <n>, "--mode", <openmp|gpu>, "--operation", <indices|pca|sam>, ...])`,
  `stderr=STDOUT`, line-buffered. Runs in a worker thread; stdout is streamed to
  the log via a `hpc_log` signal. The plugin ships **binary-free** — `cli` is the
  *external* engine, resolved at run time by `engine_locator.resolve_engine()` in
  priority order: a saved setting / **Browse** choice, then `hyperspatial_cli` on
  the system `PATH`, then a dev-only bundled `bin/<platform>/` copy.
  `engine_locator.engine_env()` prepends QGIS's own GDAL library directory so the
  engine links the exact GDAL QGIS uses. On Windows the child is launched with
  `CREATE_NO_WINDOW` to suppress a console flash. The **Backend** dropdown sets
  `--mode` (`openmp` CPU or `gpu` cross-vendor OpenCL); `gpu` self-falls-back to
  CPU if no OpenCL GPU is present, so it is always safe to pass.
* **Operation contract:** `--operation indices` writes `tile_r*.tif`; `pca` writes
  a single `pca.tif` (plus `--pca-components` / `--pca-matrix`); `sam` writes
  `sam_angle.tif` (plus optional `sam_mask.tif` / `sam_similarity.tif`) and
  requires `--sam-spectrum` (the plugin fills it from a canvas-picked pixel).
* **Progress contract:** the plugin only *displays* stdout; it does not parse
  progress. The final line `RESULT mode=... tiles=... time_sec=...` exists for
  benchmark scraping — keep its shape stable.
* **Output contract:** for `indices` the engine writes `tile_r####_c####.tif`
  into `--output` (no `summary.csv` — it was removed). The engine **clears its own
  prior outputs** at startup so re-runs don't accumulate.

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
  (Sedona `RS_FromGeoTiff` via the Spark `binaryFile` reader) — the GeoTIFFs
  *are* the bridge; there is no `summary.csv`. The tiles must therefore still
  exist when analytics runs, so keep the per-tile GeoTIFFs on disk (do not delete
  them after building the VRT mosaic).
* Every tile keeps the uniform `224 + 7` band layout, so NDVI is band
  `RS_NumBands − 6` (= `first_index_band`), NDWI `−5`, …, MSAVI `−0`. Index band
  numbers are derived from `RS_NumBands`, never hard-coded.
* CRS: each raster carries its own SRID (`RS_SRID`). `RS_ZonalStats` reprojects
  the zone geometry into the raster CRS internally, so AOI/field polygons are
  passed in EPSG:4326 and need no manual transform. The intersect pre-filter uses
  `ST_Transform(ST_SetSRID(RS_Envelope(raster), RS_SRID(raster)), 'epsg:4326')`.
* Query outputs are written as **CSV directories**. Do **not** introduce
  GeoParquet/pyarrow: bundling `pyarrow` in the Spark JVM collides with the
  system `libgdal` on macOS and crashes the process.

## 4. `hyperspatial_analytics` plugin → Spark job (subprocess argument contract)

`spark-submit --packages <sedona jars> <job>.py <args>`, with `HADOOP_CONF_DIR=""`
in the child env (unset it — a populated value triggers local XML parsing that
crashes the JVM on macOS) and with `PYTHONHOME`/`PYTHONPATH` removed (QGIS's
embedded Python may export them; the Spark env's interpreter would load QGIS's
stdlib and die before Spark starts). The Sedona JVM jars **must** be on the `spark-submit`
command line (a `SedonaContext` config can't add them after the JVM starts):

```
--packages org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.5.1,org.datasyslab:geotools-wrapper:1.5.1-28.2
```

`spark-submit` is resolved with `shutil.which`; if absent (Finder-launched QGIS)
the user picks it via **Browse**. **The argument names are not interchangeable
between jobs:**

| Job | Required arguments | Optional |
|-----|--------------------|----------|
| `q1_aoi_filter.py` | `--rasters <dir>` `--output <dir>` `--aoi <geojson>` | — |
| `q2_spatial_join.py` | `--rasters <dir>` `--agri <geojson>` `--output <dir>` | — |
| `q3_ndvi_filter.py` | `--rasters <dir>` `--output <dir>` | `--threshold <float>` (default 0.4), `--raster-out <dir>` (per-pixel veg-mask GeoTIFF per tile) |
| `q_raw_ndvi.py` (Raw cube NDVI) | `--rasters <dir>` `--output <dir>` `--red-band <n>` `--nir-band <n>` | `--nodata <float>` (the cube's sentinel; the plugin always passes it when the cube declares one), `--threshold <float>`, `--raster-out <dir>` |
| `q5_kmeans.py` (K-Means land cover) | `--rasters <dir>` `--output <dir>` | `--raster-out <dir>`, `--k <2-254>` (8), `--features shape_brightness\|shape\|raw`, `--sample <(0,1]>` (0.1), `--nodata <float>`, `--scale <float>` (stored → reflectance, EnMAP 0.0001), `--roles blue,green,…`, `--seed` (42), `--max-iter` (50), `--verify` |

> `--rasters` is a **directory** holding `tile_r*.tif`. For Q1–Q3 the plugin does
> **not** pass the engine output directory itself: a worker thread first copies
> only the 7 trailing index bands of every engine tile into a temp directory
> (`raster_prep.extract_index_tiles`) and passes that. Decoding the full
> 224 + 7-band tiles (~242 MB per 512-px tile) exhausted the Spark heap — every
> one of Q1–Q3 hit `OutOfMemoryError` on an EnMAP scene even at 6 GB — while
> the 7-band copies (~3.5 MB) run at the default 1 GB. Band order is kept, so
> index addressing by offset from `RS_NumBands` is unchanged, and the temp dir
> keeps earlier outputs under the tiles dir (e.g. `tile_r*_vegmask.tif`) out of
> the recursive `tile_r*.tif` glob. For `q_raw_ndvi.py` / `q5_kmeans.py` it is a
> directory of GeoTIFFs the plugin prepares from the raw cube (bands
> auto-detected from wavelength metadata).
>
> **Raw-cube tiles carry raw stored values with no scale/offset and no nodata
> flag** (`raster_prep.prepare_band_tiles`). GeoTools applies any GeoTIFF scale on
> read, which would shift the nodata sentinel (EnMAP −32768 → −3.2768); the
> sentinel therefore travels as `--nodata` instead. Q5 tiles are 256 px (a whole
> scene in one raster row exhausts the default 1 GB driver heap).
>
> **Q5 outputs:** `--output/clusters.csv` (`cluster, suggested_label, pixels, pct,
> ndvi, ndwi, ndsi, refl_<role>…`) and `--output/run_info.csv` (key,value), both
> plain CSV files written by the driver; `--raster-out` gets one unsigned-byte
> class-map GeoTIFF per tile (value = cluster id, nodata 255), which the plugin
> mosaics into one VRT.
> Every job requires `--output`. A wrong/missing name makes argparse exit
> non-zero *before Spark starts* — the classic "the button does nothing" failure.

`--output` is a Spark **output directory** (part-files), written with
`mode("overwrite")`. The plugin places them under `<rasters_dir>/sedona_results/<job>/`.

### AOI / boundaries GeoJSON

`--aoi` / `--agri` take a GeoJSON **file path** (geometry, Feature, or
FeatureCollection — a collection is dissolved with `shapely.unary_union`),
EPSG:4326 (lon/lat) by spec. **Required** for Q1/Q2 — the plugin aborts with a
prompt if none is chosen (there is no bundled demo fallback).

## 5. spark-sedona ↔ its two copies

There are **two** `spark-sedona/` trees: the top-level source and the copy
bundled inside the analytics plugin
(`qgis-plugin/hyperspatial_analytics/hyperspatial_analytics/spark-sedona/`).
**The plugin runs the bundled copy.** Any job fix must be applied to both, or the
packaging step must sync them, or they drift and the plugin runs stale logic.
