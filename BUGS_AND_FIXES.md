# HyperSpatial Big Data — Bug Audit & Fixes

Full-repo review of the C++ HPC engine, the QGIS plugin, and the Spark/Sedona
analytics layer. This is the **second audit pass (2026-10-07)**: it re-verifies
the first pass's fixes against the actual code and adds the issues that pass
missed — most importantly, that **every Sedona query was invoked with the wrong
command-line arguments**, so the entire analytics tab was non-functional.

Legend: 🔴 critical (breaks a core workflow) · 🟠 major · 🟡 minor/robustness · ⚪ informational.
Status: **FIXED** = changed in this pass · **FIX²** = re-verified fix from pass 1 · **FLAG** = left for your decision.

---

## 0. Headline — what pass 1 missed

### H1 🔴 FIXED — The QGIS plugin called **every** Spark job with the wrong arguments
**`qgis-plugin/hyperspatial_bigdata/dialog.py` → `run_analytics`**

The dialog built one argument list for all three jobs:
```python
spark_args = ["spark-submit", script_path, "--input", self.le_csv.text()]
# + ["--aoi"/"--agri", geojson]  for q1/q2
```
But the jobs' argparse contracts are:

| Job | Requires | Got from plugin | Result |
|-----|----------|-----------------|--------|
| `q1_aoi_filter.py` | `--input --output --aoi` | `--input --aoi` | **missing `--output`** → argparse exit 2 |
| `q2_spatial_join.py` | `--tiles --agri --output` | `--input --agri` | **wrong name + missing `--tiles`/`--output`** → exit 2 |
| `q3_ndvi_filter.py` | `--input --output [--threshold]` | `--input` | **missing `--output`** → exit 2 |

Every "RUN SPARK JOB" press failed at argument parsing **before Spark started**.
The job printed a `usage:` error to stdout, the subprocess exited non-zero, and
the UI just said the job "completed" (it never checked the exit code). Pass 1's
"C1" fixed the AOI *parsing* inside Q1 — but Q1 (and Q2/Q3) could never be reached.

**Fix:** `run_analytics` now builds the exact argument list each job declares,
always passes an `--output` directory (`<csv_dir>/sedona_results/<job>/`), uses
`--tiles` for Q2, and passes the Q3 `--threshold` from a new spin box.
`_spark_task` now reports the child's **exit code** and the output path.

### H2 🔴 FIXED — Q1/Q2 were hard-wired to the bundled demo AOI
The UI had removed the AOI input ("auto-generated invisibly per UX
requirements"), so Q1/Q2 always used the bundled `agri_boundaries.geojson` —
fixed sample polygons near California. A real EnMAP scene (e.g. UTM zone 32N over
Germany) never intersects them, so even once H1 was fixed the join would return
0 rows on real data, with no indication why.
**Fix:** added an optional **AOI / boundaries GeoJSON** picker. User file is used
if provided; otherwise it falls back to the demo *and logs a clear warning* that
the demo won't match real scenes.

---

## A. C++ HPC Engine (`hpc-engine/`)

### A1 🔴 FIX² — Uniform per-tile band count (mask, don't erase)
`BadBandRemover::apply` zero-fills bad bands instead of erasing them, so every
tile keeps the same `224 + 7` layout and the VRT mosaic can't develop holes.
Verified present (`std::fill(...)` in `BadBandRemover.hpp`).

### A2 🔴 FIX² — Spectral indices require all operands non-zero
`SpectralFeatures.hpp` guards each index on all its operand bands being non-zero
(nodata = 0.0), preventing spurious ±1 from half-nodata pixels. Verified. The
formulas (NDVI/NDWI/NDBI/NDMI/SAVI/EVI/MSAVI) are mathematically correct and the
10 000 reflectance de-scaling is applied before the additive-constant indices.

### A3 🟠 FIXED — `main_mpi.cpp` arg parsing could abort the job
`parseArgs` called `std::stoi/std::stof` directly; a non-numeric `--tile-size`
threw `std::invalid_argument` and aborted the whole MPI run.
**Fix:** added the same guarded `toInt/toFlt` helpers used by `main.cpp`.
(`#include <thread>` from pass 1 verified present.)

### A4 🟠 FIXED — `summary.csv` accumulated duplicate rows across runs
**`src/main.cpp`** opened `summary.csv` in append mode and never cleared it, and
tiles are written to deterministic names. Re-running the engine into the same
output directory therefore **appended a second full set of rows** (doubling every
tile for Spark) and left **stale `tile_*.tif`** from a prior, differently-tiled
scene that the plugin's `gdalbuildvrt` would fold into the mosaic. (The MPI path
truncated its CSV but also left stale tiles.)
**Fix:** at startup both engines now remove their own prior outputs
(`summary.csv` + `tile_r*`) from the output directory (collect-then-remove, so
directory iteration isn't mutated in place).

### A5 🟠 FIXED — Normalizer collapsed the darkest valid pixel onto nodata
**`include/Normalizer.hpp`** min-max normalised into `[0, 1]`, so any pixel equal
to the band minimum became exactly `0.0` — which the GeoTIFF writer and every
downstream reader treat as **nodata**. Valid data was silently reclassified as
missing.
**Fix:** normalise into `[kEps, 1]` (`kEps = 1e-6`), reserving `0.0` for true
nodata.

### A6 🟡 FLAG — Normalisation stretch is **per-tile** (mosaic seams)
Still true by design: each tile computes its own min/max, so identical
reflectance maps to different display values in adjacent tiles → visible
brightness seams in the raw-band mosaic. (Index bands are unaffected — they're
computed before normalisation and are not normalised.) Documented in
`Normalizer.hpp`. *Recommendation:* for scientific output, disable normalisation
(physical reflectance is already mosaic-consistent); or add a cheap global
two-pass that stores only per-band min/max.

### A7 🟡 FLAG — `GDALAllRegister()` called from worker threads
`TileWriter::writeGeoTIFF`/`RasterReader::readTile` call it inside the OpenMP
loop. Idempotent and already called in the `RasterReader` ctor before threads
start, so benign; best practice is one call in `main()` before the parallel
region.

### A8 ⚪ FLAG — `BadBandRemover` runs before `SpectralFeatures`
A band that is >50 % nodata is masked entirely, so its ~40 % valid pixels never
contribute to any index. Defensible definition of "bad band"; swap the order if
you want to keep those pixels.

---

## B. QGIS Plugin — `qgis-plugin/hyperspatial_bigdata/`

### B1 🔴 FIX² — Fixed index-band offsets regardless of checkboxes
`_hpc_task` computes `band = first_index_band + fixed_offset`, so unchecking an
index no longer shifts later layers onto the wrong band. Verified.

### B2–B4 🟠 FIX² — Thread-safe UI recovery
Spark/HPC buttons re-enable via `spark_finished`/`hpc_finished` signals emitted
in `finally`; checkbox state is captured on the GUI thread and passed to the
worker. Verified.

### B5 🔴 FIXED — Plugin failed to load when the locale was unset
**`hyperspatial_bigdata.py`**: `QSettings().value('locale/userLocale')[0:2]`
raised `TypeError` when the setting is `None` (common on fresh profiles),
aborting `__init__` so the plugin never loaded.
**Fix:** `str(QSettings().value('locale/userLocale') or 'en')[0:2]`.

### B6 🟠 FIXED — Modeless dialog / `exec()` removed
`run()` uses `show()/raise_()/activateWindow()` (no modal `exec()`), honouring
"never block the UI". Verified.

### B7 🔴 FIX² (security) — Verified TLS context
`services/enmap_client.py` uses `ssl.create_default_context` (certifi preferred),
so the EOWEB bearer token is never sent over an unverified connection. Verified.

### B8 🟡 FIXED — `search()` could raise out of the slot
The bbox parse sat *before* the `try`, so a malformed `W,S,E,N` string raised
uncaught. **Fix:** parse inside the `try`, validate 4 components, and show an
empty-result info box.

### B9 🟡 FIXED — `chmod` on the bundled binary was unguarded
A read-only install location would throw `OSError` and abort the run.
**Fix:** wrapped in try/except with a warning in the log.

### B10 🟡 FIXED — Hard-coded `zsh` for `spark-submit`
`run_analytics` wrapped the command in `["zsh", "-l", "-c", ...]`, which fails
where zsh isn't the shell. **Fix:** resolve `spark-submit` with `shutil.which`;
if not on PATH (QGIS launched from the GUI), fall back to `$SHELL -l -c`
(`/bin/bash` default), not a hard-coded shell.

### B11 🟡 FLAG — Metadata-derived band indices are never wired in
`enmap_client.extract_metadata()` computes per-scene NIR/Red/Green/SWIR indices
but `run_hpc` never passes `--nir-band/...`, so the engine uses its hard-coded
EnMAP defaults. Those defaults are correct for standard 224-band EnMAP L2A, so
this is fine today, but a reordered/non-EnMAP raster would use wrong bands. Note
the module's **linear fallback** wavelengths are only approximate, so wiring
`closest()` without a parsed XML could pick worse bands — wire it only when the
real XML wavelengths parse. *Recommend:* call `extract_metadata` after download
and pass indices when a parsed `metadata.json` exists; else delete the unused
functions.

### B12 🟡 FLAG — Dead, forbidden GeoParquet loader
`services/qgis_loader.py` (`load_geoparquet`) is imported nowhere and
contradicts the "AVOID GEOPARQUET" rule. *Recommend deleting the file.*

### B13 🟡 FLAG — Downloaded asset is always named `.tif`
`download_scene` writes `<scene_id>.tif`; EnMAP "data" assets can be ZIPs. Derive
the extension from the URL/`Content-Type`, or unzip before handing to the engine.

---

## C. Spark / Sedona (`spark-sedona/` **and** the bundled copy)

The job CLIs are correct; the **caller** (the dialog) was wrong — see H1. No job
code changes were needed in this pass. Verified the bundled copies the plugin
actually runs (`qgis-plugin/hyperspatial_bigdata/spark-sedona/`) match the
documented contracts (`q1: --input/--output/--aoi`, `q2: --tiles/--agri/--output`,
`q3: --input/--output/--threshold`).

### C1 🔴 FIX² — Q1 AOI parsed in Python → single WKT polygon (EPSG:4326)
`_load_aoi_wkt` reads a file or raw string, dissolves a FeatureCollection with
`shapely.unary_union`, and hands Sedona one WKT polygon. Verified in both copies.

### C2 🟠 FIX² — All jobs standardized on `ST_SetSRID(ST_GeomFromWKT(...), epsg)`.

### C3 🟠 FLAG — Verify `ST_Transform(geom, 'epsg:4326')` on real UTM data
Confirm against your Sedona 1.5.1 / GeoTools build: (1) 2-arg `ST_Transform`
resolves; if not, use the 3-arg form; (2) **axis order** — GeoTools treats
EPSG:4326 as lat/lon, so a known-overlapping join coming back empty means
coordinates are swapped (wrap with `ST_FlipCoordinates`).

### C4 🟡 FLAG — Q4 "spatial" partitioning isn't spatial
`q4_partitioning_benchmark.py` uses `repartition(200, "geometry")` (hash on a
serialized geometry), not Sedona spatial partitioning. Standalone benchmark; not
wired into the plugin.

### C5 🟡 FLAG — Keep the two `spark-sedona/` trees in sync
`package_plugin.sh` doesn't sync the top-level `spark-sedona/` into the plugin, so
the bundled copy is hand-maintained and drifts. *Recommend* syncing in packaging.

### C6 ⚪ FLAG — `generate_sample_data.py` overwrites `data/intermediate/summary.csv`
Triggered only when the demo GeoJSON is missing, but it clobbers any real
`summary.csv` under the plugin's `data/intermediate/`. Minor; it writes to the
plugin's bundled data dir, not the user's chosen output.

---

## D. Repo / build / packaging / docs

### D1 🟠 FIXED — Referenced docs now exist
`AGENTS.md` mandates reading `architecture.md` and `connections.md`, which were
absent. **Created** both under `docs/` (accurate to the current code: pipeline,
band layout, index formulas, the `summary.csv` schema, and every interface
contract).

### D2 🟠 FIXED (build) — `-march=native` on a shipped binary
`CMakeLists.txt` compiled with `-march=native`; the plugin ships a **prebuilt**
`bin/hyperspatial_cli`, and such a binary aborts with SIGILL on an older/different
CPU. **Fix:** default to portable `-O3`; opt back in for local-only builds with
`-DNATIVE_ARCH=ON`.

### D3 🟠 FLAG — Legacy duplicate plugins contradict the single-plugin architecture
`qgis-plugin/hpc_processor_plugin/` and `qgis-plugin/sedona_analytics_plugin/`
(and the three root `*.zip`) are the pre-unification split plugins. They
duplicate and independently re-break the same code (the Spark-arg bug H1 also
lives in `sedona_analytics_plugin`). All fixes in this pass target the canonical
`hyperspatial_bigdata` only. *Recommend removing both legacy trees and the stale
zips* to end the 3-way drift. (Deletion left for your confirmation.)

### D4 🟡 FLAG — `package_plugin.sh` comment is misleading
It claims it excludes "massive compiled binaries", but the plugin **requires** the
bundled `bin/hyperspatial_cli`; the script correctly keeps it and produces an
~87 KB zip, not "12 KB". Update the comment so nobody deletes the binary.

### D5 🟡 FLAG — `test_vrt.py` has hard-coded local paths
Scratch script scanning `/Users/aaronr/Downloads/out/tile_*.tif`; its
`bandList=[n-6]` (NDVI) is correct given the uniform-band layout. Parameterize or
move to a tests/ dir.

### D6 ⚪ INFO — `pyarrow`/`geopandas` present but unused by jobs
Installed from conda-forge to share `libarrow`; the jobs avoid both (CSV +
`json`/`shapely`). Intentional. No action.

---

## Files changed in this pass

| File | Change |
|---|---|
| `hpc-engine/src/main.cpp` | A4 — clear stale `summary.csv` + `tile_r*` before run |
| `hpc-engine/src/main_mpi.cpp` | A3 — guarded arg parsing; A4 — clear stale tiles (rank 0) |
| `hpc-engine/include/Normalizer.hpp` | A5 — normalise into `[kEps,1]`; document per-tile seams |
| `hpc-engine/CMakeLists.txt` | D2 — portable `-O3` default, `-DNATIVE_ARCH=ON` opt-in |
| `qgis-plugin/hyperspatial_bigdata/dialog.py` | H1 — correct per-job Spark args + `--output` + exit code; H2 — AOI picker; Q3 threshold; B8 — `search()` try; B9 — guarded `chmod`; B10 — portable shell |
| `qgis-plugin/hyperspatial_bigdata/hyperspatial_bigdata.py` | B5 — guard unset locale |
| `docs/architecture.md`, `docs/connections.md` | D1 — created |

## Still needs your decision
B11 (wire or delete metadata fns) · B12 (delete `qgis_loader.py`) · B13 (asset
extension) · C3 (axis-order verification on real UTM) · C4 (q4 partitioning) ·
C5 (sync spark trees in packaging) · D3 (remove legacy plugins + stale zips) ·
D4 (fix packaging comment).
