# HyperSpectral Processer — Launch-Readiness Audit

Date: 2026-10-10 · Scope: C++ engine (`hpc-engine/`), QGIS plugins (`qgis-plugin/`), CI/distribution, repo. Method: three parallel read-only audits + maintainer verification on macOS (Apple M4 Pro, GDAL 3.13.3, OpenCL 1.2).

## Verdict
**Functionally launch-ready.** All three operations were validated end-to-end in QGIS and numerically against NumPy. One HIGH correctness risk (Indices GPU threading) was **found and fixed** during the audit. Remaining items are documentation drift, repo hygiene, Windows-CI reliability, and downloader polish — none block the core processing plugin, but several should be cleared before a public plugins.qgis.org release.

## Functional validation (confirmed correct)
- **Indices** — 7 bands (NDVI…MSAVI), GPU vs CPU match to ~1e-8; `summary.csv` correctly dropped; VRT/merge + canvas load work.
- **PCA** — eigenvalues/variance match NumPy to ~1e-7, projection correlation 1.000000 (GPU=CPU=NumPy); the PC1/2/3 false-colour composite renders as expected.
- **SAM** — GPU vs CPU angle match to ~7e-8; reference pixel self-angle = 0; detection mask + angle raster correct. Scale-invariance confirmed (0–1 reference vs ×10000 image is fine).
- Memory discipline (never loads the whole cube), per-thread GDAL handles, ClMem RAII (no leaks), error handling/exit codes, build portability (MSVC/clang/gcc) — all verified sound.
- Repo attribution clean (only `aaron43210`); `data/` not committed; store zip genuinely binary-free; zip root-folder structure correct on all 3 OSes.

## Findings by severity

### HIGH — FIXED
- **Indices GPU dispatch was multi-threaded on the host** (`hpc-engine/src/main.cpp:581`), unlike SAM/PCA which force single-thread in GPU mode because Apple's OpenCL queue silently corrupts under concurrent host dispatch. **Fixed** → `if(runParallel && !use_gpu)`, rebuilt, re-verified (NDVI GPU=CPU 2.6e-8). Commit the `main.cpp` change.

### MEDIUM — recommend before public release
1. **Stale docs: `summary.csv`.** `docs/connections.md` (§ output contract, line ~55), `docs/architecture.md` (§4 and refs), and `README.md` still say the engine writes `summary.csv` — now false. Also `AGENTS.md` (one level up) still says "224 raw + 3 computed bands" and a summary.csv Spark bridge.
2. **Stale docs: plugin split.** README/architecture describe the old single 3-tab `hyperspatial_bigdata` plugin with a bundled binary; the real target is the binary-free `hyperspatial_processing` + the independent downloader/analytics plugins.
3. **Windows CI never completes.** `build-plugins.yml` compiles GDAL from source via vcpkg every run (30–60+ min, no caching); Linux/macOS pass and upload, Windows has no completed run → the Windows self-contained zip is unproven. Add vcpkg binary caching (or a prebuilt GDAL).
4. **Downloader `search()` blocks the UI thread** (`hyperspatial_downloader/.../dialog.py`) — a slow DLR endpoint freezes QGIS up to 30s. Only download is threaded; thread search() too.
5. **Downloader hardcodes a 20% cloud filter** with no UI control — cloudy AOIs silently return "No results." Expose a threshold or document it.
6. **PCA GPU vs CPU parity is float32 (GPU) vs float64 (CPU)** — ~1e-6 relative covariance error; fine for a display composite, but not bit-identical cross-backend (near-degenerate PCs may order/sign-differ). Document as "display-grade."

### LOW / polish
- `main_mpi.cpp` is out of sync — still writes `summary.csv` and has no `--operation`/PCA/SAM (not built in CI; MPI absent). Sync or mark unsupported.
- Jacobi eigensolver early-exit uses an absolute `eps=1e-300` that never fires at reflectance scale → always runs 100 sweeps (correct, tiny perf cost; use a Frobenius-relative threshold).
- CPU PCA covariance reduction is run-to-run nondeterministic in the last ULPs (mutex arrival order); GPU path is deterministic.
- `BadBandRemover` variance uses the `sumSq/n − mean²` shortcut PCA avoids (edge-case misclassification on high-DN near-constant bands).
- Processing result-loader re-reads live checkboxes/format radio instead of the RUN-time snapshot → toggling mid-run can load nothing.
- `[-1,1]` contrast stretch applied to all indices (SAVI/EVI/MSAVI ranges differ) — cosmetic.
- Scratch temp dir is created before some early-return guards → leaves an empty temp folder on abort.
- `icon.png` is a 1×1 placeholder (invisible toolbar/menu icon) — replace before store submission.
- Downloader uses scoped PyQt5 enums (`QLineEdit.EchoMode.Password`) that need PyQt ≥5.11 but metadata declares `qgisMinimumVersion=3.0`.
- Public name `HyperSpectral Processer` reads as a typo for "Processor" (maintainer's chosen spelling — confirm).
- CI actions emit Node20→24 deprecation warnings (`checkout@v4`, `upload-artifact@v4`) — bump when convenient.

## Repo hygiene
- Three build-artifact zips are still git-tracked (`hyperspatial_bigdata.zip` at root, `hyperspatial_analytics.zip`, `hyperspatial_downloader.zip`) plus the dead monolith's `qgis-plugin/hyperspatial_bigdata/bin/macos/hyperspatial_cli` (121 KB Mach-O). The `.gitignore` pattern was added but the already-tracked files were never `git rm --cached`’d; the root zip isn't covered. The whole `hyperspatial_bigdata/` monolith directory is dead and can be removed.

## Recommended order before public launch
1. Commit the Indices GPU fix (done, verified).
2. Untrack the stray zips + monolith binary; optionally delete the dead monolith dir.
3. Refresh docs (summary.csv removed; 3-plugin split; PCA/SAM operations) + `AGENTS.md`.
4. Add vcpkg caching so the Windows zip actually builds; replace the placeholder icon.
5. Downloader: thread `search()`, expose/doc the cloud filter, reconcile the min-QGIS/enum mismatch.
6. Decide on the "Processer"/"Processor" name (it's the public store name).
