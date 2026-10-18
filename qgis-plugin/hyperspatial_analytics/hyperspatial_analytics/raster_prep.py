# -*- coding: utf-8 -*-
"""QGIS-side helpers (osgeo.gdal) that turn a raw hyperspectral cube into the
GeoTIFF tiles the Sedona raw-cube jobs read via RS_FromGeoTiff:

  * detect_red_nir(path) / detect_bands(path, targets) -> pick bands nearest
    target wavelengths from per-band metadata (EnMAP/ENVI headers carry it).
  * prepare_band_tiles() -> extract a band subset into tile_r####_c####.tif
    GeoTIFFs (BSQ/ENVI or any other GDAL-readable cube).
  * prepare_red_nir_tile() -> the 2-band RED+NIR single tile used by Q4.

Metadata gotcha (why the tiles are "cleaned"): gdal.Translate copies the source
scale/offset (EnMAP: ENVI "data gain values" = 0.0001) into the GeoTIFF, and
GeoTools applies it when Sedona reads the file. The -32768 background then
arrives as -3.2768, so any nodata test in the Spark job silently stops matching.
The tiles therefore keep raw stored values: scale/offset and the nodata flag
are stripped, and the source nodata value is returned so the caller can pass it
to the job explicitly.

GDAL is QGIS's own (osgeo); this runs in the QGIS process, not the Spark venv.
"""
import os
import re
from osgeo import gdal

RED_NM, NIR_NM = 665.0, 842.0   # standard NDVI anchors

# Q5 K-Means band subset: one band per broad reflectance feature (same anchors as
# the C++ engine's BandResolver), chosen by wavelength so it works across sensors.
KMEANS_TARGETS = (("blue", 482.0), ("green", 560.0), ("red", 660.0),
                  ("nir", 860.0), ("swir1", 1650.0), ("swir2", 2200.0))
# Standard EnMAP bands for those anchors, used only for a 224-band cube that
# advertises no wavelengths.
ENMAP_DEFAULT_BANDS = {"blue": 14, "green": 30, "red": 47,
                       "nir": 75, "swir1": 153, "swir2": 193}
# A role is dropped when the nearest band is further than this from its target
# (e.g. no SWIR on a VNIR-only sensor).
MAX_WAVELENGTH_GAP_NM = 50.0


def _band_wavelengths(ds):
    """Return {1-based band: wavelength_nm}, or {} if none advertised."""
    wl = {}
    for b in range(1, ds.RasterCount + 1):
        v = (ds.GetRasterBand(b).GetMetadataItem("wavelength")
             or ds.GetRasterBand(b).GetMetadataItem("WAVELENGTH"))
        if v:
            m = re.findall(r"[-+0-9.eE]+", v)
            if m:
                wl[b] = float(m[0])
    if not wl:
        # ENVI header stores one 'wavelength = {w1, w2, ...}' array.
        s = (ds.GetMetadata("ENVI") or {}).get("wavelength")
        if s:
            for i, num in enumerate(re.findall(r"[-+0-9.eE]+", s), start=1):
                try:
                    wl[i] = float(num)
                except ValueError:
                    pass
    # Normalise micrometers -> nanometers if the values look like µm.
    if wl and max(wl.values()) < 100:
        wl = {k: v * 1000.0 for k, v in wl.items()}
    return wl


def detect_red_nir(path):
    """Best-effort (red_band, nir_band, n_bands), 1-based. Bands None if unknown."""
    ds = gdal.Open(path)
    if ds is None:
        return (None, None, 0)
    nb = ds.RasterCount
    wl = _band_wavelengths(ds)
    if not wl:
        ds = None
        return (None, None, nb)
    red = min(wl, key=lambda b: abs(wl[b] - RED_NM))
    nir = min(wl, key=lambda b: abs(wl[b] - NIR_NM))
    ds = None
    return (red, nir, nb)


def detect_bands(path, targets=KMEANS_TARGETS):
    """Return {role: 1-based band} for each (role, target_nm) in `targets`.

    Ordered like `targets`. A role whose nearest band is more than
    MAX_WAVELENGTH_GAP_NM away is left out. A 224-band cube without wavelength
    metadata falls back to ENMAP_DEFAULT_BANDS; any other cube without
    wavelengths returns {}.
    """
    ds = gdal.Open(path)
    if ds is None:
        return {}
    nb = ds.RasterCount
    wl = _band_wavelengths(ds)
    ds = None
    if not wl:
        if nb == 224:
            return {role: ENMAP_DEFAULT_BANDS[role] for role, _ in targets
                    if role in ENMAP_DEFAULT_BANDS}
        return {}
    out = {}
    for role, target in targets:
        band = min(wl, key=lambda b: abs(wl[b] - target))
        if abs(wl[band] - target) <= MAX_WAVELENGTH_GAP_NM:
            out[role] = band
    return out


def prepare_band_tiles(src_path, out_dir, bands, tile_size=256, output_type=None):
    """Extract `bands` (1-based) of `src_path` into tile_r####_c####.tif tiles.

    tile_size <= 0 writes the whole scene as one tile. Tiles keep the source
    stored values; scale/offset and the nodata flag are stripped (see module
    docstring). Returns a dict:
        tiles  : list of written paths
        nodata : the source nodata value (float) or None
        scale  : source scale factor (stored value -> physical), default 1.0
        offset : source offset, default 0.0 (non-zero is not compensated)
    """
    ds = gdal.Open(src_path)
    if ds is None:
        raise RuntimeError(f"GDAL cannot open {src_path}")
    first = ds.GetRasterBand(int(bands[0]))
    nodata = first.GetNoDataValue()
    scale = first.GetScale() or 1.0
    offset = first.GetOffset() or 0.0

    os.makedirs(out_dir, exist_ok=True)
    width, height = ds.RasterXSize, ds.RasterYSize
    step = int(tile_size) if tile_size and tile_size > 0 else max(width, height)
    opts = dict(bandList=[int(b) for b in bands], format="GTiff",
                creationOptions=["COMPRESS=LZW", "TILED=YES"])
    if output_type is not None:
        opts["outputType"] = output_type

    tiles = []
    for row0 in range(0, height, step):
        for col0 in range(0, width, step):
            dst = os.path.join(out_dir, f"tile_r{row0 // step:04d}_c{col0 // step:04d}.tif")
            gdal.Translate(dst, ds, srcWin=[col0, row0, min(step, width - col0),
                                            min(step, height - row0)], **opts)
            out = gdal.Open(dst, gdal.GA_Update)
            for b in range(1, out.RasterCount + 1):
                band = out.GetRasterBand(b)
                band.DeleteNoDataValue()
                band.SetScale(1.0)
                band.SetOffset(0.0)
            out = None
            tiles.append(dst)
    ds = None
    return {"tiles": tiles, "nodata": nodata, "scale": float(scale),
            "offset": float(offset)}


def extract_index_tiles(tiles_dir, out_dir, n_index=7):
    """Copy only the trailing `n_index` spectral-index bands of every engine tile.

    The engine's tile_r*.tif hold 224 raw + 7 index bands (~242 MB per 512-px
    tile decoded); Q1-Q3 read only the index bands, but RS_FromGeoTiff decodes
    a whole tile per row, so the full tiles exhaust the Spark driver heap even
    at 6 GB. The 7-band copies (~3.5 MB each for EnMAP) keep the band ORDER, so
    the jobs' "offset from RS_NumBands" index addressing is unchanged. Names,
    georeferencing and the 0.0 nodata are preserved. Returns the written paths.
    """
    import glob as _glob
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for src in sorted(_glob.glob(os.path.join(tiles_dir, "tile_r*.tif"))):
        ds = gdal.Open(src)
        if ds is None:
            continue
        n = ds.RasterCount
        if n < n_index:
            raise RuntimeError(f"{os.path.basename(src)} has {n} bands, expected the engine's "
                               f"raw + {n_index} index bands")
        dst = os.path.join(out_dir, os.path.basename(src))
        gdal.Translate(dst, ds, bandList=list(range(n - n_index + 1, n + 1)), format="GTiff",
                       creationOptions=["COMPRESS=LZW", "TILED=YES"])
        ds = None
        written.append(dst)
    return written


def prepare_red_nir_tile(src_path, out_dir, red_band, nir_band):
    """Extract RED,NIR into out_dir/tile_r0000_c0000.tif (2-band float GeoTIFF).

    Returns the prepare_band_tiles() dict. Pass its "nodata" to q_raw_ndvi.py
    (--nodata) so background pixels become the job's sentinel instead of NDVI 0.
    """
    return prepare_band_tiles(src_path, out_dir, [red_band, nir_band],
                              tile_size=0, output_type=gdal.GDT_Float32)
