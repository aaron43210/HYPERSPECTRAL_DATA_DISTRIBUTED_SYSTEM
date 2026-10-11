# -*- coding: utf-8 -*-
"""QGIS-side helpers (osgeo.gdal) to run the raw-cube NDVI job without the C++
engine. Two jobs:

  * detect_red_nir(path)  -> pick the RED (~665 nm) and NIR (~842 nm) bands from
    per-band wavelength metadata (EnMAP/ENVI headers carry it).
  * prepare_red_nir_tile() -> extract those two bands into a GeoTIFF named
    tile_r0000_c0000.tif that the Sedona job reads via RS_FromGeoTiff. Handles
    BSQ/ENVI and any other GDAL-readable cube.

GDAL is QGIS's own (osgeo); this runs in the GUI process, not the Spark venv.
"""
import os
import re
from osgeo import gdal

RED_NM, NIR_NM = 665.0, 842.0   # standard NDVI anchors


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


def prepare_red_nir_tile(src_path, out_dir, red_band, nir_band):
    """Extract RED,NIR into out_dir/tile_r0000_c0000.tif (2-band float GeoTIFF).

    Returns the written path. Any inherited nodata is cleared so Sedona map
    algebra evaluates NDVI on every pixel (the job maps nir+red==0 to its own
    sentinel instead).
    """
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "tile_r0000_c0000.tif")
    gdal.Translate(dst, src_path, bandList=[int(red_band), int(nir_band)],
                   format="GTiff", outputType=gdal.GDT_Float32)
    ds = gdal.Open(dst, gdal.GA_Update)
    for b in range(1, ds.RasterCount + 1):
        with __import__("contextlib").suppress(Exception):
            ds.GetRasterBand(b).DeleteNoDataValue()
    ds = None
    return dst
