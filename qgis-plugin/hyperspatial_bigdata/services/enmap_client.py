# -*- coding: utf-8 -*-
"""
EnMAP data access client (M1).

Handles:
  - STAC catalogue search (DLR EnMAP STAC endpoint)
  - Scene download (streaming, with progress callback)
  - Metadata extraction (wavelengths, band indices, CRS, nodata)

Design rules:
  - Zero third-party imports beyond stdlib + optional GDAL/numpy.
  - Can run standalone (no QGIS) for unit testing.
  - When run inside QGIS, QgsBlockingNetworkRequest is preferred but
    the module falls back to urllib.request automatically.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import ssl
from pathlib import Path
from typing import Callable, Optional

# Use a VERIFIED TLS context for all outbound requests. The previous code used
# ssl._create_unverified_context(), which disabled certificate validation for
# every request in this module — including the scene download that sends the
# user's DLR EOWEB bearer token in an Authorization header. An unverified
# connection lets a man-in-the-middle intercept that token. macOS Python builds
# often fail verification only because they ship no root CA bundle, so prefer
# certifi's bundle and fall back to the (still verified) system default context.
def _make_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


ssl_context = _make_ssl_context()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAC_URL = (
    "https://geoservice.dlr.de/eoc/ogc/stac/v1"
    "/collections/ENMAP_HSI_L2A/items"
)

# Approximate EnMAP band centre wavelengths (nm) for 224 bands.
# Used only when the XML sidecar is unavailable.
_ENMAP_DEFAULT_WAVELENGTHS = [
    420.0 + i * (2450.0 - 420.0) / 223 for i in range(224)
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_scenes(
    bbox: list,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    limit: int = 20,
    token: Optional[str] = None,
) -> list[dict]:
    """
    Search EnMAP STAC catalogue.

    Parameters
    ----------
    bbox : [west, south, east, north] in EPSG:4326
    start_date, end_date : ISO-8601 date strings, e.g. "2023-01-01"
    max_cloud_cover : percentage 0–100; scenes above this are excluded
    limit : max results to return
    token : DLR EOWEB bearer token (optional for public search)

    Returns
    -------
    List of dicts::

        {
          "id":           str,
          "date":         str,   # YYYY-MM-DD
          "cloud_cover":  float,
          "bbox":         [w, s, e, n],
          "download_url": str,
        }
    """
    params = {
        "bbox":     f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datetime": f"{start_date}/{end_date}",
        "limit":    str(max(limit, 1)),
    }
    url = STAC_URL + "?" + urllib.parse.urlencode(params)
    raw = _http_get(url, token=token)
    data = json.loads(raw)

    scenes: list[dict] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        cloud = float(props.get("eo:cloud_cover", 0.0))
        if max_cloud_cover > 0 and cloud > max_cloud_cover:
            continue
        assets = feature.get("assets", {})
        download_url = (
            assets.get("data", {}).get("href", "")
            or assets.get("SPECTRAL_IMAGE", {}).get("href", "")
            or assets.get("image", {}).get("href", "")
        )
        scenes.append({
            "id":           feature["id"],
            "date":         str(props.get("datetime", ""))[:10],
            "cloud_cover":  cloud,
            "bbox":         feature.get("bbox", []),
            "download_url": download_url,
        })
    return scenes


def download_scene(
    scene_id: str,
    download_url: str,
    output_dir: str,
    token: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Download an EnMAP scene to ``output_dir/<scene_id>/``.

    Streams the file in 1 MB chunks so RAM usage is constant regardless
    of scene size.

    Parameters
    ----------
    progress_callback : callable(bytes_downloaded, total_bytes)
        Called after each chunk. ``total_bytes`` may be 0 if Content-Length
        is not provided by the server.

    Returns
    -------
    Absolute path to the downloaded .tif file.
    """
    scene_dir = Path(output_dir) / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    dest = scene_dir / f"{scene_id}.tif"

    req = urllib.request.Request(download_url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=300, context=ssl_context) as resp:
        # Prevent silent redirect to the EOC UMS HTML login page when token is invalid/missing
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise Exception("Authentication Failed: The DLR Geoservice redirected to the EOC UMS login page. "
                            "Please provide a valid EOWEB Bearer token.")

        total = int(resp.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        chunk_size = 1 << 20  # 1 MB

        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if progress_callback:
                    progress_callback(downloaded, total)

    return str(dest)


def extract_metadata(scene_dir: str) -> dict:
    """
    Build and persist ``metadata.json`` for a downloaded EnMAP scene.

    Reads:
      - The scene GeoTIFF (via GDAL) for band count, CRS, extent, nodata
      - The XML sidecar for per-band wavelengths (falls back to defaults)

    Computes NIR / Red / Green / SWIR band indices from wavelengths so the
    C++ binary does not need to hard-code sensor-specific values.

    Returns
    -------
    dict  (also written as ``<scene_dir>/metadata.json``)
    """
    try:
        from osgeo import gdal, osr
    except ImportError as e:
        raise ImportError(
            "GDAL Python bindings required for metadata extraction. "
            "Install with: pip install GDAL"
        ) from e

    scene_path = Path(scene_dir)
    tif_path = next(scene_path.glob("*.tif"), None)
    if tif_path is None:
        raise FileNotFoundError(f"No .tif file found in {scene_dir}")

    xml_path = (
        next(scene_path.glob("*METADATA*.xml"), None)
        or next(scene_path.glob("*metadata*.xml"), None)
    )

    # ---- GDAL scene info ---------------------------------------------------
    ds = gdal.Open(str(tif_path))
    if ds is None:
        raise RuntimeError(f"GDAL cannot open {tif_path}")

    gt = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    band_count = ds.RasterCount

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * width
    ymin = ymax + gt[5] * height

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    epsg = srs.GetAttrValue("AUTHORITY", 1)
    crs = f"EPSG:{epsg}" if epsg else ds.GetProjectionRef() or "EPSG:4326"

    rb = ds.GetRasterBand(1)
    nodata_val, has_nodata = rb.GetNoDataValue(), rb.GetNoDataValue() is not None
    nodata = float(nodata_val) if has_nodata else -9999.0
    ds = None  # close

    # ---- Wavelengths -------------------------------------------------------
    wavelengths = _parse_wavelengths(xml_path, band_count)

    def closest(target_nm: float) -> int:
        """Return 0-based band index closest to target_nm."""
        return min(range(len(wavelengths)),
                   key=lambda i: abs(wavelengths[i] - target_nm))

    meta = {
        "scene_id":             tif_path.stem,
        "sensor":               "EnMAP",
        "processing_level":     "L2A",
        "bands":                band_count,
        "wavelength_min_nm":    float(min(wavelengths)),
        "wavelength_max_nm":    float(max(wavelengths)),
        "wavelengths_nm":       [float(w) for w in wavelengths],
        # 0-based indices; pass as (index + 1) to GDAL's 1-based API
        "nir_band_index":       closest(860.0),
        "red_band_index":       closest(660.0),
        "green_band_index":     closest(560.0),
        "swir_band_index":      closest(1640.0),
        "spatial_resolution_m": 30,
        "crs":                  crs,
        "bbox":                 [xmin, ymin, xmax, ymax],
        "nodata_value":         nodata,
        "file_size_bytes":      tif_path.stat().st_size,
        "local_path":           str(tif_path),
    }

    out_path = scene_path / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, token: Optional[str] = None) -> bytes:
    """Simple HTTP GET; raises urllib.error.URLError on failure."""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30, context=ssl_context) as resp:
        return resp.read()


def _parse_wavelengths(xml_path: Optional[Path], band_count: int) -> list[float]:
    """
    Extract per-band centre wavelengths from the EnMAP XML sidecar.

    Falls back to linearly spaced values (420–2450 nm) if the XML is
    absent or cannot be parsed.
    """
    if xml_path is not None and xml_path.exists():
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            wavelengths: list[float] = []

            for elem in root.iter():
                tag = elem.tag.lower()
                # EnMAP XML uses various tag names across versions
                if any(k in tag for k in ("wavelength", "centre_wavelength",
                                           "center_wavelength")):
                    text = (elem.text or "").strip()
                    try:
                        wavelengths.append(float(text))
                    except ValueError:
                        continue

            if len(wavelengths) == band_count:
                return wavelengths
        except ET.ParseError:
            pass  # fall through to default

    # Default: linear spread across EnMAP spectral range
    if band_count == 224:
        return list(_ENMAP_DEFAULT_WAVELENGTHS)
    return [420.0 + i * (2450.0 - 420.0) / max(band_count - 1, 1)
            for i in range(band_count)]


def load_metadata(scene_dir: str) -> dict:
    """Load existing metadata.json from a scene directory."""
    meta_path = Path(scene_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {scene_dir}")
    with open(meta_path) as f:
        return json.load(f)
