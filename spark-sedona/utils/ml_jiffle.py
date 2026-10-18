"""
Spark MLlib model -> Sedona Jiffle map-algebra compiler (used by Q5 K-Means).

The model is trained on a pixel sample in Spark, then applied to EVERY pixel
inside the JVM by compiling it into one Jiffle script for RS_MapAlgebra, so no
raster ever crosses into Python (Sedona 1.5.1 cannot deserialize RasterType in
Python). The feature transform therefore exists twice - once as Spark columns
(training) and once as Jiffle (mapping). Both are generated here from the same
constants so they cannot drift apart.

Feature modes (x = the pixel's band vector, ||x|| its L2 norm):
  raw               x                         Euclidean on stored values
  shape             x / ||x||                 spectral shape, SAM-like:
                                              insensitive to illumination
  shape_brightness  [x / ||x||, (ln||x|| - 7) / 3]
                                              shape + one log-brightness
                                              feature, so dark water and bright
                                              snow (similar shape) separate

Jiffle rules learned the hard way (Sedona 1.5.1 / jt-jiffle):
  * every constant is written in fixed-point and parenthesised: a negative
    centre would otherwise produce 'x--3.2', and exponent notation such as
    '1e-05' is not safe to rely on;
  * every ternary is single-level, which keeps the script easy to verify
    (see tests/test_ml_jiffle.py, which executes it in Python).

Pure Python: importable without Spark (pyspark is imported lazily).
"""
from __future__ import annotations

FEATURE_MODES = ("shape_brightness", "shape", "raw")
BRIGHTNESS_SHIFT = 7.0     # ln||x|| is ~6..10 for EnMAP DN (x10000 reflectance)
BRIGHTNESS_SCALE = 3.0     # -> brightness feature spans roughly -0.3..1.0
NODATA_OUT = 255           # class-map nodata (pixel type 'B' = unsigned byte)


def _lit(value: float) -> str:
    """A Jiffle-safe numeric literal: fixed-point and parenthesised."""
    return f"({float(value):.12f})"


def feature_count(n_bands: int, mode: str) -> int:
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode {mode!r}; expected one of {FEATURE_MODES}")
    return n_bands + 1 if mode == "shape_brightness" else n_bands


# ----------------------------------------------------------------------------
# Spark side (training)
# ----------------------------------------------------------------------------
def valid_pixel_condition(cols, nodata=None):
    """Spark Column: pixel is valid (no band equals nodata, norm > 0)."""
    from pyspark.sql import functions as F
    cond = sum((F.col(c) * F.col(c) for c in cols), F.lit(0.0)) > 0
    if nodata is not None:
        for c in cols:
            cond = cond & (F.col(c) != F.lit(float(nodata)))
    return cond


def add_spark_features(df, cols, mode, out_col="features"):
    """Append the feature vector column `out_col` for `mode` to `df`."""
    from pyspark.sql import functions as F
    from pyspark.ml.feature import VectorAssembler
    feature_count(len(cols), mode)      # validates mode
    if mode == "raw":
        inputs = list(cols)
    else:
        df = df.withColumn("_n", F.sqrt(sum((F.col(c) * F.col(c) for c in cols), F.lit(0.0))))
        inputs = []
        for c in cols:
            df = df.withColumn(f"_u_{c}", F.col(c) / F.col("_n"))
            inputs.append(f"_u_{c}")
        if mode == "shape_brightness":
            df = df.withColumn("_br", (F.log(F.col("_n")) - BRIGHTNESS_SHIFT) / BRIGHTNESS_SCALE)
            inputs.append("_br")
    return VectorAssembler(inputCols=inputs, outputCol=out_col).transform(df)


# ----------------------------------------------------------------------------
# Jiffle side (mapping every pixel)
# ----------------------------------------------------------------------------
def _jiffle_features(n_bands: int, mode: str):
    """Return (statements, feature_expressions) mirroring add_spark_features."""
    feature_count(n_bands, mode)
    stmts = ["n = sqrt(" + " + ".join(f"rast[{i}]*rast[{i}]" for i in range(n_bands)) + ");"]
    if mode == "raw":
        x = [f"rast[{i}]" for i in range(n_bands)]
    else:
        x = [f"(rast[{i}]/n)" for i in range(n_bands)]
        if mode == "shape_brightness":
            stmts.append(f"br = (log(n) - {_lit(BRIGHTNESS_SHIFT)}) / {_lit(BRIGHTNESS_SCALE)};")
            x.append("br")
    return stmts, x


def kmeans_to_jiffle(centres, n_bands, mode, nodata_in=None, nodata_out=NODATA_OUT):
    """Compile K-Means centres into a nearest-centre Jiffle script.

    Output pixel = index of the nearest centre (ties -> lowest index, as in
    MLlib), or `nodata_out` where any band equals `nodata_in` or the pixel is
    all-zero.
    """
    centres = [list(map(float, c)) for c in centres]
    if not centres:
        raise ValueError("no centres")
    if len(centres) > 254 and nodata_out == NODATA_OUT:
        raise ValueError("at most 254 clusters fit an unsigned-byte map with nodata 255")
    stmts, x = _jiffle_features(n_bands, mode)
    for k, c in enumerate(centres):
        if len(c) != len(x):
            raise ValueError(f"centre {k} has {len(c)} values, expected {len(x)} for mode {mode!r}")
        terms = [f"({xi}-{_lit(ci)})*({xi}-{_lit(ci)})" for xi, ci in zip(x, c)]
        stmts.append(f"d{k} = " + " + ".join(terms) + ";")
    stmts += ["best = 0;", "bd = d0;"]
    for k in range(1, len(centres)):
        stmts.append(f"best = d{k} < bd ? {k} : best;")
        stmts.append(f"bd = d{k} < bd ? d{k} : bd;")
    stmts.append("ok = n > 0 ? 1 : 0;")
    if nodata_in is not None:
        for i in range(n_bands):
            stmts.append(f"ok = rast[{i}] == {_lit(nodata_in)} ? 0 : ok;")
    stmts.append(f"out = ok == 1 ? best : {int(nodata_out)};")
    return " ".join(stmts)


# ----------------------------------------------------------------------------
# Cluster interpretation
# ----------------------------------------------------------------------------
def _nd(a: float, b: float) -> float:
    return (a - b) / (a + b) if (a + b) != 0 else 0.0


def cluster_indices(mean_reflectance, roles):
    """NDVI / NDWI / NDSI of a cluster's mean spectrum (None if a band is missing)."""
    r = dict(zip(roles, mean_reflectance))
    g, red, nir, s1 = (r.get(k) for k in ("green", "red", "nir", "swir1"))
    return {
        "ndvi": _nd(nir, red) if None not in (nir, red) else None,
        "ndwi": _nd(g, nir) if None not in (g, nir) else None,     # McFeeters 1996
        "ndsi": _nd(g, s1) if None not in (g, s1) else None,       # snow index
    }


def suggest_label(mean_reflectance, roles):
    """Heuristic land-cover label for a cluster's mean reflectance (0..1 units).

    Standard thresholds: NDSI > 0.4 with a bright green band is snow/ice (the
    MODIS snow rule); NDWI > 0 with dark NIR is open water; NDVI > 0.6 / > 0.3
    is dense / sparse vegetation. Checked in that order, because water also has
    a high NDSI and is told apart from snow by brightness.
    """
    idx = cluster_indices(mean_reflectance, roles)
    r = dict(zip(roles, mean_reflectance))
    if None in idx.values():
        return "Unlabelled"
    if idx["ndsi"] > 0.4 and r["green"] >= 0.2:
        return "Snow/ice"
    if idx["ndwi"] > 0.0 and r["nir"] < 0.15:
        return "Water"
    if idx["ndvi"] > 0.6:
        return "Dense vegetation"
    if idx["ndvi"] > 0.3:
        return "Sparse vegetation/grass"
    if sum(mean_reflectance) / len(mean_reflectance) < 0.05:
        return "Shadow/dark surface"
    return "Bare rock/soil"
