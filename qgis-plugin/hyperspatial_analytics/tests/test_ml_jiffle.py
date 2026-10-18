"""Unit tests for spark-sedona/utils/ml_jiffle.py (no Spark, no QGIS needed).

Run:  python -m unittest discover -s qgis-plugin/hyperspatial_analytics/tests

The generated Jiffle script is executed here by a tiny translator (every
ternary the generator emits is single-level, so `a = c ? x : y;` maps directly
to Python), then compared with an independent NumPy nearest-centre assignment.
"""
import importlib.util
import os
import re
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "spark-sedona", "utils",
                                        "ml_jiffle.py"))
_spec = importlib.util.spec_from_file_location("ml_jiffle", _MODULE)
mj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mj)


def _to_python(expr):
    expr = re.sub(r"rast\[(\d+)\]", r"r[\1]", expr)
    return expr.replace("sqrt(", "np.sqrt(").replace("log(", "np.log(")


def run_jiffle(script, pixel):
    """Execute a generated Jiffle script for one pixel; return `out`."""
    env = {}
    r = [np.float64(v) for v in pixel]
    with np.errstate(divide="ignore", invalid="ignore"):
        for stmt in filter(None, (s.strip() for s in script.split(";"))):
            lhs, rhs = (part.strip() for part in stmt.split("=", 1))
            if "?" in rhs:
                cond, rest = rhs.split("?", 1)
                yes, no = rest.split(":", 1)
                assert "?" not in rest, "generator must emit single-level ternaries"
                rhs = f"({_to_python(yes)}) if ({_to_python(cond)}) else ({_to_python(no)})"
            else:
                rhs = _to_python(rhs)
            env[lhs] = eval(rhs, {"np": np, "r": r}, env)
    return env["out"]


def numpy_features(X, mode):
    """Independent reference implementation of the three feature modes."""
    X = np.asarray(X, dtype=float)
    if mode == "raw":
        return X
    n = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / n
    if mode == "shape":
        return U
    return np.hstack([U, (np.log(n) - 7.0) / 3.0])


class JiffleCompilerTests(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def _pixels(self, n=300, bands=6):
        return self.rng.uniform(10, 9000, size=(n, bands))

    def test_matches_numpy_nearest_centre_all_modes(self):
        X = self._pixels()
        for mode in mj.FEATURE_MODES:
            F = numpy_features(X, mode)
            centres = F[self.rng.choice(len(F), 5, replace=False)] + 1e-3
            script = mj.kmeans_to_jiffle(centres, X.shape[1], mode, nodata_in=-32768)
            expected = np.argmin(((F[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
            got = np.array([run_jiffle(script, px) for px in X])
            np.testing.assert_array_equal(got, expected, err_msg=f"mode={mode}")

    def test_nodata_and_zero_pixels_map_to_255(self):
        centres = numpy_features(self._pixels(4), "shape_brightness")
        script = mj.kmeans_to_jiffle(centres, 6, "shape_brightness", nodata_in=-32768)
        self.assertEqual(run_jiffle(script, [-32768] * 6), 255)
        self.assertEqual(run_jiffle(script, [500, 600, -32768, 700, 800, 900]), 255)
        self.assertEqual(run_jiffle(script, [0] * 6), 255)

    def test_negative_and_tiny_constants_are_jiffle_safe(self):
        script = mj.kmeans_to_jiffle([[-3.2768, 1e-9], [2.5, -0.0000001]], 2, "raw",
                                     nodata_in=-32768)
        self.assertNotIn("--", script)
        self.assertIsNone(re.search(r"\de[-+]?\d", script), "no exponent notation")
        self.assertEqual(script.count("("), script.count(")"))

    def test_rejects_wrong_centre_length_and_mode(self):
        with self.assertRaises(ValueError):
            mj.kmeans_to_jiffle([[1.0] * 6], 6, "shape_brightness")   # needs 7 values
        with self.assertRaises(ValueError):
            mj.kmeans_to_jiffle([[1.0] * 6], 6, "cosine")
        with self.assertRaises(ValueError):
            mj.kmeans_to_jiffle([], 6, "raw")

    def test_ties_go_to_lowest_index_like_mllib(self):
        script = mj.kmeans_to_jiffle([[1.0, 1.0], [1.0, 1.0]], 2, "raw")
        self.assertEqual(run_jiffle(script, [1.0, 1.0]), 0)


class LabelTests(unittest.TestCase):
    ROLES = ["blue", "green", "red", "nir", "swir1", "swir2"]

    def label(self, *refl):
        return mj.suggest_label(list(refl), self.ROLES)

    def test_typical_spectra(self):
        self.assertEqual(self.label(0.75, 0.77, 0.75, 0.64, 0.07, 0.05), "Snow/ice")
        self.assertEqual(self.label(0.06, 0.08, 0.05, 0.03, 0.01, 0.01), "Water")
        self.assertEqual(self.label(0.02, 0.05, 0.03, 0.40, 0.20, 0.09), "Dense vegetation")
        self.assertEqual(self.label(0.05, 0.08, 0.07, 0.25, 0.25, 0.15), "Sparse vegetation/grass")
        self.assertEqual(self.label(0.18, 0.21, 0.21, 0.26, 0.25, 0.19), "Bare rock/soil")
        self.assertEqual(self.label(0.02, 0.02, 0.02, 0.03, 0.02, 0.01), "Shadow/dark surface")

    def test_missing_roles_are_unlabelled(self):
        self.assertEqual(mj.suggest_label([0.1, 0.2], ["red", "nir"]), "Unlabelled")


if __name__ == "__main__":
    unittest.main()
