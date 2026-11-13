"""Informative evaluator for the hinted arm: same scoring as evaluator.py,
but failures come back with an error message explaining what went wrong."""
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluator as blind
from openevolve.evaluation_result import EvaluationResult

N, A, TOL = blind.N, blind.A, blind.TOL


def _explain(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != (N, 2) or q.shape != (N,):
        return f"Expected centers of shape ({N}, 2) and radii of shape ({N},); got {p.shape} and {q.shape}."
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        return "Centers or radii contain NaN or infinite values."
    if np.any(q < 0):
        return f"Circle {int(np.argmin(q))} has a negative radius."
    x, y = p[:, 0], p[:, 1]
    wall = np.minimum.reduce([x, y, 1 - x, 1 - y]) - q
    if np.any(wall < -TOL):
        i = int(np.argmin(wall))
        return f"Circle {i} (center ({x[i]:.4f}, {y[i]:.4f}), radius {q[i]:.4f}) extends outside the unit square."
    notch = np.hypot(np.maximum(A - x, 0), np.maximum(A - y, 0)) - q
    if np.any(notch < -TOL):
        i = int(np.argmin(notch))
        return (f"Circle {i} (center ({x[i]:.4f}, {y[i]:.4f}), radius {q[i]:.4f}) overlaps the removed "
                f"block [{A}, 1] x [{A}, 1].")
    i, j = np.triu_indices(N, 1)
    gap = np.hypot(x[i] - x[j], y[i] - y[j]) - q[i] - q[j]
    if np.any(gap < -TOL):
        k = int(np.argmin(gap))
        return f"Circles {i[k]} and {j[k]} overlap by {-gap[k]:.2e}."
    return None


def _fail(message):
    return EvaluationResult(metrics={"combined_score": 0.0}, artifacts={"error": message})


def evaluate(program_path):
    out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False).name
    try:
        proc = subprocess.run(
            [sys.executable, "-c", blind._RUNNER, program_path, out],
            capture_output=True, text=True, timeout=blind.TIME_LIMIT_S,
        )
        if proc.returncode != 0:
            return _fail("The program raised an exception:\n" + proc.stderr[-1500:])
        with open(out, "rb") as fh:
            result = pickle.load(fh)
        try:
            p, q = result
        except Exception:
            return _fail("run() must return a tuple (centers, radii).")
        message = _explain(p, q)
        if message:
            return _fail("Invalid packing, scored 0. " + message)
        return EvaluationResult(metrics={"combined_score": blind._score(p, q)})
    except subprocess.TimeoutExpired:
        return _fail(f"The program ran longer than the {blind.TIME_LIMIT_S} second limit.")
    except Exception as exc:
        return _fail(f"Evaluation failed: {exc}")
    finally:
        if os.path.exists(out):
            os.unlink(out)
