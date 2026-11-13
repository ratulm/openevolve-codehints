"""Blinded evaluator for the hints study.

Deliberately reveals nothing about the task: it reports a single
`combined_score` and no error text, metric names or artifacts.

Hidden task: 21 non-overlapping discs inside the unit square with the block
[0.6, 1] x [0.6, 1] removed; score = sum_i r_i * (0.2 + x_i)^2 / J_REF.
J_REF is the best value found by reference_solver.py (160 SLSQP starts).
"""
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

N = 21
A = 0.6
TOL = 1e-9
J_REF = 1.3039689836577324
TIME_LIMIT_S = 30

_RUNNER = """
import importlib.util, pickle, sys
spec = importlib.util.spec_from_file_location("program", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(sys.argv[2], "wb") as fh:
    pickle.dump(m.run(), fh)
"""


def _score(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != (N, 2) or q.shape != (N,):
        return 0.0
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))) or np.any(q < 0):
        return 0.0
    x, y = p[:, 0], p[:, 1]
    notch = np.hypot(np.maximum(A - x, 0), np.maximum(A - y, 0))
    if np.any(np.minimum.reduce([x, y, 1 - x, 1 - y, notch]) < q - TOL):
        return 0.0
    i, j = np.triu_indices(N, 1)
    if np.any(np.hypot(x[i] - x[j], y[i] - y[j]) < q[i] + q[j] - TOL):
        return 0.0
    return float(np.sum(q * (0.2 + x) ** 2) / J_REF)


def evaluate(program_path):
    out = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False).name
    try:
        subprocess.run(
            [sys.executable, "-c", _RUNNER, program_path, out],
            capture_output=True,
            timeout=TIME_LIMIT_S,
            check=True,
        )
        with open(out, "rb") as fh:
            p, q = pickle.load(fh)
        return {"combined_score": _score(p, q)}
    except Exception:
        return {"combined_score": 0.0}
    finally:
        if os.path.exists(out):
            os.unlink(out)
