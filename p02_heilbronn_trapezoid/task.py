"""Problem 2: point placement in a trapezoid scored by a mix of the smallest and the
median triangle area (hidden from the blind arm)."""
import itertools

import numpy as np

NAME = "p02_heilbronn_trapezoid"
DESCRIPTION = (
    "Place 13 points inside (or on the boundary of) the trapezoid with corners (0, 0), (1, 0), "
    "(0.9, 0.5) and (0.3, 0.5), i.e. 0 <= y <= 0.5 and 0.6*y <= x <= 1 - 0.2*y (area 0.4). "
    "Consider the areas of all 286 triangles formed by triples of the points. Maximise "
    "(smallest area / 0.4)^0.25 * (median area / 0.4)^0.75, a weighted geometric mean of the "
    "smallest triangle area and the median triangle area, both normalised by the trapezoid's "
    "area. It is not the plain Heilbronn objective (smallest triangle area alone): the median "
    "triangle carries three times the weight of the smallest one, but any zero-area triple "
    "(coincident or collinear points) makes the score zero."
)
TIME_LIMIT_S = 30
N = 13
H = 0.5      # height of the trapezoid
SL = 0.6     # left edge:  x >= SL * y
SR = 0.2     # right edge: x <= 1 - SR * y
AREA = 0.4
W_MIN = 0.25
W_MED = 0.75
TOL = 1e-9
J_REF = 0.10736699395018234  # best of 16 anneal+SLSQP runs then 8 basin-hopping runs, see reference.py
_I, _J, _K = np.array(list(itertools.combinations(range(N), 3))).T


def triangle_areas(p):
    a, b, c = p[_I], p[_J], p[_K]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def score(output):
    try:
        p = np.asarray(output, dtype=float)
    except Exception:
        return 0.0, f"run() must return a numeric array of shape ({N}, 2) holding the point coordinates."
    try:
        if p.shape != (N, 2):
            return 0.0, f"Expected an array of {N} points with shape ({N}, 2); got shape {p.shape}."
        if not np.all(np.isfinite(p)):
            i = int(np.argmin(np.all(np.isfinite(p), axis=1)))
            return 0.0, f"Point {i} contains NaN or infinite coordinates."
        x, y = p[:, 0], p[:, 1]
        slack = np.c_[y, H - y, x - SL * y, 1.0 - SR * y - x]
        if np.any(slack < -TOL):
            i, e = np.unravel_index(int(np.argmin(slack)), slack.shape)
            rule = ["y >= 0 (bottom side)", f"y <= {H} (top side)",
                    f"x >= {SL}*y (left slanted side)", f"x <= 1 - {SR}*y (right slanted side)"][e]
            return 0.0, (f"Point {i} at ({x[i]:.6f}, {y[i]:.6f}) is outside the trapezoid with corners "
                         f"(0, 0), (1, 0), (0.9, 0.5), (0.3, 0.5): it violates {rule} by {-slack[i, e]:.3e} "
                         f"(tolerance {TOL:g}).")
        areas = triangle_areas(p)
        smallest = float(areas.min()) / AREA
        median = float(np.median(areas)) / AREA
        return float(smallest ** W_MIN * median ** W_MED / J_REF), None
    except Exception as exc:
        return 0.0, f"The returned points could not be scored: {exc}"
