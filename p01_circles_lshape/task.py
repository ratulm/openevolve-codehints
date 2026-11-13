"""Problem 1: weighted circle packing in an L-shaped region (hidden from the blind arm)."""
import numpy as np

NAME = "p01_circles_lshape"
DESCRIPTION = (
    "Place 21 non-overlapping circles inside an L-shaped region: the unit square with the "
    "top-right block [0.6, 1] x [0.6, 1] removed. Every circle must lie fully inside the region. "
    "Maximise the sum over circles of radius_i * (0.2 + x_i)^2, where x_i is the x-coordinate of "
    "the circle's center, so circles further right are worth more."
)
TIME_LIMIT_S = 30
N = 21
A = 0.6
TOL = 1e-9
J_REF = 1.3039689836577324  # best of 160 SLSQP starts, see reference.py


def score(output):
    try:
        p, q = output
    except Exception:
        return 0.0, "run() must return a tuple (centers, radii)."
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != (N, 2) or q.shape != (N,):
        return 0.0, f"Expected centers of shape ({N}, 2) and radii of shape ({N},); got {p.shape} and {q.shape}."
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        return 0.0, "Centers or radii contain NaN or infinite values."
    if np.any(q < 0):
        return 0.0, f"Circle {int(np.argmin(q))} has a negative radius."
    x, y = p[:, 0], p[:, 1]
    wall = np.minimum.reduce([x, y, 1 - x, 1 - y]) - q
    if np.any(wall < -TOL):
        i = int(np.argmin(wall))
        return 0.0, f"Circle {i} (center ({x[i]:.4f}, {y[i]:.4f}), radius {q[i]:.4f}) extends outside the unit square."
    notch = np.hypot(np.maximum(A - x, 0), np.maximum(A - y, 0)) - q
    if np.any(notch < -TOL):
        i = int(np.argmin(notch))
        return 0.0, (f"Circle {i} (center ({x[i]:.4f}, {y[i]:.4f}), radius {q[i]:.4f}) overlaps the "
                     f"removed block [{A}, 1] x [{A}, 1].")
    i, j = np.triu_indices(N, 1)
    gap = np.hypot(x[i] - x[j], y[i] - y[j]) - q[i] - q[j]
    if np.any(gap < -TOL):
        k = int(np.argmin(gap))
        return 0.0, f"Circles {i[k]} and {j[k]} overlap by {-gap[k]:.2e}."
    return float(np.sum(q * (0.2 + x) ** 2) / J_REF), None
