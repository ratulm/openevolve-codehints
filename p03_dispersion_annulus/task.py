"""Problem 3: point dispersion in an annulus under an anisotropic, position-dependent distance
(hidden from the blind arm)."""
import numpy as np

NAME = "p03_dispersion_annulus"
DESCRIPTION = (
    "Place 18 points in the annulus 0.4 <= sqrt(x^2 + y^2) <= 1.0 (the unit disc centred at the "
    "origin with a concentric hole of radius 0.4) so as to maximise the minimum pairwise distance, "
    "where the distance between points (x1, y1) and (x2, y2) is not Euclidean but "
    "sqrt((3 * (x1 - x2))^2 + (y1 - y2)^2) * (1 + 0.4 * (y1 + y2) / 2): horizontal separation "
    "counts three times as much as vertical separation, and every distance is scaled by a factor "
    "that grows linearly with the pair's mean y-coordinate (0.6 at the bottom of the disc, 1.4 at "
    "the top). The score is the smallest of the 153 pairwise distances divided by a reference value."
)
TIME_LIMIT_S = 30
N = 18
R_IN = 0.4
R_OUT = 1.0
KX = 3.0
GY = 0.4
TOL = 1e-9
D_REF = 0.9313670214426615  # best of 1000 SLSQP starts (4 seeds, all agree), see reference.py


def score(output):
    try:
        p = np.asarray(output, dtype=float)
    except Exception:
        return 0.0, f"run() must return a numeric array of shape ({N}, 2) holding the point coordinates."
    if p.shape != (N, 2):
        return 0.0, f"Expected an array of {N} points with shape ({N}, 2); got shape {p.shape}."
    if not np.all(np.isfinite(p)):
        k = int(np.argmax(~np.isfinite(p).all(axis=1)))
        return 0.0, f"Point {k} has a NaN or infinite coordinate."
    x, y = p[:, 0], p[:, 1]
    r = np.hypot(x, y)
    if np.any(r < R_IN - TOL):
        k = int(np.argmin(r))
        return 0.0, (f"Point {k} at ({x[k]:.6f}, {y[k]:.6f}) is inside the hole: its distance from the "
                     f"origin is {r[k]:.6f}, below the inner radius {R_IN}.")
    if np.any(r > R_OUT + TOL):
        k = int(np.argmax(r))
        return 0.0, (f"Point {k} at ({x[k]:.6f}, {y[k]:.6f}) is outside the disc: its distance from the "
                     f"origin is {r[k]:.6f}, above the outer radius {R_OUT}.")
    i, j = np.triu_indices(N, 1)
    d = np.sqrt((KX * (x[i] - x[j])) ** 2 + (y[i] - y[j]) ** 2) * (1.0 + GY * 0.5 * (y[i] + y[j]))
    return float(np.min(d) / D_REF), None
