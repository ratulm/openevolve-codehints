# EVOLVE-BLOCK-START
# Goal: choose one point in the box [-4, 4]^3 for the function `landscape`
# defined in the fixed part below (a constant 13 minus five wells of different
# depths and widths, plus a small ripple, a weak bowl and a steep wall outside
# the box).
#
# Score: the point is NOT judged by landscape(point), and the lowest function
# value is NOT the target. The evaluator computes a robust value
#     R(point) = mean of landscape(point + offset) over 27 fixed offsets,
# where the offsets are the 3x3x3 grid {-0.5, 0, 0.5}^3 (the point itself is
# one of the 27). The score is R_ref / R(point) with R_ref = 6.2517, so 1.0
# means matching the reference and lower R is better (above 1.0 is possible).
# A wide, moderately low basin therefore scores better than a narrow, very
# deep well: the global minimum of `landscape` has R of about 8.6 and scores
# only about 0.73.
#
# Validity: run() must return exactly 3 finite numbers, each inside [-4, 4]
# (tolerance 1e-9). Anything else, an exception, or a run longer than
# 30 seconds scores 0.
import numpy as np


_G = np.array([-0.5, 0.0, 0.5])
_OFFSETS = np.array([[a, b, c] for a in _G for b in _G for c in _G])


def _landscape_many(p):
    # Vectorized copy of `landscape` for an (N, 3) array of points.
    d2 = np.sum((p[:, None, :] - well_centers[None]) ** 2, axis=2)
    wells = np.sum(well_depths * np.exp(-d2 / (2.0 * well_widths ** 2)), axis=1)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ripple = (0.5 * np.sin(2.3 * x + 1.1 * y) * np.cos(1.7 * z - 0.6 * x)
              + 0.3 * np.cos(2.9 * y - 1.3 * z))
    bowl = 0.02 * np.sum(p ** 2, axis=1)
    wall = 50.0 * np.sum(np.maximum(np.abs(p) - 4.0, 0.0), axis=1)
    return 13.0 - wells + ripple + bowl + wall


def _robust_many(pts):
    # Robust value R (mean over the 27 offsets) for each of the (N, 3) points.
    shifted = (pts[:, None, :] + _OFFSETS[None]).reshape(-1, 3)
    return _landscape_many(shifted).reshape(len(pts), 27).mean(axis=1)


def choose_point():
    # Coarse grid search on the robust objective, then compass-search refinement
    # of the best few candidates, clipped to the box.
    g = np.linspace(-4.0, 4.0, 41)
    grid = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    vals = np.concatenate([_robust_many(c) for c in np.array_split(grid, 20)])
    starts = grid[np.argsort(vals)[:10]]

    dirs = np.vstack([np.eye(3), -np.eye(3),
                      np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1)
                                for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]) / np.sqrt(3)])
    best_x, best_v = None, np.inf
    for x in starts:
        v = _robust_many(x[None])[0]
        step = 0.2
        while step > 1e-9:
            cand = np.clip(x + step * dirs, -4.0, 4.0)
            cv = _robust_many(cand)
            i = np.argmin(cv)
            if cv[i] < v:
                x, v = cand[i], cv[i]
            else:
                step *= 0.5
        if v < best_v:
            best_x, best_v = x, v
    return [float(c) for c in np.clip(best_x, -4.0, 4.0)]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
import numpy as np

# Centers, depths and widths of the five wells of the landscape.
well_centers = np.array([
    [1.8, -1.6, 1.2],
    [-2.0, 1.5, -1.0],
    [-1.85, 1.4, -0.9],
    [2.5, 2.4, -2.6],
    [-2.6, -2.4, 2.2],
])
well_depths = np.array([9.0, 6.5, 4.5, 6.0, 4.0])
well_widths = np.array([1.0, 0.8, 0.15, 0.3, 0.8])


def landscape(point):
    # The function of 3 variables whose neighbourhood average is scored.
    point = np.asarray(point, dtype=float)
    squared_distances = np.sum((point - well_centers) ** 2, axis=1)
    wells = np.sum(well_depths * np.exp(-squared_distances / (2.0 * well_widths ** 2)))
    ripple = (0.5 * np.sin(2.3 * point[0] + 1.1 * point[1]) * np.cos(1.7 * point[2] - 0.6 * point[0])
              + 0.3 * np.cos(2.9 * point[1] - 1.3 * point[2]))
    bowl = 0.02 * np.sum(point ** 2)
    # Steep penalty for leaving the box [-4, 4]^3.
    wall = 50.0 * np.sum(np.maximum(np.abs(point) - 4.0, 0.0))
    return 13.0 - wells + ripple + bowl + wall


def run():
    # Return the chosen point; the evaluator does the scoring.
    return choose_point()
