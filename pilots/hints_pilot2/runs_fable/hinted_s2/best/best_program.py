# EVOLVE-BLOCK-START
# Goal: place 21 non-overlapping circles inside an L-shaped region, which is
# the unit square with the top-right block [0.6, 1] x [0.6, 1] removed.
#
# Score: the evaluator computes sum_i radius_i * (0.2 + x_i)**2 and divides it
# by a reference value of about 1.30, so 1.0 means matching the reference
# (higher is possible). Circles further right are worth much more: at equal
# radius, a circle at x = 0.9 is worth about 13 times one at x = 0.1. It is NOT
# the plain sum of radii and NOT the covered area.
#
# Validity: exactly 21 circles, every circle fully inside the region, no two
# circles overlapping (tolerance 1e-9). Any violation, an exception, or a run
# longer than 30 seconds scores 0.
#
# HINT: hand-placed layouts plateau well below the reference. A stronger
# approach is to treat this as a constrained optimization problem: the
# variables are the 21 centers and 21 radii, the objective is the weighted sum
# above, and the constraints are the wall, removed-block and pairwise
# non-overlap conditions. Solve it with scipy.optimize.minimize(method="SLSQP")
# from many different starting layouts, shrink the radii by about 1e-8
# afterwards so the result is strictly valid, and keep the best valid result.
import time

import numpy as np
from scipy.optimize import minimize

N = 21
IU, JU = np.triu_indices(N, 1)
ROWS = np.arange(len(IU))
AR = np.arange(N)


def _block_gap(x, y):
    return np.hypot(np.maximum(0.6 - x, 0.0), np.maximum(0.6 - y, 0.0))


def _repair(x, y, r):
    # Make any candidate strictly valid: clip to walls/block, fix overlaps.
    r = np.minimum(r, np.minimum.reduce([x, y, 1 - x, 1 - y, _block_gap(x, y)]))
    r = np.maximum(r, 0.0)
    d = np.hypot(x[IU] - x[JU], y[IU] - y[JU])
    for _ in range(3):
        for i, j, dd in zip(IU, JU, d):
            if r[i] + r[j] > dd:
                s = dd / (r[i] + r[j])
                r[i] *= s
                r[j] *= s
    return np.maximum(r - 1e-8, 0.0)


def _score(x, r):
    return float(np.sum(r * (0.2 + x) ** 2))


def _obj(v):
    x, r = v[:N], v[2 * N:]
    g = np.zeros(3 * N)
    g[:N] = -2 * r * (0.2 + x)
    g[2 * N:] = -(0.2 + x) ** 2
    return -np.sum(r * (0.2 + x) ** 2), g


def _con(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    a = np.maximum(0.6 - x, 0.0)
    b = np.maximum(0.6 - y, 0.0)
    return np.concatenate([
        (x[IU] - x[JU]) ** 2 + (y[IU] - y[JU]) ** 2 - (r[IU] + r[JU]) ** 2,
        x - r, 1 - x - r, y - r, 1 - y - r,
        a * a + b * b - r * r,
    ])


def _jac(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    M = len(IU)
    J = np.zeros((M + 5 * N, 3 * N))
    dx, dy, sr = 2 * (x[IU] - x[JU]), 2 * (y[IU] - y[JU]), -2 * (r[IU] + r[JU])
    J[ROWS, IU] = dx
    J[ROWS, JU] = -dx
    J[ROWS, N + IU] = dy
    J[ROWS, N + JU] = -dy
    J[ROWS, 2 * N + IU] = sr
    J[ROWS, 2 * N + JU] = sr
    for k, (col, sg) in enumerate([(0, 1.0), (0, -1.0), (N, 1.0), (N, -1.0)]):
        J[M + k * N + AR, col + AR] = sg
        J[M + k * N + AR, 2 * N + AR] = -1.0
    q = M + 4 * N + AR
    J[q, AR] = -2 * np.maximum(0.6 - x, 0.0)
    J[q, N + AR] = -2 * np.maximum(0.6 - y, 0.0)
    J[q, 2 * N + AR] = -2 * r
    return J


def f():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    # Safe fallback: the original small grid, always valid.
    grid = np.array([[0.06 + 0.12 * c, 0.06 + 0.12 * row]
                     for row in range(5) for c in range(5 if row < 3 else 3)])
    bx, by = grid[:, 0].copy(), grid[:, 1].copy()
    br = np.maximum(greedy_radii_without_overlap(grid) - 1e-8, 0.0)
    best = _score(bx, br)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = [dict(type="ineq", fun=_con, jac=_jac)]
    trial = 0
    while time.time() - t0 < 22.0:
        if time.time() - t0 < 8.0:
            # Phase 1: random start in the L, half the time biased to the
            # valuable right side.
            x = rng.uniform(0.03, 0.97, N) ** (0.5 if trial % 2 else 1.0)
            y = rng.uniform(0.03, 0.97, N)
            bad = (x > 0.57) & (y > 0.57)
            y[bad] = rng.uniform(0.03, 0.55, int(bad.sum()))
            v0 = np.concatenate([x, y, rng.uniform(0.02, 0.08, N)])
        else:
            # Phase 2: basin hopping - perturb the best layout and re-solve.
            v0 = np.concatenate([bx, by, br])
            mode = trial % 3
            if mode == 0:
                # Jitter all centers a little, shrink radii to regain slack.
                v0[:2 * N] += rng.normal(0.0, 0.02, 2 * N)
                v0[2 * N:] *= 0.9
            elif mode == 1:
                # Relocate the 1-2 least valuable circles to a random spot.
                value = v0[2 * N:] * (0.2 + v0[:N]) ** 2
                for i in np.argsort(value)[:int(rng.integers(1, 3))]:
                    px, py = rng.uniform(0.03, 0.97, 2)
                    if px > 0.57 and py > 0.57:
                        py = rng.uniform(0.03, 0.55)
                    v0[i], v0[N + i], v0[2 * N + i] = px, py, 0.01
            else:
                # Swap the positions of two circles (keeps their radii).
                i, j = (int(k) for k in rng.choice(N, 2, replace=False))
                v0[i], v0[j] = v0[j], v0[i]
                v0[N + i], v0[N + j] = v0[N + j], v0[N + i]
                v0[:2 * N] += rng.normal(0.0, 0.005, 2 * N)
                v0[2 * N:] *= 0.9
            v0[:2 * N] = np.clip(v0[:2 * N], 0.01, 0.99)
        trial += 1
        try:
            res = minimize(_obj, v0, jac=True, method="SLSQP", bounds=bounds,
                           constraints=cons, options=dict(maxiter=150, ftol=1e-10))
            v = res.x
            if not np.all(np.isfinite(v)):
                continue
            x, y = np.clip(v[:N], 0, 1), np.clip(v[N:2 * N], 0, 1)
            r = _repair(x, y, v[2 * N:].copy())
            s = _score(x, r)
            if s > best:
                best, bx, by, br = s, x, y, r
        except Exception:
            continue
    return np.column_stack([bx, by]), br


def greedy_radii_without_overlap(circle_centers):
    # Greedy and order-dependent: far from the best radii these centers allow.
    num_circles = circle_centers.shape[0]
    circle_radii = np.zeros(num_circles)

    # Each radius is limited by the four outer walls and by the removed block.
    for i in range(num_circles):
        center_x, center_y = circle_centers[i]
        distance_to_removed_block = np.hypot(max(0.6 - center_x, 0.0), max(0.6 - center_y, 0.0))
        circle_radii[i] = min(
            center_x, center_y, 1 - center_x, 1 - center_y, distance_to_removed_block
        )

    # Shrink overlapping pairs proportionally until they just touch.
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            center_distance = np.sqrt(np.sum((circle_centers[i] - circle_centers[j]) ** 2))

            if circle_radii[i] + circle_radii[j] > center_distance:
                shrink_factor = center_distance / (circle_radii[i] + circle_radii[j])
                circle_radii[i] *= shrink_factor
                circle_radii[j] *= shrink_factor

    return circle_radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
