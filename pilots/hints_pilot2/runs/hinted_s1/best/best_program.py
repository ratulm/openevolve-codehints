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
import numpy as np
import time
from scipy.optimize import minimize

N = 21
IU = np.triu_indices(N, 1)


def _obj(z):
    return -np.sum(z[2*N:] * (0.2 + z[:N]) ** 2)


def _obj_g(z):
    x, r = z[:N], z[2*N:]
    g = np.zeros(3 * N)
    g[:N] = -2 * r * (0.2 + x)
    g[2*N:] = -(0.2 + x) ** 2
    return g


def _cons(z):
    x, y, r = z[:N], z[N:2*N], z[2*N:]
    walls = np.concatenate([x - r, y - r, 1 - x - r, 1 - y - r])
    dx = np.maximum(0.6 - x, 0.0)
    dy = np.maximum(0.6 - y, 0.0)
    blk = np.sqrt(dx**2 + dy**2 + 1e-18) - r
    d = np.sqrt((x[IU[0]] - x[IU[1]])**2 + (y[IU[0]] - y[IU[1]])**2 + 1e-18)
    return np.concatenate([walls, blk, d - r[IU[0]] - r[IU[1]]])


def _valid(x, y, r):
    if np.any(r < 0) or np.any(x - r < 0) or np.any(y - r < 0):
        return False
    if np.any(x + r > 1) or np.any(y + r > 1):
        return False
    if np.any(np.hypot(np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)) < r):
        return False
    d = np.hypot(x[IU[0]] - x[IU[1]], y[IU[0]] - y[IU[1]])
    return bool(np.all(d >= r[IU[0]] + r[IU[1]]))


def f():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    best, best_s, best_z = None, -1.0, None
    k = 0
    while time.time() - t0 < 9:
        pts = []
        while len(pts) < N:
            p = rng.random(2)
            if k % 2 and rng.random() < 0.5:
                p = np.array([0.4 + 0.6 * rng.random(), 0.6 * rng.random()])
            if p[0] > 0.6 and p[1] > 0.6:
                continue
            pts.append(p)
        pts = np.array(pts)
        k += 1
        z0 = np.concatenate([pts[:, 0], pts[:, 1], np.full(N, 0.02)])
        try:
            res = minimize(_obj, z0, jac=_obj_g, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": _cons}],
                           options={"maxiter": 300, "ftol": 1e-10})
        except Exception:
            continue
        z = res.x
        x, y = z[:N].copy(), z[N:2*N].copy()
        r = np.maximum(z[2*N:] - 1e-8, 0.0)
        if _valid(x, y, r):
            s = np.sum(r * (0.2 + x) ** 2)
            if s > best_s:
                best_s, best, best_z = s, (np.column_stack([x, y]), r), z.copy()
    # Phase 2: basin hopping around the incumbent (relocate 1-3 circles + jitter).
    while best_z is not None and time.time() - t0 < 21:
        z0 = best_z.copy()
        m = int(rng.integers(1, 4))
        idx = rng.choice(N, m, replace=False)
        for i in idx:
            while True:
                px, py = rng.random(2)
                if not (px > 0.6 and py > 0.6):
                    break
            z0[i], z0[N + i], z0[2 * N + i] = px, py, 0.01
        z0[:2 * N] = np.clip(z0[:2 * N] + rng.normal(0, 0.01, 2 * N), 0, 1)
        try:
            res = minimize(_obj, z0, jac=_obj_g, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": _cons}],
                           options={"maxiter": 300, "ftol": 1e-10})
        except Exception:
            continue
        z = res.x
        x, y = z[:N].copy(), z[N:2*N].copy()
        r = np.maximum(z[2*N:] - 1e-8, 0.0)
        if _valid(x, y, r):
            s = np.sum(r * (0.2 + x) ** 2)
            if s > best_s:
                best_s, best, best_z = s, (np.column_stack([x, y]), r), z.copy()
    if best is not None:
        return best
    return _fallback()


def _fallback():
    num_circles = 21
    circle_centers = np.zeros((num_circles, 2))

    # Three rows of 5 below the removed block, then two rows of 3 beside it.
    circle_index = 0
    for row in range(5):
        circles_in_row = 5 if row < 3 else 3
        for column in range(circles_in_row):
            circle_centers[circle_index] = [0.06 + 0.12 * column, 0.06 + 0.12 * row]
            circle_index += 1

    # TODO: replace this greedy radius assignment with joint optimization of
    # centers and radii for the position-weighted objective (see HINT above).
    circle_radii = greedy_radii_without_overlap(circle_centers)

    return circle_centers, circle_radii


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
