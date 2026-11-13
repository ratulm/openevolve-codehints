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

_N = 21
_I, _J = np.triu_indices(_N, 1)


def _cons(z):
    N = _N
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    dx, dy = np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)
    blk = np.sqrt(dx * dx + dy * dy + 1e-18) - r
    pair = (x[_I] - x[_J]) ** 2 + (y[_I] - y[_J]) ** 2 - (r[_I] + r[_J]) ** 2
    return np.concatenate([x - r, y - r, 1 - x - r, 1 - y - r, blk, pair])


def _cons_jac(z):
    N = _N
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    E, Z = np.eye(N), np.zeros((N, N))
    rows = [np.hstack([E, Z, -E]), np.hstack([Z, E, -E]),
            np.hstack([-E, Z, -E]), np.hstack([Z, -E, -E])]
    dx, dy = np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    rows.append(np.hstack([np.diag(-dx / d), np.diag(-dy / d), -E]))
    M = len(_I)
    P = np.zeros((M, 3 * N))
    k = np.arange(M)
    ddx, ddy, sr = x[_I] - x[_J], y[_I] - y[_J], r[_I] + r[_J]
    P[k, _I] = 2 * ddx
    P[k, _J] = -2 * ddx
    P[k, N + _I] = 2 * ddy
    P[k, N + _J] = -2 * ddy
    P[k, 2 * N + _I] = -2 * sr
    P[k, 2 * N + _J] = -2 * sr
    rows.append(P)
    return np.vstack(rows)


def _obj(z):
    return -np.sum(z[2 * _N:] * (0.2 + z[:_N]) ** 2)


def _obj_jac(z):
    x, r = z[:_N], z[2 * _N:]
    g = np.zeros(3 * _N)
    g[:_N] = -2 * r * (0.2 + x)
    g[2 * _N:] = -(0.2 + x) ** 2
    return g


def _make_valid(c, r):
    r = np.array(r, dtype=float)
    x, y = c[:, 0], c[:, 1]
    lim = np.minimum.reduce([x, y, 1 - x, 1 - y,
                             np.hypot(np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0))])
    r = np.clip(np.minimum(r, lim) - 1e-9, 0, None)
    for _ in range(3):
        for i in range(_N):
            for j in range(i + 1, _N):
                d = float(np.hypot(*(c[i] - c[j])))
                if r[i] + r[j] > d - 1e-10:
                    s = max(d - 1e-10, 0.0) / (r[i] + r[j])
                    r[i] *= s
                    r[j] *= s
    return r


def _random_start(rng):
    pts = []
    pr = 0.4 + 0.4 * rng.random()
    while len(pts) < _N:
        if rng.random() < pr:
            p = np.array([0.6 + 0.4 * rng.random(), 0.6 * rng.random()])
        else:
            p = rng.random(2)
            if p[0] > 0.6 and p[1] > 0.6:
                continue
        pts.append(p)
    pts = np.array(pts)
    r = (0.01 + 0.05 * rng.random()) * (0.5 + rng.random(_N))
    return np.concatenate([pts[:, 0], pts[:, 1], r])


def _solve(z0, minimize, bounds):
    res = minimize(_obj, z0, jac=_obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                   options={"maxiter": 400, "ftol": 1e-12})
    z = res.x
    c = np.stack([z[:_N], z[_N:2 * _N]], axis=1)
    r = _make_valid(c, z[2 * _N:] - 1e-8)
    return z, c, r, np.sum(r * (0.2 + c[:, 0]) ** 2)


def f():
    t0 = time.time()
    c0, r0 = _grid_layout()
    best_c, best_r = c0, r0
    best_s = np.sum(r0 * (0.2 + c0[:, 0]) ** 2)
    best_z = None
    try:
        from scipy.optimize import minimize
    except Exception:
        return best_c, best_r
    rng = np.random.default_rng(12345)
    lo = np.zeros(3 * _N)
    hi = np.concatenate([np.ones(2 * _N), np.full(_N, 0.5)])
    bounds = list(zip(lo, hi))
    it = 0
    while time.time() - t0 < 21:
        it += 1
        try:
            mode = it % 3
            if best_z is not None and mode == 1:
                # Relocate one (often the smallest) circle into the valuable strip.
                z0 = best_z.copy()
                k = int(np.argmin(best_r)) if rng.random() < 0.5 else int(rng.integers(_N))
                z0[k] = 0.6 + 0.4 * rng.random()
                z0[_N + k] = 0.6 * rng.random()
                z0[2 * _N + k] = 0.01
                z0[:2 * _N] += 0.005 * rng.standard_normal(2 * _N)
                z0 = np.clip(z0, lo, hi)
                z0[2 * _N:] *= 0.95
            elif best_z is not None and mode == 2:
                sc = 0.005 + 0.025 * rng.random()
                z0 = np.clip(best_z + sc * rng.standard_normal(3 * _N), lo, hi)
                z0[2 * _N:] *= 0.9
            else:
                z0 = _random_start(rng)
            z, c, r, s = _solve(z0, minimize, bounds)
            if np.all(np.isfinite(c)) and np.all(np.isfinite(r)) and s > best_s:
                best_c, best_r, best_s, best_z = c, r, s, z
        except Exception:
            continue
    return best_c, best_r


def _grid_layout():
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
