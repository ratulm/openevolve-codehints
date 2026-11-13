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


def _obj(v):
    x, r = v[:N], v[2 * N:]
    w = (0.2 + x) ** 2
    g = np.zeros(3 * N)
    g[:N] = -r * 2 * (0.2 + x)
    g[2 * N:] = -w
    return -np.sum(r * w), g


def _block(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    dx, dy = np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)
    d = np.hypot(dx, dy)
    d = np.where(d < 1e-12, -np.minimum(x - 0.6, y - 0.6), d)
    return d - r


def _block_jac(v):
    x, y = v[:N], v[N:2 * N]
    dx, dy = np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)
    d = np.hypot(dx, dy)
    J = np.zeros((N, 3 * N))
    k = np.arange(N)
    ok = d > 1e-12
    s = np.where(ok, d, 1.0)
    J[k, k] = np.where(ok, -dx / s, -1.0 * (x <= y))
    J[k, N + k] = np.where(ok, -dy / s, -1.0 * (x > y))
    J[k, 2 * N + k] = -1
    return J


def _walls(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    return np.concatenate([x - r, y - r, 1 - x - r, 1 - y - r])


_E = np.eye(N)
_Z = np.zeros((N, N))
WJ = np.block([[_E, _Z, -_E], [_Z, _E, -_E], [-_E, _Z, -_E], [_Z, -_E, -_E]])


def _pairs(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    i, j = IU
    return (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 - (r[i] + r[j]) ** 2


def _pairs_jac(v):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    i, j = IU
    m = len(i)
    J = np.zeros((m, 3 * N))
    a = np.arange(m)
    dx, dy, sr = 2 * (x[i] - x[j]), 2 * (y[i] - y[j]), -2 * (r[i] + r[j])
    J[a, i] = dx
    J[a, j] = -dx
    J[a, N + i] = dy
    J[a, N + j] = -dy
    J[a, 2 * N + i] = sr
    J[a, 2 * N + j] = sr
    return J


CONS = [
    dict(type="ineq", fun=_walls, jac=lambda v: WJ),
    dict(type="ineq", fun=_block, jac=_block_jac),
    dict(type="ineq", fun=_pairs, jac=_pairs_jac),
]


def _valid(c, r):
    x, y = c[:, 0], c[:, 1]
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
        return False
    if np.any(r <= 0) or np.any(x - r < 0) or np.any(y - r < 0) or np.any(x + r > 1) or np.any(y + r > 1):
        return False
    if np.any(_block(np.concatenate([x, y, r])) < 0):
        return False
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    s = r[:, None] + r[None]
    return bool(np.all((d - s)[IU] >= 0))


def _score(c, r):
    return np.sum(r * (0.2 + c[:, 0]) ** 2)


def f():
    rng = np.random.default_rng(0)
    best, best_s = None, -1.0
    t0 = time.time()
    k = 0
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    while time.time() - t0 < 20:
        if best is not None and time.time() - t0 > 9:
            # Basin hopping: perturb the incumbent and re-optimize.
            c0, r0 = best
            x0, y0 = c0[:, 0].copy(), c0[:, 1].copy()
            rr = r0.copy()
            sig = rng.choice([0.01, 0.03, 0.06])
            x0 += rng.normal(0, sig, N)
            y0 += rng.normal(0, sig, N)
            rr *= 0.9
            if rng.random() < 0.5:
                w = rr * (0.2 + x0) ** 2
                idx = np.argsort(w)[:rng.integers(1, 3)]
                for t in idx:
                    while True:
                        p = rng.random(2)
                        if not (p[0] > 0.6 and p[1] > 0.6):
                            break
                    x0[t], y0[t], rr[t] = p[0], p[1], 0.01
            x0, y0 = np.clip(x0, 0.01, 0.99), np.clip(y0, 0.01, 0.99)
            v0 = np.concatenate([x0, y0, np.maximum(rr, 0.005)])
        else:
            pts = []
            while len(pts) < N:
                p = rng.random(2)
                if not (p[0] > 0.6 and p[1] > 0.6):
                    if k % 2 == 0 or rng.random() < 0.3 + 0.7 * p[0]:
                        pts.append(p)
            pts = np.array(pts)
            v0 = np.concatenate([pts[:, 0], pts[:, 1], np.full(N, 0.02)])
        k += 1
        try:
            res = minimize(_obj, v0, jac=True, method="SLSQP", bounds=bounds,
                           constraints=CONS, options=dict(maxiter=300, ftol=1e-10))
        except Exception:
            continue
        v = res.x
        c = np.stack([v[:N], v[N:2 * N]], 1)
        r = v[2 * N:] - 1e-8
        if _valid(c, r):
            s = _score(c, r)
            if s > best_s:
                best_s, best = s, (c, r)
    if best is not None:
        return best
    return f_grid()


def f_grid():
    # Naive starting layout: a small grid squeezed into the bottom-left corner,
    # which is the least valuable part of the region.
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
