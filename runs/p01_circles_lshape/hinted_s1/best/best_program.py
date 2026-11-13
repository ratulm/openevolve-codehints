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
import numpy as np


def f():
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

    circle_radii = greedy_radii_without_overlap(circle_centers)

    try:
        return optimize_layout(circle_centers, circle_radii)
    except Exception:
        return circle_centers, circle_radii


def _score(c, r):
    return float(np.sum(r * (0.2 + c[:, 0]) ** 2))


def _wall_limits(c):
    x, y = c[:, 0], c[:, 1]
    dx, dy = 0.6 - x, 0.6 - y
    d = np.where((dx > 0) & (dy > 0), np.hypot(dx, dy), np.maximum(dx, dy))
    return np.minimum.reduce([x, y, 1 - x, 1 - y, d])


def _repair(c, r):
    # Make the solution strictly valid: wall caps, small margin, pair shrink.
    r = np.minimum(r, _wall_limits(c))
    r = np.maximum(r * (1 - 1e-9) - 1e-12, 0.0)
    n = len(r)
    for _ in range(5):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(*(c[i] - c[j]))
                s = r[i] + r[j]
                if s > d:
                    f = max(d / s * (1 - 1e-9), 0.0)
                    r[i] *= f
                    r[j] *= f
                    changed = True
        if not changed:
            break
    return r


def _valid(c, r):
    if np.any(r < 0) or np.any(_wall_limits(c) - r < 0):
        return False
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(c[i] - c[j])) < r[i] + r[j]:
                return False
    return True


def optimize_layout(init_c, init_r, time_budget=20.0):
    import time
    from scipy.optimize import minimize

    n = init_c.shape[0]
    iu = np.triu_indices(n, 1)

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dx, dy = 0.6 - x, 0.6 - y
        d = np.where((dx > 0) & (dy > 0), np.hypot(dx, dy), np.maximum(dx, dy))
        D = np.hypot(x[iu[0]] - x[iu[1]], y[iu[0]] - y[iu[1]])
        return np.concatenate([x - r, y - r, 1 - x - r, 1 - y - r, d - r,
                               D - r[iu[0]] - r[iu[1]]])

    def obj(z):
        return -np.sum(z[2 * n:] * (0.2 + z[:n]) ** 2)

    def grad(z):
        g = np.zeros_like(z)
        x, r = z[:n], z[2 * n:]
        g[:n] = -2 * r * (0.2 + x)
        g[2 * n:] = -(0.2 + x) ** 2
        return g

    best_c, best_r = init_c.copy(), init_r.copy()
    best_v = _score(best_c, best_r)
    rng = np.random.default_rng(12345)
    start = time.time()
    trial = 0
    have_opt = False

    def rand_pt(bias):
        while True:
            p = rng.random(2)
            if bias:
                p[0] = 1 - p[0] ** 2  # bias toward the valuable right side
            if not (p[0] > 0.6 and p[1] > 0.6):
                return p

    while time.time() - start < time_budget:
        elapsed = time.time() - start
        if have_opt and elapsed > 0.45 * time_budget:
            # Basin-hopping phase: perturb the incumbent best solution.
            sigma = 10 ** rng.uniform(-2.5, -1.1)
            pts = best_c + rng.normal(0, sigma, best_c.shape)
            rr = best_r * 0.9
            if rng.random() < 0.25:
                k = rng.integers(n)
                pts[k] = rand_pt(rng.random() < 0.5)
                rr[k] = 0.01
            pts = np.clip(pts, 0.0, 1.0)
            trial += 1
            z0 = np.concatenate([pts[:, 0], pts[:, 1], np.maximum(rr, 0.0)])
        else:
            pts = np.array([rand_pt(trial % 2 == 1) for _ in range(n)])
            trial += 1
            z0 = np.concatenate([pts[:, 0], pts[:, 1], np.full(n, 0.01)])
        try:
            res = minimize(obj, z0, jac=grad, method='SLSQP',
                           constraints={'type': 'ineq', 'fun': cons},
                           bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                           options={'maxiter': 300, 'ftol': 1e-10})
        except Exception:
            continue
        z = res.x
        c = np.stack([z[:n], z[n:2 * n]], axis=1)
        c = np.clip(c, 0.0, 1.0)
        r = _repair(c, z[2 * n:].copy())
        v = _score(c, r)
        if v > best_v and _valid(c, r):
            best_v, best_c, best_r = v, c, r
            have_opt = True

    return best_c, best_r


def greedy_radii_without_overlap(circle_centers):
    # Greedy and order-dependent radius assignment for fixed centers.
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
