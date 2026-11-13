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


import time

N = 21


def _block_dist(x, y):
    return np.hypot(np.maximum(0.6 - x, 0.0), np.maximum(0.6 - y, 0.0))


def _obj(v, k):
    x, y, r = v[:N], v[N:2 * N], v[2 * N:]
    w = (0.2 + x) ** 2
    s = -np.sum(r * w)
    gx = -(r * 2 * (0.2 + x))
    gy = np.zeros(N)
    gr = -w.copy()
    p = 0.0
    # wall terms: violation a = r - x, etc.
    for a, sx, sy in ((r - x, -1, 0), (r - y, 0, -1), (r + x - 1, 1, 0), (r + y - 1, 0, 1)):
        a = np.maximum(a, 0.0)
        p += np.sum(a * a)
        gr += 2 * k * a
        gx += 2 * k * a * sx
        gy += 2 * k * a * sy
    # removed block
    ux = np.maximum(0.6 - x, 0.0)
    uy = np.maximum(0.6 - y, 0.0)
    bd = np.hypot(ux, uy)
    a = np.maximum(r - bd, 0.0)
    p += np.sum(a * a)
    gr += 2 * k * a
    safe = np.where(bd > 1e-12, bd, 1.0)
    gx += 2 * k * a * np.where(bd > 1e-12, ux / safe, 0.0)
    gy += 2 * k * a * np.where(bd > 1e-12, uy / safe, 0.0)
    # negative radius
    a = np.maximum(-r, 0.0)
    p += np.sum(a * a)
    gr -= 2 * k * a
    # pairwise overlap
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-12)
    o = np.maximum(r[:, None] + r[None, :] - d, 0.0)
    np.fill_diagonal(o, 0.0)
    p += 0.5 * np.sum(o * o)
    gr += 2 * k * np.sum(o, axis=1)
    gx += 2 * k * np.sum(-o * dx / d, axis=1)
    gy += 2 * k * np.sum(-o * dy / d, axis=1)
    return s + k * p, np.concatenate([gx, gy, gr])


def _optimize(time_limit=12.0):
    from scipy.optimize import minimize
    rng = np.random.default_rng(12345)
    best_c, best_r, best_s = None, None, -1.0
    t0 = time.time()
    trial = 0
    while time.time() - t0 < time_limit:
        p = rng.random((N, 2))
        if trial % 2 == 1:
            p[:, 0] = 1 - p[:, 0] ** 2  # bias toward right
        m = (p[:, 0] > 0.6) & (p[:, 1] > 0.6)
        p[m, 1] *= 0.6
        v = np.concatenate([p[:, 0], p[:, 1], np.full(N, 0.04)])
        for k in (10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
            v = minimize(_obj, v, args=(k,), jac=True, method="L-BFGS-B",
                         options={"maxiter": 3000}).x
        c = np.stack([v[:N], v[N:2 * N]], axis=1)
        r = greedy_radii_without_overlap(c, v[2 * N:])
        s = np.sum(r * (0.2 + c[:, 0]) ** 2)
        if s > best_s:
            best_c, best_r, best_s = c, r, s
        trial += 1
    return best_c, best_r


def f():
    try:
        c, r = _optimize()
        if c is not None:
            return c, r
    except Exception:
        pass
    # Fallback: naive grid layout.
    num_circles = 21
    circle_centers = np.zeros((num_circles, 2))
    circle_index = 0
    for row in range(5):
        circles_in_row = 5 if row < 3 else 3
        for column in range(circles_in_row):
            circle_centers[circle_index] = [0.06 + 0.12 * column, 0.06 + 0.12 * row]
            circle_index += 1
    circle_radii = greedy_radii_without_overlap(circle_centers)
    return circle_centers, circle_radii


def greedy_radii_without_overlap(circle_centers, initial_radii=None):
    # Radius assignment for fixed centers, optionally capped by given radii.
    num_circles = circle_centers.shape[0]
    circle_radii = np.zeros(num_circles)

    # Each radius is limited by the four outer walls and by the removed block.
    for i in range(num_circles):
        center_x, center_y = circle_centers[i]
        distance_to_removed_block = np.hypot(max(0.6 - center_x, 0.0), max(0.6 - center_y, 0.0))
        circle_radii[i] = max(0.0, min(
            center_x, center_y, 1 - center_x, 1 - center_y, distance_to_removed_block
        ))
        if initial_radii is not None:
            circle_radii[i] = max(0.0, min(circle_radii[i], initial_radii[i]))

    # Shrink overlapping pairs proportionally until they just touch.
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            center_distance = np.sqrt(np.sum((circle_centers[i] - circle_centers[j]) ** 2))

            if circle_radii[i] + circle_radii[j] > center_distance:
                shrink_factor = center_distance / (circle_radii[i] + circle_radii[j])
                circle_radii[i] *= shrink_factor
                circle_radii[j] *= shrink_factor

    return circle_radii * (1 - 1e-9)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
