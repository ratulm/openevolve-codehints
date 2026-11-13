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


N = 21


def _wall(x, y):
    dbx = np.maximum(0.6 - x, 0)
    dby = np.maximum(0.6 - y, 0)
    return np.minimum.reduce([x, y, 1 - x, 1 - y, np.sqrt(dbx**2 + dby**2)])


def _loss(z, mu):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    w = (0.2 + x) ** 2
    dbx = np.maximum(0.6 - x, 0)
    dby = np.maximum(0.6 - y, 0)
    db = np.sqrt(dbx**2 + dby**2 + 1e-12)
    v = [np.maximum(r - x, 0), np.maximum(r - y, 0), np.maximum(r + x - 1, 0),
         np.maximum(r + y - 1, 0), np.maximum(r - db, 0), np.maximum(-r, 0)]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2 + 1e-12)
    O = np.maximum(r[:, None] + r[None, :] - d, 0)
    np.fill_diagonal(O, 0)
    pen = sum(np.sum(a**2) for a in v) + 0.5 * np.sum(O**2)
    k = 2 * mu
    gx = -r * 2 * (0.2 + x) + k * (-v[0] + v[2] + v[4] * dbx / db) - k * np.sum(O * dx / d, 1)
    gy = k * (-v[1] + v[3] + v[4] * dby / db) - k * np.sum(O * dy / d, 1)
    gr = -w + k * (v[0] + v[1] + v[2] + v[3] + v[4] - v[5]) + k * O.sum(1)
    return -np.sum(r * w) + mu * pen, np.concatenate([gx, gy, gr])


def _fix(x, y, r):
    r = np.maximum(np.minimum(r, _wall(x, y) - 1e-12), 0)
    for _ in range(500):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if r[i] + r[j] > d - 1e-12:
                    s = max(d - 1e-12, 0) / (r[i] + r[j])
                    r[i] *= s
                    r[j] *= s
                    changed = True
        if not changed:
            break
    return r


def f():
    try:
        from scipy.optimize import minimize
    except Exception:
        return f_grid()
    import time
    rng = np.random.default_rng(0)
    best, best_score, best_z = None, -1.0, None
    t0 = time.time()

    def solve(z, mus):
        for mu in mus:
            z = minimize(_loss, z, args=(mu,), jac=True, method='L-BFGS-B').x
        return z

    def evaluate(z):
        x, y, r = z[:N].copy(), z[N:2 * N].copy(), z[2 * N:].copy()
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = _fix(x, y, r)
        return np.sum(r * (0.2 + x) ** 2), (np.stack([x, y], 1), r)

    while time.time() - t0 < 9:
        if rng.random() < 0.5:
            x = 1 - rng.uniform(0, 1, N) ** 2
        else:
            x = rng.uniform(0, 1, N)
        y = rng.uniform(0, 1, N)
        m = (x > 0.6) & (y > 0.6)
        y[m] = rng.uniform(0, 0.6, m.sum())
        z = np.concatenate([x, y, rng.uniform(0.01, 0.08, N)])
        try:
            z = solve(z, [10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
            s, sol = evaluate(z)
        except Exception:
            continue
        if s > best_score:
            best_score, best, best_z = s, sol, z

    # Basin hopping around the incumbent.
    while best_z is not None and time.time() - t0 < 19:
        z = best_z.copy()
        sig = rng.choice([0.005, 0.02, 0.05])
        z[:2 * N] += rng.normal(0, sig, 2 * N)
        z[2 * N:] *= 0.9
        if rng.random() < 0.3:
            i, j = rng.choice(N, 2, replace=False)
            for o in (0, N):
                z[o + i], z[o + j] = z[o + j], z[o + i]
        try:
            z = solve(z, [1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
            s, sol = evaluate(z)
        except Exception:
            continue
        if s > best_score:
            best_score, best, best_z = s, sol, z
    if best is None:
        return f_grid()
    return best


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

    circle_radii = greedy_radii_without_overlap(circle_centers)

    return circle_centers, circle_radii


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
