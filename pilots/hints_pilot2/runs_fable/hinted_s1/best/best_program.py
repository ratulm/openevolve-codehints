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


def is_valid(c, r):
    # Strict validity check mirroring the evaluator's rules.
    x, y = c[:, 0], c[:, 1]
    if np.any(r <= 0) or np.any(x - r < 0) or np.any(y - r < 0):
        return False
    if np.any(x + r > 1) or np.any(y + r > 1):
        return False
    if np.any(np.hypot(np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)) < r):
        return False
    i, j = np.triu_indices(len(r), 1)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    return bool(np.all(d - r[i] - r[j] >= 1e-9))


def f(budget=20.0):
    # Multi-start SLSQP over centers and radii for sum r * (0.2 + x)^2.
    import time
    from scipy.optimize import minimize

    t0 = time.time()
    n = 21
    I, J = np.triu_indices(n, 1)
    m = len(I)
    A = np.arange(n)
    R = 5 * n + np.arange(m)

    def obj(z):
        x, r = z[:n], z[2 * n:]
        g = np.zeros(3 * n)
        g[:n] = -2 * r * (0.2 + x)
        g[2 * n:] = -((0.2 + x) ** 2)
        return -np.sum(r * (0.2 + x) ** 2), g

    def con(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        blk = np.hypot(np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)) - r
        pair = (x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2 - (r[I] + r[J]) ** 2
        return np.concatenate([x - r, y - r, 1 - x - r, 1 - y - r, blk, pair])

    def jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        G = np.zeros((5 * n + m, 3 * n))
        G[A, A] = 1
        G[A, 2 * n + A] = -1
        G[n + A, n + A] = 1
        G[n + A, 2 * n + A] = -1
        G[2 * n + A, A] = -1
        G[2 * n + A, 2 * n + A] = -1
        G[3 * n + A, n + A] = -1
        G[3 * n + A, 2 * n + A] = -1
        ex, ey = np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)
        h = np.maximum(np.hypot(ex, ey), 1e-12)
        G[4 * n + A, A] = -ex / h
        G[4 * n + A, n + A] = -ey / h
        G[4 * n + A, 2 * n + A] = -1
        dx, dy, s = x[I] - x[J], y[I] - y[J], r[I] + r[J]
        G[R, I] = 2 * dx
        G[R, J] = -2 * dx
        G[R, n + I] = 2 * dy
        G[R, n + J] = -2 * dy
        G[R, 2 * n + I] = -2 * s
        G[R, 2 * n + J] = -2 * s
        return G

    cons = [dict(type="ineq", fun=con, jac=jac)]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    rng = np.random.default_rng(0)
    best_c, best_r = naive_layout()
    best = np.sum(best_r * (0.2 + best_c[:, 0]) ** 2) if is_valid(best_c, best_r) else -1.0
    k = 0
    best_z = None
    while time.time() - t0 < budget:
        if best_z is None or time.time() - t0 < 0.4 * budget:
            # Exploration: random start inside the L region; half the starts
            # lean to the right.
            pts = []
            while len(pts) < n:
                p = rng.random(2)
                if k % 2 == 0:
                    p[0] = p[0] ** 0.6
                if not (p[0] > 0.58 and p[1] > 0.58):
                    pts.append(p)
            pts = np.array(pts)
            z0 = np.concatenate([pts[:, 0], pts[:, 1], np.full(n, 0.03)])
        else:
            # Refinement (basin hopping): jitter a random subset of the best
            # layout's centers, shrink radii, and re-optimize.
            z0 = best_z.copy()
            s = rng.choice([0.01, 0.03, 0.08])
            q = rng.random(n) < rng.choice([0.2, 0.5, 1.0])
            z0[:n] += s * rng.standard_normal(n) * q
            z0[n:2 * n] += s * rng.standard_normal(n) * q
            z0[:2 * n] = np.clip(z0[:2 * n], 0.01, 0.99)
            z0[2 * n:] *= 0.7
        k += 1
        try:
            res = minimize(obj, z0, jac=True, method="SLSQP", bounds=bounds,
                           constraints=cons, options=dict(maxiter=200, ftol=1e-10))
            z = res.x
        except Exception:
            continue
        if not np.all(np.isfinite(z)):
            continue
        c = np.column_stack([z[:n], z[n:2 * n]])
        r = z[2 * n:] - 1e-8
        for _ in range(8):
            if is_valid(c, r):
                break
            r = r * (1 - 1e-6) - 1e-8
        if is_valid(c, r):
            v = np.sum(r * (0.2 + c[:, 0]) ** 2)
            if v > best:
                best, best_c, best_r = v, c.copy(), r.copy()
                best_z = z.copy()
    return best_c, best_r


def naive_layout():
    # Fallback layout: a small grid squeezed into the bottom-left corner,
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
