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


def f(time_budget=18.0):
    # Multi-start SLSQP over centers and radii; falls back to the grid layout.
    try:
        result = optimized_layout(time_budget)
        if result is not None:
            return result
    except Exception:
        pass
    return grid_layout()


def optimized_layout(time_budget):
    from scipy.optimize import minimize

    t0 = time.time()
    n = 21
    I, J = np.triu_indices(n, 1)
    m = len(I)
    ar = np.arange(n)
    pr = np.arange(m)

    def obj(z):
        x, r = z[:n], z[2 * n:]
        g = np.zeros(3 * n)
        g[:n] = -2 * r * (0.2 + x)
        g[2 * n:] = -(0.2 + x) ** 2
        return -np.sum(r * (0.2 + x) ** 2), g

    def block(x, y):
        # Signed distance to the removed block (negative inside) and gradient.
        ax, ay = 0.6 - x, 0.6 - y
        px, py = np.maximum(ax, 0), np.maximum(ay, 0)
        d = np.hypot(px, py)
        inside = (ax < 0) & (ay < 0)
        ds = np.where(d > 0, d, 1.0)
        gx, gy = -px / ds, -py / ds
        xm = ax > ay
        d = np.where(inside, np.maximum(ax, ay), d)
        gx = np.where(inside, np.where(xm, -1.0, 0.0), gx)
        gy = np.where(inside, np.where(xm, 0.0, -1.0), gy)
        return d, gx, gy

    def con(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        dist = np.hypot(x[I] - x[J], y[I] - y[J])
        return np.concatenate([x - r, y - r, 1 - x - r, 1 - y - r,
                               block(x, y)[0] - r, dist - r[I] - r[J]])

    # Constant part of the constraint Jacobian.
    A = np.zeros((5 * n + m, 3 * n))
    A[ar, ar] = 1
    A[ar, 2 * n + ar] = -1
    A[n + ar, n + ar] = 1
    A[n + ar, 2 * n + ar] = -1
    A[2 * n + ar, ar] = -1
    A[2 * n + ar, 2 * n + ar] = -1
    A[3 * n + ar, n + ar] = -1
    A[3 * n + ar, 2 * n + ar] = -1
    A[4 * n + ar, 2 * n + ar] = -1
    A[5 * n + pr, 2 * n + I] = -1
    A[5 * n + pr, 2 * n + J] = -1

    def jac(z):
        x, y = z[:n], z[n:2 * n]
        G = A.copy()
        _, gx, gy = block(x, y)
        G[4 * n + ar, ar] = gx
        G[4 * n + ar, n + ar] = gy
        dx, dy = x[I] - x[J], y[I] - y[J]
        dist = np.maximum(np.hypot(dx, dy), 1e-12)
        G[5 * n + pr, I] = dx / dist
        G[5 * n + pr, J] = -dx / dist
        G[5 * n + pr, n + I] = dy / dist
        G[5 * n + pr, n + J] = -dy / dist
        return G

    def finalize(z):
        # Repair to strict feasibility: clip radii to walls/block, then pairs.
        x, y = np.clip(z[:n], 0, 1), np.clip(z[n:2 * n], 0, 1)
        d = block(x, y)[0]
        if np.any(d < 0):
            return None
        r = np.minimum(z[2 * n:], np.minimum.reduce([x, y, 1 - x, 1 - y, d]))
        r = np.maximum(r, 0)
        dist = np.hypot(x[I] - x[J], y[I] - y[J])
        for _ in range(2000):
            over = r[I] + r[J] - dist
            k = int(np.argmax(over))
            if over[k] <= 1e-13:
                break
            s = dist[k] / (r[I[k]] + r[J[k]])
            r[I[k]] *= s
            r[J[k]] *= s
        r = np.maximum(r - 2e-9, 0) * (1 - 1e-8)
        if np.any(r[I] + r[J] > dist) or not np.all(np.isfinite(r)):
            return None
        return x, y, r, float(np.sum(r * (0.2 + x) ** 2))

    rng = np.random.default_rng(12345)
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    cons = [{"type": "ineq", "fun": con, "jac": jac}]
    best, best_val, best_z = None, -1.0, None
    trial = 0
    while time.time() - t0 < time_budget:
        if best_z is not None and time.time() - t0 > 0.35 * time_budget:
            # Phase 2: basin hopping around the incumbent.
            z0 = best_z.copy()
            mode = trial % 3
            if mode == 0:
                # Multi-scale jitter of every center, with shrunken radii.
                z0[:2 * n] += rng.normal(0, rng.choice([0.005, 0.02, 0.04]), 2 * n)
                z0[2 * n:] *= 0.7
            else:
                k = int(rng.integers(1, 4))
                if mode == 1:
                    # Relocate some of the least valuable circles.
                    order = np.argsort(z0[2 * n:] * (0.2 + z0[:n]) ** 2)
                    idx = rng.choice(order[:8], k, replace=False)
                else:
                    idx = rng.choice(n, k, replace=False)
                for i in idx:
                    while True:
                        qx, qy = rng.uniform(0.03, 0.97, 2)
                        if not (qx > 0.57 and qy > 0.57):
                            break
                    z0[i], z0[n + i], z0[2 * n + i] = qx, qy, 0.003
                z0[2 * n:] *= 0.95
        else:
            pts = []
            while len(pts) < n:
                p = rng.uniform(0.03, 0.97, 2)
                p[0] = p[0] ** rng.choice([0.5, 1.0])  # bias to the right
                if not (p[0] > 0.57 and p[1] > 0.57):
                    pts.append(p)
            pts = np.array(pts)
            z0 = np.concatenate([pts[:, 0], pts[:, 1], np.full(n, 0.02)])
        z0[:2 * n] = np.clip(z0[:2 * n], 0.01, 0.99)
        trial += 1
        try:
            res = minimize(obj, z0, jac=True, method="SLSQP", bounds=bounds,
                           constraints=cons,
                           options={"maxiter": 120, "ftol": 1e-10})
            out = finalize(res.x)
        except Exception:
            continue
        if out is not None and out[3] > best_val:
            x, y, r, best_val = out
            best = (np.column_stack([x, y]), r)
            best_z = np.concatenate([x, y, r])
    return best


def grid_layout():
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
