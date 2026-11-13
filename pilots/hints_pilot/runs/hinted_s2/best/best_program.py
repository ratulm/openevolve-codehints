# EVOLVE-BLOCK-START
# Goal: pack 26 non-overlapping circles inside the unit square so that the
# sum of their radii is as large as possible (best known value is about 2.635).
#
# HINT: hand-placed geometric patterns (rings, grids, hexagonal rows) plateau
# well below the best known value. A stronger approach is to treat this as a
# constrained optimization problem: the variables are the 26 centers and 26
# radii, the objective is the sum of radii, and the constraints are
# radius_i <= distance to each wall and |center_i - center_j| >= radius_i +
# radius_j for every pair. Solve it with scipy.optimize.minimize(method="SLSQP")
# starting from several structured layouts with small random perturbations,
# and keep the best valid result. Circles of unequal size pack better than
# equal ones.
import numpy as np


def construct_packing():
    # Naive hand-placed starting layout: this is only a seed, not a solution.
    num_circles = 26
    circle_centers = np.zeros((num_circles, 2))

    # One circle in the middle of the square.
    circle_centers[0] = [0.5, 0.5]

    # Inner ring of 8 circles.
    for i in range(8):
        ring_angle = 2 * np.pi * i / 8
        circle_centers[i + 1] = [0.5 + 0.3 * np.cos(ring_angle), 0.5 + 0.3 * np.sin(ring_angle)]

    # Outer ring of 16 circles. Radius 0.7 puts many centers outside the square,
    # so the clip below piles them onto the walls and wastes space.
    for i in range(16):
        ring_angle = 2 * np.pi * i / 16
        circle_centers[i + 9] = [0.5 + 0.7 * np.cos(ring_angle), 0.5 + 0.7 * np.sin(ring_angle)]

    # Force every center inside the unit square.
    circle_centers = np.clip(circle_centers, 0.01, 0.99)

    # TODO: replace this greedy radius assignment with joint optimization of
    # centers and radii (see HINT at the top).
    circle_radii = greedy_radii_without_overlap(circle_centers)

    # Joint optimization of centers and radii (SLSQP multi-start); keep the
    # greedy layout only as a fallback.
    try:
        opt_centers, opt_radii = optimize_packing(num_circles)
        if opt_radii.sum() > circle_radii.sum():
            circle_centers, circle_radii = opt_centers, opt_radii
    except Exception:
        pass

    sum_of_radii = np.sum(circle_radii)

    return circle_centers, circle_radii, sum_of_radii


def optimize_packing(n=26, time_budget=25.0):
    # Maximize sum of radii over centers and radii with SLSQP from many starts.
    import time
    from scipy.optimize import minimize

    t0 = time.time()
    rng = np.random.default_rng(0)
    I, J = np.triu_indices(n, 1)
    m = len(I)
    rows = np.arange(m)
    k = np.arange(n)
    # Constant Jacobian of wall constraints: x-r, 1-x-r, y-r, 1-y-r >= 0.
    W = np.zeros((4 * n, 3 * n))
    W[k, k] = 1
    W[n + k, k] = -1
    W[2 * n + k, n + k] = 1
    W[3 * n + k, n + k] = -1
    for q in range(4):
        W[q * n + k, 2 * n + k] = -1
    w0 = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)])

    def con(v):
        x, y, r = v[:n], v[n:2 * n], v[2 * n:]
        d = np.hypot(x[I] - x[J], y[I] - y[J])
        return np.concatenate([d - r[I] - r[J], W @ v + w0])

    def jac(v):
        x, y = v[:n], v[n:2 * n]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        G = np.zeros((m, 3 * n))
        G[rows, I] = dx / d
        G[rows, J] = -dx / d
        G[rows, n + I] = dy / d
        G[rows, n + J] = -dy / d
        G[rows, 2 * n + I] = -1
        G[rows, 2 * n + J] = -1
        return np.vstack([G, W])

    grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    cons = [dict(type="ineq", fun=con, jac=jac)]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    # Structured seeds: staggered rows and a 5x5 grid plus one extra circle.
    base = []
    for rowsz in ([5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
                  [4, 6, 6, 6, 4], [6, 7, 6, 7], [5, 4, 5, 4, 5, 3]):
        base.append(np.array([[(b + 0.5) / c, (a + 0.5) / len(rowsz)]
                              for a, c in enumerate(rowsz) for b in range(c)]))
    g5 = [[(a + 0.5) / 5, (b + 0.5) / 5] for a in range(5) for b in range(5)]
    base.append(np.array(g5 + [[0.4, 0.4]]))

    best, best_v, it = -1.0, None, 0
    need_polish = False
    while time.time() - t0 < time_budget:
        polish = need_polish and best_v is not None and it >= len(base)
        if polish:
            # Tight-tolerance restart from the incumbent itself (no noise).
            need_polish = False
            v0 = best_v.copy()
        else:
            if it < len(base):
                p = base[it] + rng.normal(0, 0.01, (n, 2))
            elif best_v is not None and it % 2 == 0:
                # Perturb the incumbent solution at varying scales.
                sig = (0.005, 0.015, 0.04)[(it // 2) % 3]
                p = np.column_stack([best_v[:n], best_v[n:2 * n]]) + rng.normal(0, sig, (n, 2))
            else:
                p = base[rng.integers(len(base))] + rng.normal(0, 0.03, (n, 2))
            p = np.clip(p, 0.05, 0.95)
            v0 = np.concatenate([p[:, 0], p[:, 1], np.full(n, 0.03)])
        it += 1
        try:
            res = minimize(lambda v: -v[2 * n:].sum(), v0, jac=lambda v: grad,
                           bounds=bounds, constraints=cons, method="SLSQP",
                           options=dict(maxiter=1000 if polish else 300,
                                        ftol=1e-14 if polish else 1e-10))
        except Exception:
            continue
        v = np.array(res.x, dtype=float)
        if not np.all(np.isfinite(v)):
            continue
        v[:2 * n] = np.clip(v[:2 * n], 0, 1)
        x, y = v[:n], v[n:2 * n]
        # Repair to strict feasibility: clip to walls, shrink any overlaps.
        r = np.maximum(v[2 * n:], 0)
        r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
        d = np.hypot(x[I] - x[J], y[I] - y[J])
        for _ in range(200):
            ov = r[I] + r[J] - d
            q = np.argmax(ov)
            if ov[q] <= 0:
                break
            s = d[q] / (r[I[q]] + r[J[q]])
            r[I[q]] *= s
            r[J[q]] *= s
        r = np.maximum(r - 1e-12, 0)
        if (r[I] + r[J] <= d).all() and r.sum() > best:
            best = r.sum()
            v[2 * n:] = r
            best_v = v
            # Polish a new incumbent once (not after a polish step itself).
            need_polish = not polish
    return np.column_stack([best_v[:n], best_v[n:2 * n]]), best_v[2 * n:].copy()


def greedy_radii_without_overlap(circle_centers):
    # Greedy and order-dependent: far from the largest radii these centers allow.
    num_circles = circle_centers.shape[0]
    circle_radii = np.ones(num_circles)

    # Each radius is limited by the distance to the nearest wall.
    for i in range(num_circles):
        center_x, center_y = circle_centers[i]
        circle_radii[i] = min(center_x, center_y, 1 - center_x, 1 - center_y)

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
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
