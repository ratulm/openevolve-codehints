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
    seed_centers = circle_centers
    circle_centers, circle_radii = refine_packing(seed_centers)
    if circle_radii is None:  # optimizer failed: fall back to the greedy seed
        circle_centers = seed_centers
        circle_radii = greedy_radii_without_overlap(circle_centers)

    sum_of_radii = np.sum(circle_radii)

    return circle_centers, circle_radii, sum_of_radii


def refine_packing(seed_centers, time_budget=30.0):
    # Joint SLSQP optimization of centers and radii from several perturbed starts.
    import time
    from scipy.optimize import minimize

    t0 = time.time()
    n = seed_centers.shape[0]
    rng = np.random.default_rng(0)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    rows = np.arange(m)

    def pair_fun(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        return (x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2 - (r[iu] + r[ju]) ** 2

    def pair_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        J = np.zeros((m, 3 * n))
        dx, dy, sr = 2 * (x[iu] - x[ju]), 2 * (y[iu] - y[ju]), 2 * (r[iu] + r[ju])
        J[rows, iu] = dx
        J[rows, ju] = -dx
        J[rows, n + iu] = dy
        J[rows, n + ju] = -dy
        J[rows, 2 * n + iu] = -sr
        J[rows, 2 * n + ju] = -sr
        return J

    # Linear wall constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0.
    I = np.eye(n)
    Z = np.zeros((n, n))
    W = np.vstack([np.hstack([I, Z, -I]), np.hstack([-I, Z, -I]),
                   np.hstack([Z, I, -I]), np.hstack([Z, -I, -I])])
    w0 = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)])
    cons = [dict(type="ineq", fun=pair_fun, jac=pair_jac),
            dict(type="ineq", fun=lambda z: W @ z + w0, jac=lambda z: W)]
    grad = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    def rows_layout(counts):
        pts = []
        for a, c in enumerate(counts):
            for b in range(c):
                pts.append([(b + 0.5) / c, (a + 0.5) / len(counts)])
        return np.array(pts)[:n]

    layouts = [rows_layout(c) for c in ([5, 6, 5, 5, 5], [5, 5, 6, 5, 5], [6, 5, 5, 5, 5],
                                        [5, 4, 5, 4, 5, 3], [6, 5, 6, 5, 4], [4, 5, 4, 5, 4, 4],
                                        [5, 5, 5, 5, 6])] + [seed_centers]
    best = (-1.0, seed_centers, None)
    k = 0
    while time.time() - t0 < time_budget:
        base = layouts[k % len(layouts)]
        noise = 0.01 + 0.03 * rng.random()
        if k >= 2 * len(layouts) and best[2] is not None and k % 2:
            base, noise = best[1], 0.02  # basin-hopping around the best so far
        k += 1
        c = np.clip(base + noise * rng.standard_normal((n, 2)), 0.03, 0.97)
        z0 = np.concatenate([c[:, 0], c[:, 1], np.full(n, 0.03)])
        try:
            res = minimize(lambda z: -z[2 * n:].sum(), z0, jac=lambda z: grad, bounds=bounds,
                           constraints=cons, method="SLSQP",
                           options=dict(maxiter=300, ftol=1e-10))
        except Exception:
            continue
        z = res.x
        c = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0, 1)
        # Make strictly feasible: cap radii by walls, then shrink overlapping pairs.
        r = np.maximum(np.minimum(z[2 * n:], np.min(np.hstack([c, 1 - c]), axis=1)), 0)
        d = np.sqrt(((c[iu] - c[ju]) ** 2).sum(1))
        for _ in range(50):
            over = r[iu] + r[ju] - d
            p = np.argmax(over)
            if over[p] <= 0:
                break
            s = d[p] / (r[iu[p]] + r[ju[p]])
            r[iu[p]] *= s
            r[ju[p]] *= s
        r = r * (1 - 1e-9)
        if (r[iu] + r[ju] <= d).all() and r.sum() > best[0]:
            best = (r.sum(), c, r)
    # Final polish: restart SLSQP from the incumbent (centers AND radii) with a
    # tighter tolerance, then repair with a much smaller safety margin.
    if best[2] is not None:
        try:
            z0 = np.concatenate([best[1][:, 0], best[1][:, 1], best[2]])
            res = minimize(lambda z: -z[2 * n:].sum(), z0, jac=lambda z: grad, bounds=bounds,
                           constraints=cons, method="SLSQP",
                           options=dict(maxiter=200, ftol=1e-15))
            z = res.x
            c = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0, 1)
            r = np.maximum(np.minimum(z[2 * n:], np.min(np.hstack([c, 1 - c]), axis=1)), 0)
            d = np.sqrt(((c[iu] - c[ju]) ** 2).sum(1))
            for _ in range(200):
                over = r[iu] + r[ju] - d
                p = np.argmax(over)
                if over[p] <= 0:
                    break
                s = d[p] / (r[iu[p]] + r[ju[p]])
                r[iu[p]] *= s
                r[ju[p]] *= s
            r = r * (1 - 1e-12)
            wall_ok = (r <= np.min(np.hstack([c, 1 - c]), axis=1)).all()
            if wall_ok and (r[iu] + r[ju] <= d).all() and r.sum() > best[0]:
                best = (r.sum(), c, r)
        except Exception:
            pass
    return best[1], best[2]


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
