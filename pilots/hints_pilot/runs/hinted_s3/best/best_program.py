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

    # Multi-start joint optimization of centers and radii (SLSQP).
    import time
    start_time = time.time()
    rng = np.random.default_rng(0)
    seeds = []
    for rows in ([5, 5, 6, 5, 5], [6, 5, 5, 5, 5], [5, 6, 5, 6, 4],
                 [4, 5, 4, 5, 4, 4], [5, 4, 5, 4, 5, 3]):
        pts = []
        for k, m in enumerate(rows):
            pts += [[(i + 0.5) / m, (k + 0.5) / len(rows)] for i in range(m)]
        seeds.append(np.array(pts))
    best_c, best_r = circle_centers, greedy_radii_without_overlap(circle_centers)
    trial = 0
    while time.time() - start_time < 25:
        if time.time() - start_time < 10:
            # Phase 1: multi-start from structured row layouts.
            noise = 0.005 + 0.01 * ((trial // len(seeds)) % 5)
            c = seeds[trial % len(seeds)] + rng.normal(0, noise, (num_circles, 2))
        else:
            # Phase 2: basin hopping - perturb a few circles of the best packing.
            c = best_c.copy()
            idx = rng.choice(num_circles, rng.integers(1, 6), replace=False)
            if trial % 3 == 0:
                c[idx] = rng.uniform(0.05, 0.95, (len(idx), 2))
            else:
                c[idx] += rng.normal(0, 0.05, (len(idx), 2))
        trial += 1
        c = np.clip(c, 0.03, 0.97)
        try:
            c, r = optimize_packing(c, greedy_radii_without_overlap(c) * 0.9)
        except Exception:
            continue
        if np.all(np.isfinite(r)) and r.sum() > best_r.sum():
            best_c, best_r = c, r
    circle_centers, circle_radii = best_c, best_r

    sum_of_radii = np.sum(circle_radii)

    return circle_centers, circle_radii, sum_of_radii


def optimize_packing(centers, radii):
    # Maximize sum of radii over centers and radii with SLSQP (analytic jacobians).
    from scipy.optimize import minimize
    n = len(radii)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    k = np.arange(m)
    a = np.arange(n)

    def cons(v):
        x, y, r = v[:n], v[n:2 * n], v[2 * n:]
        d = np.hypot(x[iu] - x[ju], y[iu] - y[ju])
        return np.concatenate([d - r[iu] - r[ju], x - r, 1 - x - r, y - r, 1 - y - r])

    def jac(v):
        x, y = v[:n], v[n:2 * n]
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        J = np.zeros((m + 4 * n, 3 * n))
        J[k, iu] = dx / d
        J[k, ju] = -dx / d
        J[k, n + iu] = dy / d
        J[k, n + ju] = -dy / d
        J[k, 2 * n + iu] = -1
        J[k, 2 * n + ju] = -1
        for b, (off, s) in enumerate([(0, 1), (0, -1), (n, 1), (n, -1)]):
            J[m + b * n + a, off + a] = s
            J[m + b * n + a, 2 * n + a] = -1
        return J

    g = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    res = minimize(lambda v: -v[2 * n:].sum(),
                   np.concatenate([centers[:, 0], centers[:, 1], radii]),
                   jac=lambda v: g, method="SLSQP",
                   bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                   constraints=dict(type="ineq", fun=cons, jac=jac),
                   options=dict(maxiter=300, ftol=1e-10))
    v = res.x
    c = np.clip(np.column_stack([v[:n], v[n:2 * n]]), 0, 1)
    # Repair: cap radii by walls, then shrink any residual overlaps.
    r = np.minimum(np.maximum(v[2 * n:], 0), np.minimum(c, 1 - c).min(axis=1))
    d = np.hypot(c[iu, 0] - c[ju, 0], c[iu, 1] - c[ju, 1])
    for _ in range(500):
        ov = r[iu] + r[ju] - d
        w = np.argmax(ov)
        if ov[w] <= 0:
            break
        s = d[w] / (r[iu[w]] + r[ju[w]])
        r[iu[w]] *= s
        r[ju[w]] *= s
    return c, r * (1 - 1e-9)


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
