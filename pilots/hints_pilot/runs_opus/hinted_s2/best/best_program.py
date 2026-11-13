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
from scipy.optimize import minimize


def construct_packing():
    # Joint SLSQP optimization of centers + radii from multiple seeds.
    n = 26
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    I = np.arange(n)
    k = 4 * n + np.arange(m)

    def obj(z):
        return -z[2 * n:].sum()

    def gobj(z):
        g = np.zeros(3 * n)
        g[2 * n:] = -1
        return g

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.hypot(x[iu] - x[ju], y[iu] - y[ju])
        return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r, d - r[iu] - r[ju]])

    def jac(z):
        x, y = z[:n], z[n:2 * n]
        J = np.zeros((4 * n + m, 3 * n))
        J[I, I] = 1
        J[n + I, I] = -1
        J[2 * n + I, n + I] = 1
        J[3 * n + I, n + I] = -1
        for b in range(4):
            J[b * n + I, 2 * n + I] = -1
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        d = np.hypot(dx, dy) + 1e-12
        J[k, iu] = dx / d
        J[k, ju] = -dx / d
        J[k, n + iu] = dy / d
        J[k, n + ju] = -dy / d
        J[k, 2 * n + iu] = -1
        J[k, 2 * n + ju] = -1
        return J

    rng = np.random.default_rng(0)
    seeds = []
    for rows in ([5, 5, 6, 5, 5], [6, 5, 4, 5, 6], [5, 6, 5, 6, 4], [4, 5, 4, 5, 4, 4]):
        pts = []
        for i, c in enumerate(rows):
            for j in range(c):
                pts.append([(j + 0.5 + 0.25 * (i % 2)) / (c + 0.25), (i + 0.5) / len(rows)])
        seeds.append(np.array(pts[:n]))
    for _ in range(12):
        seeds.append(rng.uniform(0.1, 0.9, (n, 2)))

    best, bz = -1, None
    bnds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n

    def solve(z0):
        try:
            res = minimize(obj, z0, jac=gobj, method="SLSQP", bounds=bnds,
                           constraints={"type": "ineq", "fun": cons, "jac": jac},
                           options={"maxiter": 600, "ftol": 1e-12})
        except Exception:
            return -1, None
        if cons(res.x).min() > -1e-7:
            return -res.fun, res.x.copy()
        return -1, None

    for si, s in enumerate(seeds):
        for _ in range(2):
            p = np.clip(s + rng.normal(0, 0.02, s.shape), 0.05, 0.95)
            r0 = np.full(n, 0.05) if si < 4 else rng.uniform(0.03, 0.12, n)
            v, z = solve(np.concatenate([p[:, 0], p[:, 1], r0]))
            if v > best:
                best, bz = v, z

    # Basin hopping: perturb a subset of centers of the incumbent and re-solve.
    import time
    t0 = time.time()
    if bz is not None:
        for it in range(120):
            if time.time() - t0 > 25:
                break
            z = bz.copy()
            step = 0.03 if it % 2 else 0.08
            mask = rng.random(2 * n) < 0.3
            z[:2 * n] = np.clip(z[:2 * n] + rng.normal(0, step, 2 * n) * mask, 0.02, 0.98)
            z[2 * n:] *= 0.9
            v, z = solve(z)
            if v > best + 1e-9:
                best, bz = v, z

    if bz is None:
        c = np.clip(seeds[0], 0.01, 0.99)
        r = greedy_radii_without_overlap(c)
        return c, r, r.sum()

    c = np.stack([bz[:n], bz[n:2 * n]], 1)
    r = np.maximum(bz[2 * n:] - 1e-9, 0)
    r = np.minimum(r, np.min(np.stack([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]]), 0))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(c[i] - c[j])
            if r[i] + r[j] > d:
                f = d / (r[i] + r[j])
                r[i] *= f
                r[j] *= f
    return c, r, float(np.sum(r))


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
