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

N = 26
IU = np.triu_indices(N, 1)


def _obj(z):
    return -np.sum(z[2 * N:])


def _objg(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1
    return g


def _cons(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    i, j = IU
    pair = np.hypot(x[i] - x[j], y[i] - y[j]) - r[i] - r[j]
    wall = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    return np.concatenate([pair, wall])


def _consj(z):
    x, y = z[:N], z[N:2 * N]
    i, j = IU
    dx, dy = x[i] - x[j], y[i] - y[j]
    d = np.hypot(dx, dy) + 1e-12
    M = len(i)
    J = np.zeros((M + 4 * N, 3 * N))
    k = np.arange(M)
    J[k, i], J[k, j] = dx / d, -dx / d
    J[k, N + i], J[k, N + j] = dy / d, -dy / d
    J[k, 2 * N + i] = -1
    J[k, 2 * N + j] = -1
    a = np.arange(N)
    for w, (off, sgn) in enumerate([(0, 1), (0, -1), (N, 1), (N, -1)]):
        J[M + w * N + a, off + a] = sgn
        J[M + w * N + a, 2 * N + a] = -1
    return J


def _solve(c, rng, noise, r0=None):
    c = np.clip(c + rng.normal(0, noise, c.shape), 0.03, 0.97)
    r0 = np.full(N, 0.02) if r0 is None else np.clip(r0, 0.0, 0.5)
    z0 = np.concatenate([c[:, 0], c[:, 1], r0])
    bnds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    res = minimize(_obj, z0, jac=_objg, method="SLSQP", bounds=bnds,
                   constraints={"type": "ineq", "fun": _cons, "jac": _consj},
                   options={"maxiter": 800, "ftol": 1e-12})
    return res.x


def _seeds(rng):
    seeds = []
    for rows in ([5, 5, 6, 5, 5], [5, 6, 5, 6, 4], [6, 5, 4, 5, 6], [4, 5, 4, 5, 4, 4]):
        pts = []
        for ri, cnt in enumerate(rows):
            for c in range(cnt):
                pts.append([(c + 0.5 + 0.25 * (ri % 2)) / (cnt + 0.25),
                            (ri + 0.5) / len(rows)])
        seeds.append(np.array(pts[:N]))
    # ring seed: 1 + 8 + 17
    pts = [[0.5, 0.5]]
    pts += [[0.5 + 0.24 * np.cos(2 * np.pi * k / 8),
             0.5 + 0.24 * np.sin(2 * np.pi * k / 8)] for k in range(8)]
    pts += [[0.5 + 0.44 * np.cos(2 * np.pi * k / 17),
             0.5 + 0.44 * np.sin(2 * np.pi * k / 17)] for k in range(17)]
    seeds.append(np.clip(np.array(pts), 0.05, 0.95))
    # 5x5 grid plus one extra circle in an interstice
    g = (np.arange(5) + 0.5) / 5
    pts = [[a, b] for a in g for b in g] + [[0.4, 0.4]]
    seeds.append(np.array(pts))
    for _ in range(10):
        seeds.append(rng.uniform(0.1, 0.9, (N, 2)))
    return seeds


def construct_packing():
    rng = np.random.default_rng(0)
    best, best_s = None, -1.0

    def consider(z):
        nonlocal best, best_s
        s = z[2 * N:].sum()
        if _cons(z).min() > -1e-9 and s > best_s:
            best, best_s = z, s

    import time
    t0 = time.time()
    for s in _seeds(rng):
        consider(_solve(s, rng, 0.01))
    stall = 0
    for it in range(400):  # basin hopping around the incumbent
        if time.time() - t0 > 12.0:
            break
        prev = best_s
        c = np.stack([best[:N], best[N:2 * N]], 1).copy()
        r = best[2 * N:].copy()
        mode = it % 3
        if stall > 25:  # big kick: relocate several small circles
            k = int(rng.integers(4, 8))
            idx = np.argsort(r)[:k]
            c[idx] = rng.uniform(0.05, 0.95, (k, 2))
            r[idx] = max(0.01, 0.5 * np.median(r))
            consider(_solve(c, rng, 0.01, r * 0.9))
            stall = 0 if best_s > prev + 1e-9 else stall - 10
            continue
        if mode == 0:  # global jitter, warm-started radii
            consider(_solve(c, rng, rng.uniform(0.01, 0.05), r * 0.9))
        elif mode == 1:  # relocate the 1-3 smallest circles
            k = rng.integers(1, 4)
            idx = np.argsort(r)[:k]
            c[idx] = rng.uniform(0.05, 0.95, (k, 2))
            r[idx] = max(0.01, 0.5 * np.median(r))
            consider(_solve(c, rng, 0.005, r * 0.95))
        else:  # swap a small and a large circle
            order = np.argsort(r)
            a = order[rng.integers(0, 6)]
            b = order[-1 - rng.integers(0, 6)]
            c[[a, b]] = c[[b, a]]
            r[[a, b]] = r[[b, a]]
            consider(_solve(c, rng, 0.005, r * 0.9))
        stall = 0 if best_s > prev + 1e-9 else stall + 1

    centers = np.stack([best[:N], best[N:2 * N]], 1)
    radii = np.maximum(best[2 * N:] - 1e-10, 0)
    return centers, radii, float(radii.sum())


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
