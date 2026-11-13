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
I, J = np.triu_indices(N, 1)


def _cons(z):
    x, y, r = z[:N], z[N:2 * N], z[2 * N:]
    d = np.sqrt((x[I] - x[J]) ** 2 + (y[I] - y[J]) ** 2)
    return np.concatenate([d - r[I] - r[J], x - r, 1 - x - r, y - r, 1 - y - r])


def _jac(z):
    x, y = z[:N], z[N:2 * N]
    M = len(I)
    A = np.zeros((M + 4 * N, 3 * N))
    k = np.arange(M)
    dx, dy = x[I] - x[J], y[I] - y[J]
    d = np.sqrt(dx * dx + dy * dy) + 1e-12
    A[k, I] = dx / d
    A[k, J] = -dx / d
    A[k, N + I] = dy / d
    A[k, N + J] = -dy / d
    A[k, 2 * N + I] = -1
    A[k, 2 * N + J] = -1
    e = np.arange(N)
    for t, (c, s) in enumerate([(0, 1), (0, -1), (N, 1), (N, -1)]):
        A[M + t * N + e, c + e] = s
        A[M + t * N + e, 2 * N + e] = -1
    return A


def _seeds(rng):
    yield rng.uniform(0.1, 0.9, (N, 2))
    for rows in ([5, 5, 6, 5, 5], [6, 5, 4, 5, 6], [5, 6, 4, 6, 5],
                 [4, 5, 4, 5, 4, 4], [6, 5, 5, 5, 5]):
        pts = []
        for i, n in enumerate(rows):
            for j in range(n):
                pts.append([(j + 0.5 + 0.25 * (i % 2)) / (n + 0.25), (i + 0.5) / len(rows)])
        yield np.array(pts[:N])
    a8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    a17 = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    c = [[0.5, 0.5]]
    c += [[0.5 + 0.25 * np.cos(a), 0.5 + 0.25 * np.sin(a)] for a in a8]
    c += [[0.5 + 0.45 * np.cos(a), 0.5 + 0.45 * np.sin(a)] for a in a17]
    yield np.clip(np.array(c), 0.05, 0.95)


def _solve(p, rng, r0=None, sigma=0.03):
    p = np.clip(p + rng.normal(0, sigma, p.shape), 0.02, 0.98)
    r0 = np.full(N, 0.05) if r0 is None else r0
    z0 = np.concatenate([p[:, 0], p[:, 1], r0])
    grad = np.r_[np.zeros(2 * N), -np.ones(N)]
    res = minimize(lambda z: -z[2 * N:].sum(), z0, jac=lambda z: grad,
                   method="SLSQP", bounds=[(0, 1)] * (2 * N) + [(0, 0.5)] * N,
                   constraints={"type": "ineq", "fun": _cons, "jac": _jac},
                   options={"maxiter": 600, "ftol": 1e-12})
    z = res.x
    return np.c_[z[:N], z[N:2 * N]], z[2 * N:]


def _fix(c, r):
    # Guarantee strict validity: clip to walls, shrink any residual overlaps.
    r = np.minimum.reduce([np.maximum(r, 0), c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    for _ in range(50):
        d = np.linalg.norm(c[I] - c[J], axis=1)
        o = r[I] + r[J] - d
        if (o <= 0).all():
            break
        for k in np.where(o > 0)[0]:
            s = d[k] / (r[I[k]] + r[J[k]])
            r[I[k]] *= s
            r[J[k]] *= s
    return r * (1 - 1e-9)


def construct_packing():
    rng = np.random.default_rng(0)
    best = (None, None, -1.0)
    for _ in range(8):
        for p in _seeds(rng):
            c, r = _solve(p, rng)
            r = _fix(c, r)
            if r.sum() > best[2]:
                best = (c, r, float(r.sum()))
    # Basin-hopping refinement around the incumbent.
    for t in range(40):
        sigma = (0.005, 0.01, 0.02, 0.04)[t % 4]
        c, r = _solve(best[0], rng, r0=best[1] * 0.9, sigma=sigma)
        r = _fix(c, r)
        if r.sum() > best[2]:
            best = (c, r, float(r.sum()))
    return best





# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
