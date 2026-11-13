# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    n = 26
    rng = np.random.default_rng(0)
    a, b = np.triu_indices(n, 1)

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.hypot(x[a] - x[b], y[a] - y[b])
        return np.concatenate([d - r[a] - r[b], x - r, 1 - x - r, y - r, 1 - y - r])

    def solve(z0):
        return minimize(lambda z: -z[2 * n:].sum(), z0,
                        jac=lambda z: np.r_[np.zeros(2 * n), -np.ones(n)],
                        constraints={"type": "ineq", "fun": cons}, method="SLSQP",
                        bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                        options={"maxiter": 1000, "ftol": 1e-12}).x

    best, bz = -1, None

    def consider(z):
        nonlocal best, bz
        if cons(z).min() > -1e-10 and z[2 * n:].sum() > best:
            best, bz = z[2 * n:].sum(), z

    for t in range(16):
        if t % 2 == 0:
            g = np.array([(0.1 + 0.2 * i + 0.1 * (j % 2), 0.1 + 0.2 * j)
                          for j in range(5) for i in range(5)]
                         + [tuple(rng.uniform(0.1, 0.9, 2))])
            g = g + rng.normal(0, 0.03, (n, 2))
        else:
            g = rng.uniform(0.1, 0.9, (n, 2))
        g = np.clip(g, 0.05, 0.95)
        consider(solve(np.concatenate([g[:, 0], g[:, 1], np.full(n, 0.05)])))

    # Basin-hopping refinement around the incumbent
    for t in range(40):
        z = bz.copy()
        z[:2 * n] += rng.normal(0, 0.02 if t % 2 else 0.05, 2 * n)
        if t % 6 == 0:  # relocate the smallest circle
            k = np.argmin(z[2 * n:])
            z[k], z[n + k] = rng.uniform(0.05, 0.95, 2)
        z[:2 * n] = np.clip(z[:2 * n], 0.02, 0.98)
        z[2 * n:] *= rng.uniform(0.7, 0.9)
        consider(solve(z))
    p = np.stack([bz[:n], bz[n:2 * n]], 1)
    r = np.maximum(bz[2 * n:] - 1e-9, 0)
    return p, r, r.sum()


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
