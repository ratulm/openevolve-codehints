"""Offline max-min dispersion solver for the annulus instance (not shown to the LLM)."""
import numpy as np
from scipy.optimize import minimize

N = 18
R_IN, R_OUT = 0.4, 1.0
KX, GY = 3.0, 0.4  # true metric: sqrt((KX*dx)^2 + dy^2) * (1 + GY * mean_y)
IU = np.triu_indices(N, 1)


def true_dist(p):
    dx = p[IU[0], 0] - p[IU[1], 0]
    dy = p[IU[0], 1] - p[IU[1], 1]
    my = 0.5 * (p[IU[0], 1] + p[IU[1], 1])
    return np.sqrt((KX * dx) ** 2 + dy**2) * (1.0 + GY * my)


def euclid_dist(p):
    return np.hypot(p[IU[0], 0] - p[IU[1], 0], p[IU[0], 1] - p[IU[1], 1])


def clip(p):
    r = np.hypot(p[:, 0], p[:, 1])
    s = np.clip(r, R_IN + 1e-12, R_OUT - 1e-12) / np.maximum(r, 1e-300)
    return p * s[:, None]


def solve(dist, starts, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    cons = [
        {"type": "ineq", "fun": lambda z: dist(z[:-1].reshape(N, 2)) ** 2 - z[-1] ** 2},
        {"type": "ineq", "fun": lambda z: np.sum(z[:-1].reshape(N, 2) ** 2, axis=1) - R_IN**2},
        {"type": "ineq", "fun": lambda z: R_OUT**2 - np.sum(z[:-1].reshape(N, 2) ** 2, axis=1)},
    ]
    best_p, best_v = None, -np.inf
    for k in range(starts):
        a = rng.uniform(0, 2 * np.pi, N)
        r = np.sqrt(rng.uniform(R_IN**2, R_OUT**2, N))
        p0 = np.c_[r * np.cos(a), r * np.sin(a)]
        z0 = np.concatenate([p0.ravel(), [dist(p0).min()]])
        res = minimize(lambda z: -z[-1], z0, jac=lambda z: np.r_[np.zeros(2 * N), -1.0],
                       method="SLSQP", constraints=cons, options={"maxiter": 400, "ftol": 1e-12})
        p = clip(res.x[:-1].reshape(N, 2))
        v = dist(p).min()
        if v > best_v:
            best_p, best_v = p, v
            if verbose:
                print(k, v, flush=True)
    return best_p, best_v
