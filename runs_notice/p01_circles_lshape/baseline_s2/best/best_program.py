# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def f():
    n = 21
    best = None
    rng = np.random.default_rng(0)
    for t in range(14):
        if t == 0:
            p = []
            for j in range(5):
                m = 5 if j < 3 else 3
                for i in range(m):
                    p.append([0.06 + 0.12 * i, 0.06 + 0.12 * j])
            p = np.array(p[:n])
        else:
            p = []
            while len(p) < n:
                x, y = rng.uniform(0.04, 0.96, 2)
                if x < 0.56 or y < 0.56:
                    p.append([x, y])
            p = np.array(p)
        try:
            z, s = h(p)
        except Exception:
            continue
        if best is None or s > best[1]:
            best = (z, s)
    z = best[0]
    return z[:, :2].copy(), z[:, 2].copy()


def h(p):
    n = len(p)
    iu = np.triu_indices(n, 1)
    x0 = np.hstack([p, np.full((n, 1), 0.02)]).ravel()

    def c(z):
        z = z.reshape(n, 3)
        a, b, r = z[:, 0], z[:, 1], z[:, 2]
        d = np.sqrt((a[:, None] - a) ** 2 + (b[:, None] - b) ** 2 + 1e-18)
        w = np.hypot(np.maximum(0.6 - a, 0), np.maximum(0.6 - b, 0))
        return np.concatenate([a - r, b - r, 1 - a - r, 1 - b - r, w - r,
                               (d - r[:, None] - r)[iu]])

    try:
        res = minimize(lambda z: -z[2::3].sum(), x0,
                       jac=lambda z: np.tile([0, 0, -1.0], n),
                       method="SLSQP", constraints={"type": "ineq", "fun": c},
                       bounds=[(0, 1), (0, 1), (0, 0.5)] * n,
                       options={"maxiter": 500, "ftol": 1e-12})
        z = res.x.reshape(n, 3).copy()
    except Exception:
        z = x0.reshape(n, 3).copy()
        z[:, 2] = 0.01
    return g(z)


def g(z):
    n = len(z)
    z[:, :2] = np.clip(z[:, :2], 0, 1)
    a, b = z[:, 0], z[:, 1]
    w = np.hypot(np.maximum(0.6 - a, 0), np.maximum(0.6 - b, 0))
    z[:, 2] = np.clip(z[:, 2], 0, np.minimum.reduce([a, b, 1 - a, 1 - b, w]))
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(*(z[i, :2] - z[j, :2]))
                if z[i, 2] + z[j, 2] > d:
                    s = d / (z[i, 2] + z[j, 2])
                    z[i, 2] *= s
                    z[j, 2] *= s
    z[:, 2] *= 1 - 1e-9
    return z, z[:, 2].sum()


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
