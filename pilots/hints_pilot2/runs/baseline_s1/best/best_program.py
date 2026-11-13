# EVOLVE-BLOCK-START
import numpy as np


def wall(p):
    # distance from each center to the forbidden corner region [0.6,1]^2
    a = np.maximum(0.6 - p[:, 0], 0)
    b = np.maximum(0.6 - p[:, 1], 0)
    return np.sqrt(a * a + b * b + 1e-18)


def f():
    n = 21
    p0 = np.zeros((n, 2))
    k = 0
    for j in range(5):
        for i in range(5 if j < 3 else 3):
            p0[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1
    bp, bq = p0, g(p0, np.full(n, 0.06))
    try:
        from scipy.optimize import minimize
    except Exception:
        return bp, bq

    iu, ju = np.triu_indices(n, 1)

    def cons(z):
        p, r = z[:2 * n].reshape(n, 2), z[2 * n:]
        d = np.sqrt(((p[iu] - p[ju]) ** 2).sum(1) + 1e-18)
        return np.concatenate([d - r[iu] - r[ju], p[:, 0] - r, p[:, 1] - r,
                               1 - p[:, 0] - r, 1 - p[:, 1] - r, wall(p) - r])

    obj = lambda z: -z[2 * n:].sum()
    jac = lambda z: np.r_[np.zeros(2 * n), -np.ones(n)]
    rng = np.random.default_rng(0)
    starts = [p0, p0[:, ::-1].copy()]
    while len(starts) < 30:
        s = rng.uniform(0.05, 0.95, (4 * n, 2))
        s = s[wall(s) > 0.05][:n]
        if len(s) == n:
            starts.append(s)
    for s in starts:
        z0 = np.r_[s.ravel(), np.full(n, 0.02)]
        try:
            res = minimize(obj, z0, jac=jac, method="SLSQP",
                           bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                           constraints={"type": "ineq", "fun": cons},
                           options={"maxiter": 500, "ftol": 1e-12})
        except Exception:
            continue
        p = np.clip(res.x[:2 * n].reshape(n, 2), 0, 1)
        q = g(p, res.x[2 * n:].copy())
        if q.sum() > bq.sum():
            bp, bq = p, q
    return bp, bq


def g(p, q):
    # repair: enforce walls, forbidden corner, and pairwise non-overlap
    q = np.minimum.reduce([q - 1e-9, p[:, 0], p[:, 1], 1 - p[:, 0],
                           1 - p[:, 1], wall(p)])
    q = np.maximum(q, 0)
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            d = np.hypot(*(p[i] - p[j]))
            if q[i] + q[j] > d:
                s = d / (q[i] + q[j])
                q[i] *= s
                q[j] *= s
    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
