# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def f():
    n = 21
    rng = np.random.default_rng(0)
    p = np.zeros((n, 2))
    k = 0
    for j in range(5):
        for i in range(5 if j < 3 else 3):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1
    bp, bq = p, g(p, np.ones(n))
    best = bq.sum()

    def samp(m):
        out = []
        while len(out) < m:
            u, v = rng.uniform(0.04, 0.96, 2)
            if u < 0.55 or v < 0.55:
                out.append([u, v])
        return np.array(out)

    def tr(p0, r0):
        nonlocal best, bp, bq
        try:
            pp, q = opt(np.concatenate([np.clip(p0, 0.01, 0.99).ravel(), r0]), n)
        except Exception:
            return
        q = g(pp, q)
        if q.sum() > best:
            best, bp, bq = q.sum(), pp.copy(), q

    tr(p + 0.01 * rng.standard_normal((n, 2)), np.full(n, 0.02))
    for t in range(35):
        tr(samp(n), rng.uniform(0.005, 0.05, n))
    # basin hopping: relocate the smallest circles of the incumbent
    for t in range(30):
        p0 = bp + 0.015 * rng.standard_normal((n, 2))
        k = 1 + t % 2
        idx = np.argsort(bq)[:k]
        p0[idx] = samp(k)
        r0 = bq * 0.85
        r0[idx] = 0.01
        tr(p0, r0)
    return bp, bq


def opt(x0, n):
    iu = np.triu_indices(n, 1)

    def cons(x):
        p = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))[iu]
        w = np.hypot(np.maximum(0.6 - p[:, 0], 0), np.maximum(0.6 - p[:, 1], 0))
        return np.concatenate([d - r[iu[0]] - r[iu[1]],
                               p[:, 0] - r, 1 - p[:, 0] - r,
                               p[:, 1] - r, 1 - p[:, 1] - r, w - r])

    bnds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    jac = np.r_[np.zeros(2 * n), -np.ones(n)]
    res = minimize(lambda x: -x[2 * n:].sum(), x0, jac=lambda x: jac,
                   constraints=[{'type': 'ineq', 'fun': cons}], bounds=bnds,
                   method='SLSQP', options={'maxiter': 600, 'ftol': 1e-12})
    return np.clip(res.x[:2 * n].reshape(n, 2), 0, 1), res.x[2 * n:]


def g(p, q):
    n = len(q)
    q = np.clip(np.asarray(q, dtype=float), 0, None).copy()
    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = max(0.0, min(q[i], u, v, 1 - u, 1 - v, w))
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(*(p[i] - p[j]))
                if q[i] + q[j] > d:
                    s = d / (q[i] + q[j]) * (1 - 1e-12)
                    q[i] *= s
                    q[j] *= s
    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
