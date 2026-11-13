# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 21
    p = np.zeros((n, 2))
    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1
    q0 = np.full(n, 0.5)

    try:
        from scipy.optimize import minimize
    except ImportError:
        return p, g(p, q0)

    iu, ju = np.triu_indices(n, 1)

    def cons(x):
        P = x[:2 * n].reshape(n, 2)
        R = x[2 * n:]
        d = ((P[iu] - P[ju]) ** 2).sum(1) - (R[iu] + R[ju]) ** 2
        a = np.maximum(0.6 - P[:, 0], 0.0)
        b = np.maximum(0.6 - P[:, 1], 0.0)
        return np.concatenate([d, P[:, 0] - R, P[:, 1] - R,
                               1 - P[:, 0] - R, 1 - P[:, 1] - R,
                               a * a + b * b - R * R])

    jac = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    rng = np.random.default_rng(0)
    best, bs = None, -1.0
    starts = [p.copy()]
    for t in range(20):
        s = rng.uniform(0.02, 0.98, (n, 2))
        m = (s[:, 0] > 0.6) & (s[:, 1] > 0.6)
        s[m] *= 0.5
        starts.append(s)
    for s in starts:
        x0 = np.concatenate([s.ravel(), np.full(n, 0.02)])
        try:
            res = minimize(lambda x: -x[2 * n:].sum(), x0, jac=lambda x: jac,
                           constraints=[dict(type="ineq", fun=cons)],
                           bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                           method="SLSQP",
                           options=dict(maxiter=600, ftol=1e-12))
        except Exception:
            continue
        x = res.x
        if cons(x).min() > -1e-8 and -res.fun > bs:
            bs, best = -res.fun, x

    if best is None:
        return p, g(p, q0)
    p = best[:2 * n].reshape(n, 2).copy()
    return p, g(p, best[2 * n:].copy())


def g(p, q):
    n = p.shape[0]
    q = np.clip(q - 1e-9, 0, None)

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = max(0.0, min(q[i], u, v, 1 - u, 1 - v, w))

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

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
