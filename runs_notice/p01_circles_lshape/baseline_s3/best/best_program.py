# EVOLVE-BLOCK-START
import numpy as np


def base():
    n = 21
    p = np.zeros((n, 2))
    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1
    return p


def wd(P):
    return np.sqrt(np.maximum(0.6 - P[:, 0], 0) ** 2 + np.maximum(0.6 - P[:, 1], 0) ** 2 + 1e-18)


def starts(n, rng):
    p0 = base()
    out = [p0 * 1.6, p0 * 1.3]
    for _ in range(14):
        s = p0 * rng.uniform(0.9, 1.6) + rng.normal(0, 0.03, p0.shape)
        out.append(np.clip(s, 0.02, 0.98))
    return out


def h(p, n):
    from scipy.optimize import minimize
    I, J = np.triu_indices(n, 1)

    def c(x):
        P = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        d = np.sqrt(((P[I] - P[J]) ** 2).sum(1) + 1e-18)
        return np.concatenate([d - r[I] - r[J], P[:, 0] - r, P[:, 1] - r,
                               1 - P[:, 0] - r, 1 - P[:, 1] - r, wd(P) - r])

    x0 = np.concatenate([p.ravel(), np.full(n, 0.03)])
    b = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    jf = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    o = minimize(lambda x: -np.sum(x[2 * n:]), x0, jac=lambda x: jf,
                 bounds=b, constraints=[dict(type='ineq', fun=c)],
                 method='SLSQP', options=dict(maxiter=800, ftol=1e-12))
    return o.x


def f():
    n = 21
    p0 = base()
    best_p, best_q = p0, g(p0, None)
    try:
        rng = np.random.default_rng(12345)
        for s in starts(n, rng):
            try:
                x = h(s, n)
            except Exception:
                continue
            P = np.clip(x[:2 * n].reshape(n, 2), 0.0, 1.0)
            Q = g(P, x[2 * n:])
            if np.sum(Q) > np.sum(best_q):
                best_p, best_q = P, Q
    except Exception:
        pass
    return best_p, best_q


def g(p, r=None):
    n = p.shape[0]
    q = np.minimum.reduce([p[:, 0], p[:, 1], 1 - p[:, 0], 1 - p[:, 1], wd(p)])
    if r is not None:
        q = np.minimum(q, r)
    q = np.maximum(q - 1e-10, 0.0)
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((p[i] - p[j]) ** 2))
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
