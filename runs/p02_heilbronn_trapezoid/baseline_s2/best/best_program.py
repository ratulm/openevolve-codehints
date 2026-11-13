# EVOLVE-BLOCK-START
import numpy as np


from itertools import combinations

_T = None


def area(p):
    a, b, c = p[_T[:, 0]], p[_T[:, 1]], p[_T[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def f():
    global _T
    n = 13
    _T = np.array(list(combinations(range(n), 3)))
    rng = np.random.default_rng(1)

    p0 = np.zeros((n, 2))
    for i in range(n):
        t = 2 * np.pi * i / n
        p0[i] = [0.52 + 0.2 * np.cos(t), 0.25 + 0.12 * np.sin(t)]

    def sa(x):
        p = x[:2 * n].reshape(n, 2)
        a, b, c = p[_T[:, 0]], p[_T[:, 1]], p[_T[:, 2]]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def box(x):
        u, v = x[0:2 * n:2], x[1:2 * n:2]
        return np.r_[v, 0.5 - v, u - 0.6 * v, 1.0 - 0.2 * v - u]

    def polish(p):
        try:
            from scipy.optimize import minimize
        except Exception:
            return p
        for _ in range(4):
            try:
                s = np.sign(sa(p.ravel()))
                s[s == 0] = 1.0
                x0 = np.r_[p.ravel(), area(p).min()]
                cons = [dict(type='ineq', fun=lambda x, s=s: s * sa(x) - x[-1]),
                        dict(type='ineq', fun=box)]
                r = minimize(lambda x: -x[-1], x0, method='SLSQP',
                             jac=lambda x: np.r_[np.zeros(2 * n), -1.0],
                             constraints=cons,
                             options=dict(maxiter=800, ftol=1e-15))
                q = g(r.x[:2 * n].reshape(n, 2))
            except Exception:
                break
            if np.all(np.isfinite(q)) and area(q).min() > area(p).min() + 1e-13:
                p = q
            else:
                break
        return p

    starts = [g(p0)]
    for _ in range(13):
        v = rng.uniform(0, 0.5, n)
        u = 0.6 * v + rng.uniform(0, 1, n) * (1 - 0.8 * v)
        starts.append(g(np.c_[u, v]))

    best, bval = starts[0], area(starts[0]).min()
    for p in starts:
        p = p.copy()
        A = area(p)
        cur = A.min()
        step = 0.08
        for it in range(3000):
            k = _T[np.argmin(A)][rng.integers(3)] if rng.random() < 0.7 else rng.integers(n)
            q = p.copy()
            q[k] += rng.normal(0, step, 2)
            q = g(q)
            Aq = area(q)
            m = Aq.min()
            if m >= cur:
                p, A, cur = q, Aq, m
            step = max(0.002, step * 0.9995)
        p = polish(p)
        cur = area(p).min()
        if cur > bval:
            best, bval = p, cur
    for _ in range(3):
        best = polish(best)
    return g(best)


def g(p):
    q = np.array(p, dtype=float)

    for i in range(q.shape[0]):
        v = min(max(q[i, 1], 0.0), 0.5)
        u = min(max(q[i, 0], 0.6 * v), 1.0 - 0.2 * v)
        q[i] = [u, v]

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
