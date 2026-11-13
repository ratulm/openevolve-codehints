# EVOLVE-BLOCK-START
import numpy as np
import itertools


def _areas(p, T):
    a, b, c = p[T[:, 0]], p[T[:, 1]], p[T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _climb(p, T, rng, iters=1500):
    p = g(p)
    cur = np.abs(_areas(p, T)).min()
    n = len(p)
    for it in range(iters):
        step = max(0.002, 0.08 * (1 - it / iters))
        q = p.copy()
        a = np.abs(_areas(q, T))
        if rng.random() < 0.7:
            i = T[np.argmin(a)][rng.integers(3)]
        else:
            i = rng.integers(n)
        q[i] += rng.normal(0, step, 2)
        q = g(q)
        m = np.abs(_areas(q, T)).min()
        if m >= cur:
            p, cur = q, m
    return p, cur


def _polish(p, T):
    try:
        from scipy.optimize import minimize
    except Exception:
        return p
    n = len(p)
    for _ in range(4):
        s = np.sign(_areas(p, T))
        s[s == 0] = 1
        x0 = np.r_[p.ravel(), np.abs(_areas(p, T)).min()]

        def c1(x, s=s):
            return s * _areas(x[:-1].reshape(n, 2), T) - x[-1]

        def c2(x):
            u, v = x[0:-1:2], x[1:-1:2]
            return np.r_[v, 0.5 - v, u - 0.6 * v, 1 - 0.2 * v - u]

        try:
            r = minimize(lambda x: -x[-1], x0, method='SLSQP',
                         constraints=[dict(type='ineq', fun=c1),
                                      dict(type='ineq', fun=c2)],
                         options=dict(maxiter=500, ftol=1e-13))
        except Exception:
            break
        q = g(r.x[:-1].reshape(n, 2))
        if np.abs(_areas(q, T)).min() > np.abs(_areas(p, T)).min() + 1e-12:
            p = q
        else:
            break
    return p


def f():
    n = 13
    T = np.array(list(itertools.combinations(range(n), 3)))
    rng = np.random.default_rng(12345)
    best, bv = None, -1.0
    for _ in range(30):
        s = np.c_[rng.uniform(0, 1, n), rng.uniform(0, 0.5, n)]
        p, _v = _climb(s, T, rng)
        p = _polish(p, T)
        v = np.abs(_areas(p, T)).min()
        if v > bv:
            best, bv = p, v
    return g(best)


def g(p):
    q = np.array(p, dtype=float)
    q[:, 1] = np.clip(q[:, 1], 0.0, 0.5)
    q[:, 0] = np.clip(q[:, 0], 0.6 * q[:, 1], 1.0 - 0.2 * q[:, 1])
    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
