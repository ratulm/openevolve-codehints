# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def f():
    n = 13
    T = np.array(list(combinations(range(n), 3)))

    def areas(q):
        a, b, c = q[T[:, 0]], q[T[:, 1]], q[T[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def score(q):
        return np.min(np.abs(areas(q)))

    def polish(q):
        # maximize t s.t. sign_i * area_i >= t, points inside region
        try:
            from scipy.optimize import minimize
            sg = np.sign(areas(q))
            sg[sg == 0] = 1
            x0 = np.append(q.ravel(), score(q))
            cons = [
                {'type': 'ineq', 'fun': lambda x: sg * areas(x[:-1].reshape(n, 2)) - x[-1]},
                {'type': 'ineq', 'fun': lambda x: x[1:-1:2]},
                {'type': 'ineq', 'fun': lambda x: 0.5 - x[1:-1:2]},
                {'type': 'ineq', 'fun': lambda x: x[0:-1:2] - 0.6 * x[1:-1:2]},
                {'type': 'ineq', 'fun': lambda x: 1 - 0.2 * x[1:-1:2] - x[0:-1:2]},
            ]
            r = minimize(lambda x: -x[-1], x0, method='SLSQP', constraints=cons,
                         options={'maxiter': 400, 'ftol': 1e-12})
            c = g(r.x[:-1].reshape(n, 2))
            return c if score(c) > score(q) else q
        except Exception:
            return q

    def climb(p, rng, iters=3000):
        cs, s = score(p), 0.05
        for it in range(iters):
            c = p.copy()
            k = rng.integers(n)
            c[k] += rng.normal(0, s, 2)
            c = g(c)
            sc = score(c)
            if sc >= cs:
                p, cs = c, sc
            if it % 500 == 499:
                s *= 0.6
        return p

    rng = np.random.default_rng(0)
    p0 = np.zeros((n, 2))
    for i in range(n):
        t = 2 * np.pi * i / n
        p0[i] = [0.52 + 0.2 * np.cos(t), 0.25 + 0.12 * np.sin(t)]
    starts = [g(p0)] + [g(np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 0.5, n)]))
                        for _ in range(40)]

    best, bs = None, -1.0
    for p in starts:
        p = polish(climb(p, rng))
        sc = score(p)
        if sc > bs:
            best, bs = p, sc
    best = polish(climb(best, rng, 4000))
    return best


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
