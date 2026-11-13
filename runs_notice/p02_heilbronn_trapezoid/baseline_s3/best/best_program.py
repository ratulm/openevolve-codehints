# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 13
    p = np.zeros((n, 2))

    for i in range(n):
        t = 2 * np.pi * i / n
        p[i] = [0.56 + 0.375 * np.cos(t), 0.25 + 0.225 * np.sin(t)]

    p = g(p)
    try:
        p = h(p)
    except Exception:
        pass
    return g(p)


def _areas(q, T):
    a, b, c = q[T[:, 0]], q[T[:, 1]], q[T[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def h(p0):
    import itertools
    from scipy.optimize import minimize

    n = p0.shape[0]
    T = np.array(list(itertools.combinations(range(n), 3)))
    best = g(p0)
    bs = _areas(best, T).min()
    rng = np.random.default_rng(1)

    def box(x):
        xs, ys = x[0:2 * n:2], x[1:2 * n:2]
        return np.r_[ys, 0.5 - ys, xs - 0.6 * ys, 1.0 - 0.2 * ys - xs]

    cons = [dict(type='ineq',
                 fun=lambda x: _areas(x[:2 * n].reshape(n, 2), T) - x[-1]),
            dict(type='ineq', fun=box)]

    for r in range(12):
        q = best if r == 0 else g(best + rng.normal(0, 0.02, best.shape))
        x0 = np.r_[q.ravel(), _areas(q, T).min()]
        res = minimize(lambda x: -x[-1], x0, method='SLSQP',
                       constraints=cons, options=dict(maxiter=300, ftol=1e-12))
        qq = g(res.x[:2 * n].reshape(n, 2))
        s = _areas(qq, T).min()
        if s > bs:
            bs, best = s, qq
    return best


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
