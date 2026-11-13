# EVOLVE-BLOCK-START
import numpy as np


from itertools import combinations

N = 13
T = np.array(list(combinations(range(N), 3)))


def _proj(p):
    v = np.clip(p[:, 1], 0.0, 0.5)
    u = np.clip(p[:, 0], 0.6 * v, 1.0 - 0.2 * v)
    return np.stack([u, v], 1)


def _areas(p):
    a, b, c = p[T[:, 0]], p[T[:, 1]], p[T[:, 2]]
    return 0.5 * ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])
                  - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0]))


def _polish(p):
    # Epigraph formulation: maximize t s.t. sign_k * area_k >= t, p feasible
    try:
        from scipy.optimize import minimize
        sg = np.sign(_areas(p))
        sg[sg == 0] = 1
        x0 = np.r_[p.ravel(), np.abs(_areas(p)).min()]

        def c1(x):
            return sg * _areas(x[:-1].reshape(N, 2)) - x[-1]

        def c2(x):
            u, v = x[0:-1:2], x[1:-1:2]
            return np.r_[v, 0.5 - v, u - 0.6 * v, 1.0 - 0.2 * v - u]

        r = minimize(lambda x: -x[-1], x0, method='SLSQP',
                     constraints=[dict(type='ineq', fun=c1),
                                  dict(type='ineq', fun=c2)],
                     options=dict(maxiter=400, ftol=1e-13))
        q = _proj(r.x[:-1].reshape(N, 2))
        if np.isfinite(q).all():
            return q
    except Exception:
        pass
    return p


def f():
    n = N
    score = lambda p: np.abs(_areas(p)).min()
    rng = np.random.default_rng(0)
    best, bs = None, -1.0
    for _ in range(30):
        p = _proj(rng.random((n, 2)) * [1.0, 0.5])
        s = score(p)
        step = 0.08
        for it in range(4000):
            i = rng.integers(n)
            q = p.copy()
            q[i] += rng.normal(0, step, 2)
            q = _proj(q)
            sq = score(q)
            if sq >= s:
                p, s = q, sq
            if it % 700 == 699:
                step *= 0.6
        for _k in range(2):
            q = _polish(p)
            sq = score(q)
            if sq > s:
                p, s = q, sq
        if s > bs:
            best, bs = p, s

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
