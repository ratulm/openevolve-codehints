# EVOLVE-BLOCK-START
import numpy as np


def f():
    import itertools
    n = 13
    T = np.array(list(itertools.combinations(range(n), 3)))

    def s(p):
        a, b, c = p[T[:, 0]], p[T[:, 1]], p[T[:, 2]]
        return np.min(np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))) / 2

    I, J, K = T.T

    def A(p):
        a, b, c = p[I], p[J], p[K]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2

    def polish(p):
        # Exact local max-min via SLSQP: maximise t s.t. sign*area >= t
        try:
            from scipy.optimize import minimize
            sg = np.sign(A(p))
            sg[sg == 0] = 1

            def cons(z):
                q = z[:-1].reshape(n, 2)
                x, y = q[:, 0], q[:, 1]
                return np.r_[sg * A(q) - z[-1], y, 0.5 - y,
                             x - 0.6 * y, 1 - 0.2 * y - x]

            z0 = np.r_[p.ravel(), s(p)]
            res = minimize(lambda z: -z[-1], z0,
                           jac=lambda z: np.r_[np.zeros(2 * n), -1.0],
                           constraints=[{'type': 'ineq', 'fun': cons}],
                           method='SLSQP',
                           options={'maxiter': 400, 'ftol': 1e-12})
            q = g(res.x[:-1].reshape(n, 2))
            return q if s(q) > s(p) else p
        except Exception:
            return p

    def climb(p, iters=3000):
        # hill climb with geometric step decay
        sp, st = s(p), 0.05
        for it in range(iters):
            q = p.copy()
            q[r.integers(n)] += r.normal(0, st, 2)
            q = g(q)
            sq = s(q)
            if sq >= sp:
                p, sp = q, sq
            if it % 500 == 499:
                st *= 0.6
        return p

    r = np.random.default_rng(0)
    t = 2 * np.pi * np.arange(n) / n
    starts = [g(np.c_[0.52 + 0.2 * np.cos(t), 0.25 + 0.12 * np.sin(t)])]
    starts += [g(np.column_stack([r.uniform(0, 1, n), r.uniform(0, 0.5, n)]))
               for _ in range(40)]
    best, bs = None, -1.0
    for p in starts:
        p = polish(climb(p))
        sp = s(p)
        if sp > bs:
            best, bs = p, sp
    return polish(climb(best, 4000))


def g(p):
    q = np.array(p, dtype=float)
    v = np.clip(q[:, 1], 0.0, 0.5)
    u = np.minimum(np.maximum(q[:, 0], 0.6 * v), 1.0 - 0.2 * v)
    return np.c_[u, v]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
