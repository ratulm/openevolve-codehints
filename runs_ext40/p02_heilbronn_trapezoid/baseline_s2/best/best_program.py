# EVOLVE-BLOCK-START
import numpy as np


def f():
    import itertools
    n = 13
    T = np.array(list(itertools.combinations(range(n), 3)))

    def m(p):
        a, b, c = p[T[:, 0]], p[T[:, 1]], p[T[:, 2]]
        return np.min(np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                             - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])))

    I, J, K = T.T

    def A(p):
        a, b, c = p[I], p[J], p[K]
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    def jacA(p):
        # gradient of signed area wrt all coordinates
        Jm = np.zeros((len(T), 2 * n))
        a, b, c = p[I], p[J], p[K]
        r_ = np.arange(len(T))
        Jm[r_, 2 * I] = b[:, 1] - c[:, 1]
        Jm[r_, 2 * I + 1] = c[:, 0] - b[:, 0]
        Jm[r_, 2 * J] = c[:, 1] - a[:, 1]
        Jm[r_, 2 * J + 1] = a[:, 0] - c[:, 0]
        Jm[r_, 2 * K] = a[:, 1] - b[:, 1]
        Jm[r_, 2 * K + 1] = b[:, 0] - a[:, 0]
        return Jm

    def polish(p):
        try:
            from scipy.optimize import minimize
            for _ in range(3):
                s = np.sign(A(p))
                s[s == 0] = 1
                x0 = np.append(p.ravel(), m(p))
                L = np.zeros((4 * n, 2 * n + 1))
                lb = np.zeros(4 * n)
                for i in range(n):
                    L[i, 2 * i + 1] = 1                      # v >= 0
                    L[n + i, 2 * i + 1] = -1; lb[n + i] = -0.5  # v <= .5
                    L[2 * n + i, 2 * i] = 1; L[2 * n + i, 2 * i + 1] = -0.6
                    L[3 * n + i, 2 * i] = -1; L[3 * n + i, 2 * i + 1] = -0.2; lb[3 * n + i] = -1
                cons = [
                    {'type': 'ineq',
                     'fun': lambda x: s * A(x[:-1].reshape(n, 2)) - x[-1],
                     'jac': lambda x: np.hstack([s[:, None] * jacA(x[:-1].reshape(n, 2)),
                                                 -np.ones((len(T), 1))])},
                    {'type': 'ineq', 'fun': lambda x: L @ x - lb, 'jac': lambda x: L},
                ]
                res = minimize(lambda x: -x[-1], x0,
                               jac=lambda x: np.r_[np.zeros(2 * n), -1.0],
                               constraints=cons, method='SLSQP',
                               options={'maxiter': 500, 'ftol': 1e-15})
                q = g(res.x[:-1].reshape(n, 2))
                if m(q) > m(p) + 1e-15:
                    p = q
                else:
                    break
        except Exception:
            pass
        return p

    def climb(p, iters, st):
        sc = m(p)
        for it in range(iters):
            q = p.copy()
            i = r.integers(n)
            q[i] += r.normal(0, st, 2)
            q = g(q)
            v = m(q)
            if v > sc:
                p, sc = q, v
            st = max(st * 0.998, 1e-4)
        return p

    r = np.random.default_rng(0)
    # baseline: original ellipse layout (never do worse than it)
    e = np.array([[0.52 + 0.2 * np.cos(2 * np.pi * i / n),
                   0.25 + 0.12 * np.sin(2 * np.pi * i / n)] for i in range(n)])
    best = polish(climb(g(e), 500, 0.01))
    bs = m(best)

    import time
    t0 = time.time()
    # many cheap multistarts: uniform feasible samples, short climb, SLSQP polish
    for s in range(150):
        if time.time() - t0 > 60:
            break
        v = r.uniform(0, 0.5, n)
        u = r.uniform(0, 1, n)
        p = g(np.c_[0.6 * v + u * (1 - 0.8 * v), v])
        p = polish(climb(p, 150, 0.03))
        sc = m(p)
        if sc > bs:
            best, bs = p, sc
    # basin-hopping around the incumbent
    sigs = [0.03, 0.01, 0.003]
    for k in range(120):
        if time.time() - t0 > 90:
            break
        q = polish(g(best + r.normal(0, sigs[k % 3], best.shape)))
        sc = m(q)
        if sc > bs:
            best, bs = q, sc
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
