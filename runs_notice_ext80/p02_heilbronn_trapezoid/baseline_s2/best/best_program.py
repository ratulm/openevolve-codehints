# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 13
    p = np.zeros((n, 2))

    for i in range(n):
        t = 2 * np.pi * i / n
        p[i] = [0.56 + 0.38 * np.cos(t), 0.25 + 0.245 * np.sin(t)]

    p = g(p)
    try:
        p = h(p)
    except Exception:
        pass
    return g(p)


def a(p, T):
    x, y = p[:, 0], p[:, 1]
    i, j, k = T[:, 0], T[:, 1], T[:, 2]
    return 0.5 * np.abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))


def h(p0):
    import itertools
    from scipy.optimize import minimize

    n = p0.shape[0]
    T = np.array(list(itertools.combinations(range(n), 3)))
    best, bs = p0.copy(), a(p0, T).min()
    rng = np.random.default_rng(1)

    def c_area(z):
        return a(z[:-1].reshape(n, 2), T) - z[-1]

    def c_box(z):
        x, y = z[0:-1:2], z[1:-1:2]
        return np.concatenate([y, 0.5 - y, x - 0.6 * y, 1.0 - 0.2 * y - x])

    cons = [dict(type="ineq", fun=c_area), dict(type="ineq", fun=c_box)]
    sds = [0.02, 0.04, 0.01, 0.07]
    for r in range(16):
        s = best if r == 0 else g(best + rng.normal(0, sds[r % 4], best.shape))
        z0 = np.concatenate([s.ravel(), [a(s, T).min()]])
        res = minimize(lambda z: -z[-1], z0, method="SLSQP",
                       constraints=cons, options=dict(maxiter=500, ftol=1e-12))
        q = g(res.x[:-1].reshape(n, 2))
        m = a(q, T).min()
        if m > bs:
            bs, best = m, q
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
