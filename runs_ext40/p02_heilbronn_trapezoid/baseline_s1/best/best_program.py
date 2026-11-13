# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 13
    p = np.zeros((n, 2))

    for i in range(n):
        t = 2 * np.pi * i / n
        p[i] = [0.52 + 0.2 * np.cos(t), 0.25 + 0.12 * np.sin(t)]

    import itertools
    idx = np.array(list(itertools.combinations(range(n), 3)))

    def min_area(q):
        a, b, c = q[idx[:, 0]], q[idx[:, 1]], q[idx[:, 2]]
        cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return np.min(np.abs(cr)) / 2

    def signed(q):
        a, b, c = q[idx[:, 0]], q[idx[:, 1]], q[idx[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) / 2

    def polish(q):
        try:
            from scipy.optimize import minimize
            sg = np.sign(signed(q))
            sg[sg == 0] = 1
            x0 = np.r_[q.ravel(), min_area(q)]

            def tri(x):
                return sg * signed(x[:-1].reshape(n, 2)) - x[-1]

            def box(x):
                u, v = x[0:-1:2], x[1:-1:2]
                return np.r_[v, 0.5 - v, u - 0.6 * v, 1.0 - 0.2 * v - u]

            r = minimize(lambda x: -x[-1], x0, method='SLSQP',
                         constraints=[{'type': 'ineq', 'fun': tri},
                                      {'type': 'ineq', 'fun': box}],
                         options={'maxiter': 400, 'ftol': 1e-13})
            c = g(r.x[:-1].reshape(n, 2))
            sc = min_area(c)
            if sc > min_area(q):
                return c, sc
        except Exception:
            pass
        return q, min_area(q)

    def deep_polish(q):
        s = min_area(q)
        for _ in range(6):
            c, sc = polish(q)
            if sc <= s * (1 + 1e-9):
                break
            q, s = c, sc
        return q, s

    rng = np.random.default_rng(12345)
    best = g(p)
    best_s = min_area(best)

    def climb(q, step, it, lo, dec):
        s = min_area(q)
        for _ in range(it):
            c = q.copy()
            c[rng.integers(n)] += rng.normal(0, step, 2)
            c = g(c)
            sc = min_area(c)
            if sc >= s:
                q, s = c, sc
            step = max(lo, step * dec)
        return q, s

    starts = [best.copy()] + [g(rng.random((n, 2)) * [1.0, 0.5]) for _ in range(40)]
    res = [climb(q, 0.06, 2000, 0.001, 0.999) for q in starts]
    res.sort(key=lambda r: -r[1])
    for q, _ in res[:12]:
        q, s = deep_polish(q)
        if s > best_s:
            best, best_s = q.copy(), s

    return g(best)


def g(p):
    q = np.array(p, dtype=float)

    v = np.clip(q[:, 1], 0.0, 0.5)
    u = np.minimum(np.maximum(q[:, 0], 0.6 * v), 1.0 - 0.2 * v)
    return np.column_stack([u, v])


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
