# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 21
    p = np.zeros((n, 2))

    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1

    best_p, best_q = p, g(p)
    try:
        import time
        from scipy.optimize import minimize

        a, b = np.triu_indices(n, 1)

        def cons(z):
            x, y, r = z[:n], z[n:2 * n], z[2 * n:]
            w = np.sqrt(np.maximum(0.6 - x, 0) ** 2
                        + np.maximum(0.6 - y, 0) ** 2 + 1e-18)
            d = np.sqrt((x[a] - x[b]) ** 2 + (y[a] - y[b]) ** 2 + 1e-18)
            return np.concatenate(
                [x - r, y - r, 1 - x - r, 1 - y - r, w - r, d - r[a] - r[b]])

        grad = np.r_[np.zeros(2 * n), -np.ones(n)]
        rng = np.random.default_rng(1)
        t0 = time.time()
        for t in range(30):
            if time.time() - t0 > 25:
                break
            if t == 0:
                s = p + rng.normal(0, 0.004, p.shape)
                r0 = 0.05
            else:
                s = rng.random((400, 2))
                s = s[~((s[:, 0] > 0.6) & (s[:, 1] > 0.6))][:n]
                r0 = 0.03
            z = np.concatenate([s[:, 0], s[:, 1], np.full(n, r0)])
            o = minimize(lambda z: -z[2 * n:].sum(), z, jac=lambda z: grad,
                         constraints={"type": "ineq", "fun": cons},
                         bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                         method="SLSQP", options={"maxiter": 300})
            c = np.column_stack([o.x[:n], o.x[n:2 * n]])
            r = g(c, o.x[2 * n:] - 1e-9)
            if np.all(np.isfinite(r)) and r.min() >= 0 and r.sum() > best_q.sum():
                best_p, best_q = c, r
    except Exception:
        pass

    return best_p, best_q


def g(p, cap=None):
    n = p.shape[0]
    q = np.zeros(n)

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = max(min(u, v, 1 - u, 1 - v, w), 0.0)
        if cap is not None:
            q[i] = max(min(q[i], cap[i]), 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

            if q[i] + q[j] > d:
                s = d / (q[i] + q[j])
                q[i] *= s
                q[j] *= s

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
