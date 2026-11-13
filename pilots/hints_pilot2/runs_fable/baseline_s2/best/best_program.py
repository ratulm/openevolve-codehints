# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 21
    p = np.zeros((n, 2))

    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
            k += 1

    q = g(p)

    try:
        p2, q2 = h(p, q)
        if q2.sum() > q.sum():
            p, q = p2, q2
    except Exception:
        pass

    return p, q


def c(z, n, a, b):
    # all constraints (>= 0 means feasible): pairs, walls, cut-out corner
    x, y, r = z[:n], z[n:2 * n], z[2 * n:]
    d = np.hypot(x[a] - x[b], y[a] - y[b]) - r[a] - r[b]
    w = np.hypot(np.maximum(0.6 - x, 0.0), np.maximum(0.6 - y, 0.0)) - r
    return np.concatenate([d, x - r, y - r, 1 - x - r, 1 - y - r, w])


def h(p, q):
    # refine centres and radii jointly; keep only strictly valid improvements
    import time
    from scipy.optimize import minimize

    n = p.shape[0]
    a, b = np.triu_indices(n, 1)
    t0 = time.time()
    rng = np.random.default_rng(1)
    bp, bq = p, q
    for t in range(6):
        if time.time() - t0 > 20:
            break
        s = 0.0 if t == 0 else 0.03
        z = np.concatenate([
            np.clip(p[:, 0] + s * rng.standard_normal(n), 0.03, 0.97),
            np.clip(p[:, 1] + s * rng.standard_normal(n), 0.03, 0.97),
            q * (1.0 if t == 0 else 0.6),
        ])
        res = minimize(
            lambda z: -z[2 * n:].sum(), z, method="SLSQP",
            constraints=dict(type="ineq", fun=lambda z: c(z, n, a, b)),
            bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
            options=dict(maxiter=150, ftol=1e-9),
        )
        z = res.x.copy()
        z[2 * n:] -= 1e-7
        if np.all(np.isfinite(z)) and z[2 * n:].min() > 0 \
                and c(z, n, a, b).min() >= 0 and z[2 * n:].sum() > bq.sum():
            bp = np.column_stack([z[:n], z[n:2 * n]])
            bq = z[2 * n:].copy()
    return bp, bq


def g(p):
    n = p.shape[0]
    q = np.zeros(n)

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = min(u, v, 1 - u, 1 - v, w)

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
