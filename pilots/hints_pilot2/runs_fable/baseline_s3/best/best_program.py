# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 21
    p = np.zeros((n, 2))

    # 0.2-spaced grid filling the whole L-shaped region (radius 0.1 each)
    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
            k += 1

    q = g(p)

    try:
        p, q = h(p, q)
    except Exception:
        pass

    return p, q


def h(p0, q0):
    # Refine with SLSQP: maximize sum of radii, keep the best valid layout
    import time
    from scipy.optimize import minimize

    t0 = time.time()
    n = p0.shape[0]
    a, b = np.triu_indices(n, 1)

    def c(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.hypot(x[a] - x[b], y[a] - y[b]) - r[a] - r[b]
        w = np.hypot(np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)) - r
        return np.concatenate([d, x - r, y - r, 1 - x - r, 1 - y - r, w])

    e = np.concatenate([np.zeros(2 * n), -np.ones(n)])
    rng = np.random.default_rng(1)
    bp, bq = p0, q0
    for t in range(400):
        if time.time() - t0 > 18:
            break
        if t % 3 == 0:
            # perturbed grid start, noise grows over time
            s = 0.02 + 0.004 * t
            z = np.concatenate([p0[:, 0], p0[:, 1], q0 * 0.7])
            z[:2 * n] += rng.normal(0, s, 2 * n)
            z = np.clip(z, 0.02, 0.98)
        else:
            # random start inside the L-shaped region with small radii
            u = rng.uniform(0.05, 0.95, (40 * n, 2))
            u = u[(u[:, 0] < 0.55) | (u[:, 1] < 0.55)][:n]
            if u.shape[0] < n:
                continue
            z = np.concatenate([u[:, 0], u[:, 1], rng.uniform(0.02, 0.06, n)])
        try:
            res = minimize(lambda z: -z[2 * n:].sum(), z, jac=lambda z: e,
                           constraints={"type": "ineq", "fun": c},
                           bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                           method="SLSQP", options={"maxiter": 250})
        except Exception:
            continue
        z = res.x
        if not np.all(np.isfinite(z)):
            continue
        p = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0, 1)
        q = g(p, np.maximum(z[2 * n:], 0) * (1 - 1e-9))
        if np.all(np.isfinite(q)) and q.sum() > bq.sum():
            bp, bq = p, q
    return bp, bq


def g(p, r=None):
    n = p.shape[0]
    q = np.zeros(n)

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = min(u, v, 1 - u, 1 - v, w)
        if r is not None:
            q[i] = min(q[i], r[i])

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
