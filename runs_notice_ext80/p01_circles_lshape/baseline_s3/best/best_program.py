# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 21
    rng = np.random.default_rng(1)
    best = None
    for t in range(41):
        if t == 0:
            c = []
            for j, m in enumerate([5, 4, 5, 4, 3]):
                for i in range(m):
                    c.append([(i + 0.5 + 0.5 * (m < 5)) / 5, (j + 0.5) / 5])
            c = np.array(c[:n]) + rng.normal(0, 0.01, (n, 2))
            c = np.clip(c, 0.01, 0.99)
        else:
            c = rng.uniform(0.05, 0.95, (n, 2))
            c[:, 0] *= 0.95
        mk = (c[:, 0] > 0.58) & (c[:, 1] > 0.58)
        c[mk, rng.integers(0, 2, mk.sum())] *= 0.5
        if t == 0:
            p, r = h(c, np.full(n, 0.04))
        else:
            p, r = c, np.full(n, 0.02)
        p, r = s(p, r)
        q = g(p, r)
        if best is None or q.sum() > best[1].sum():
            best = (p, q)
    for t in range(40):
        c = best[0] + rng.normal(0, 0.01 + 0.015 * (t % 3), (n, 2))
        k = rng.choice(n, 1 + t % 3, replace=False)
        c[k] = rng.uniform(0.05, 0.95, (len(k), 2))
        c = np.clip(c, 0.01, 0.99)
        mk = (c[:, 0] > 0.6) & (c[:, 1] > 0.6)
        c[mk, rng.integers(0, 2, mk.sum())] *= 0.5
        r0 = best[1] * 0.9
        r0[k] = 0.01
        p, r = s(c, r0)
        q = g(p, r)
        if q.sum() > best[1].sum():
            best = (p, q)
    return best


def h(p, r, steps=3000):
    n = len(r)
    x = np.concatenate([p.ravel(), r]).astype(float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for k in range(steps):
        lam = 10 * 10 ** (4 * k / steps)
        P = x[:2 * n].reshape(n, 2)
        R = x[2 * n:]
        D = P[:, None] - P[None]
        d = np.sqrt((D ** 2).sum(-1) + 1e-12)
        o = np.maximum(R[:, None] + R[None] - d, 0)
        np.fill_diagonal(o, 0)
        gR = -1 + lam * 2 * o.sum(1)
        gP = -lam * 2 * ((o / d)[:, :, None] * D).sum(1)
        for a in range(2):
            lo = np.maximum(R - P[:, a], 0)
            hi = np.maximum(R + P[:, a] - 1, 0)
            gR += lam * 2 * (lo + hi)
            gP[:, a] += lam * 2 * (hi - lo)
        ax = np.maximum(0.6 - P[:, 0], 0)
        ay = np.maximum(0.6 - P[:, 1], 0)
        sq = np.sqrt(ax * ax + ay * ay + 1e-12)
        vo = np.maximum(R - sq, 0)
        gR += lam * 2 * vo
        gP[:, 0] += lam * 2 * vo * ax / sq
        gP[:, 1] += lam * 2 * vo * ay / sq
        gr = np.concatenate([gP.ravel(), gR])
        m = 0.9 * m + 0.1 * gr
        v = 0.999 * v + 0.001 * gr ** 2
        lr = 0.002 * (1 - k / steps) + 1e-5
        mh = m / (1 - 0.9 ** (k + 1))
        vh = v / (1 - 0.999 ** (k + 1))
        x -= lr * mh / (np.sqrt(vh) + 1e-12)
        x[:2 * n] = np.clip(x[:2 * n], 0, 1)
        x[2 * n:] = np.clip(x[2 * n:], 0, 0.5)
    return x[:2 * n].reshape(n, 2), x[2 * n:]


def s(p, r):
    from scipy.optimize import minimize
    n = len(r)
    iu = np.triu_indices(n, 1)

    def cons(x):
        P = x[:2 * n].reshape(n, 2)
        R = x[2 * n:]
        d = np.sqrt(((P[iu[0]] - P[iu[1]]) ** 2).sum(1) + 1e-14)
        a = np.maximum(0.6 - P[:, 0], 0)
        e = np.maximum(0.6 - P[:, 1], 0)
        return np.concatenate([d - R[iu[0]] - R[iu[1]],
                               P[:, 0] - R, 1 - P[:, 0] - R,
                               P[:, 1] - R, 1 - P[:, 1] - R,
                               a * a + e * e - R * R])

    x0 = np.concatenate([p.ravel(), r])
    try:
        o = minimize(lambda x: -x[2 * n:].sum(), x0,
                     jac=lambda x: np.concatenate([np.zeros(2 * n), -np.ones(n)]),
                     constraints={'type': 'ineq', 'fun': cons},
                     bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                     method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-12})
        x = o.x if np.all(np.isfinite(o.x)) else x0
    except Exception:
        x = x0
    return np.clip(x[:2 * n].reshape(n, 2), 0, 1), x[2 * n:]


def g(p, r):
    n = p.shape[0]
    q = np.clip(np.nan_to_num(r), 0, None).copy()

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = max(0.0, min(q[i], u, v, 1 - u, 1 - v, w))

    for _ in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((p[i] - p[j]) ** 2))
                if q[i] + q[j] > d:
                    s = d / (q[i] + q[j])
                    q[i] *= s
                    q[j] *= s

    return q * (1 - 1e-9)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
