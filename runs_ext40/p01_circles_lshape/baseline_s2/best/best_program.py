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

    q = g(p)
    best = (p, q)
    rng = np.random.default_rng(0)

    for t in range(30):
        if t < 6:
            p0 = p + rng.normal(0, 0.015, p.shape)
        else:
            pts = []
            while len(pts) < n:
                c = rng.uniform(0.05, 0.95, 2)
                if c[0] < 0.55 or c[1] < 0.55:
                    pts.append(c)
            p0 = np.array(pts)
        try:
            c, r = opt(np.clip(p0, 0.02, 0.98), np.full(n, 0.04))
            qq = g(c, r)
            if np.all(np.isfinite(qq)) and qq.sum() > best[1].sum():
                best = (c, qq)
        except Exception:
            pass

    return best


def opt(p, r, steps=4000):
    n = len(r)
    x = np.concatenate([p.ravel(), r])
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    for k in range(1, steps + 1):
        mu = 10.0 * 1.002 ** k
        lr = 0.01 * (1 - k / (steps + 1)) + 1e-4
        p = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        df = p[:, None] - p[None]
        d = np.sqrt((df ** 2).sum(-1) + 1e-12)
        o = r[:, None] + r[None] - d
        np.fill_diagonal(o, 0.0)
        o = np.maximum(o, 0.0)
        gr = -1.0 + 2 * mu * o.sum(1)
        gp = 2 * mu * (-(o / d)[:, :, None] * df).sum(1)
        lo = np.maximum(r[:, None] - p, 0.0)
        hi = np.maximum(r[:, None] + p - 1, 0.0)
        gr += 2 * mu * (lo + hi).sum(1)
        gp += 2 * mu * (hi - lo)
        a = np.maximum(0.6 - p, 0.0)
        w = np.sqrt((a ** 2).sum(1))
        ob = np.maximum(r - w, 0.0)
        dirn = np.where(w[:, None] > 1e-9, a / np.maximum(w, 1e-9)[:, None],
                        np.full((n, 2), 0.7071))
        gr += 2 * mu * ob
        gp += 2 * mu * ob[:, None] * dirn
        gx = np.concatenate([gp.ravel(), gr])
        m1 = b1 * m1 + (1 - b1) * gx
        m2 = b2 * m2 + (1 - b2) * gx * gx
        x -= lr * (m1 / (1 - b1 ** k)) / (np.sqrt(m2 / (1 - b2 ** k)) + 1e-12)
        x[:2 * n] = np.clip(x[:2 * n], 0.0, 1.0)
        x[2 * n:] = np.clip(x[2 * n:], 0.0, 0.5)
    return x[:2 * n].reshape(n, 2).copy(), x[2 * n:].copy()


def g(p, r0=None):
    n = p.shape[0]
    q = np.zeros(n)

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        c = r0[i] if r0 is not None else 0.5
        q[i] = min(u, v, 1 - u, 1 - v, w, c)
        if r0 is not None:
            q[i] = max(q[i] - 1e-10, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))
            if r0 is not None:
                d = d * (1 - 1e-10)

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
