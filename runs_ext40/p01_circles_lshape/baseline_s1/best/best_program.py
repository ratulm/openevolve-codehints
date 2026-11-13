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

    best = (p, g(p, np.ones(n)))
    bs = best[1].sum()
    rng = np.random.default_rng(0)
    starts = [p.copy()] + [rng.uniform(0.05, 0.95, (n, 2)) for _ in range(12)]
    for s0 in starts:
        s0 = np.where((s0 > 0.6).all(1)[:, None], s0 * 0.5, s0)
        try:
            pp, r = h(s0 + rng.normal(0, 1e-3, s0.shape), np.full(n, 0.05))
            q = g(pp, r)
            if np.all(np.isfinite(q)) and q.sum() > bs:
                bs, best = q.sum(), (pp, q)
        except Exception:
            pass
    return best


def h(p, r, it=4000, lr=3e-3):
    n = len(r)
    E = np.eye(n)
    mp = vp = mr = vr = 0.0
    for k in range(it):
        c = 10 * 1e4 ** (k / it)
        D = p[:, None] - p[None]
        d = np.sqrt((D ** 2).sum(-1)) + E
        o = np.maximum(r[:, None] + r[None] - d, 0) * (1 - E)
        gr = -np.ones(n) + 2 * c * o.sum(1)
        gp = -2 * c * (o[:, :, None] * D / d[:, :, None]).sum(1)
        for a in range(2):
            lo = np.maximum(r - p[:, a], 0)
            hi = np.maximum(r + p[:, a] - 1, 0)
            gr += 2 * c * (lo + hi)
            gp[:, a] += 2 * c * (hi - lo)
        e = np.maximum(0.6 - p, 0)
        ed = np.sqrt((e ** 2).sum(1)) + 1e-12
        ov = np.maximum(r - ed, 0)
        gr += 2 * c * ov
        gp += 2 * c * (ov / ed)[:, None] * e
        mp = 0.9 * mp + 0.1 * gp
        vp = 0.999 * vp + 0.001 * gp ** 2
        mr = 0.9 * mr + 0.1 * gr
        vr = 0.999 * vr + 0.001 * gr ** 2
        s = lr * (1 - k / it) + 1e-5
        p = np.clip(p - s * mp / (np.sqrt(vp) + 1e-9), 0, 1)
        r = np.maximum(r - s * mr / (np.sqrt(vr) + 1e-9), 0)
    return p, r


def g(p, w):
    n = p.shape[0]
    o = np.hypot(np.maximum(0.6 - p[:, 0], 0), np.maximum(0.6 - p[:, 1], 0))
    q = np.minimum.reduce([np.maximum(w, 0), o, p[:, 0], p[:, 1],
                           1 - p[:, 0], 1 - p[:, 1]]).astype(float)

    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

                if q[i] + q[j] > d:
                    s = d / (q[i] + q[j]) * (1 - 1e-9)
                    q[i] *= s
                    q[j] *= s

    return q * (1 - 1e-9)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
