# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def pol(p, r):
    n = len(r)
    iu = np.triu_indices(n, 1)

    def cons(z):
        q = z[:2 * n].reshape(n, 2)
        rr = z[2 * n:]
        d = np.sqrt(((q[:, None] - q[None]) ** 2).sum(-1))[iu]
        w = np.hypot(np.maximum(0.6 - q[:, 0], 0), np.maximum(0.6 - q[:, 1], 0))
        return np.concatenate([d - rr[iu[0]] - rr[iu[1]],
                               q[:, 0] - rr, 1 - q[:, 0] - rr,
                               q[:, 1] - rr, 1 - q[:, 1] - rr, w - rr])

    jac = np.r_[np.zeros(2 * n), -np.ones(n)]
    try:
        res = minimize(lambda z: -z[2 * n:].sum(), np.r_[p.ravel(), r],
                       jac=lambda z: jac, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': cons}],
                       bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                       options={'maxiter': 600, 'ftol': 1e-12})
        z = res.x
    except Exception:
        return p, r
    return np.clip(z[:2 * n].reshape(n, 2), 0, 1), z[2 * n:]


def opt(x, y, r, T=4000):
    n = len(x)
    ey = np.eye(n, dtype=bool)
    st = [np.zeros(n) for _ in range(6)]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(T):
        mu = 10.0 * (1e5) ** (t / T)
        lr = 0.01 * (1 - t / T) + 1e-4
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy) + ey
        v = np.maximum(r[:, None] + r[None, :] - d, 0)
        v[ey] = 0
        gr = -1 + 2 * mu * v.sum(1)
        gx = 2 * mu * (v * (-dx / d)).sum(1)
        gy = 2 * mu * (v * (-dy / d)).sum(1)
        a1 = np.maximum(r - x, 0); a2 = np.maximum(r + x - 1, 0)
        a3 = np.maximum(r - y, 0); a4 = np.maximum(r + y - 1, 0)
        gr += 2 * mu * (a1 + a2 + a3 + a4)
        gx += 2 * mu * (a2 - a1)
        gy += 2 * mu * (a4 - a3)
        # forbidden square [0.6,1]^2 (matches g's w constraint)
        ax = np.maximum(0.6 - x, 0); ay = np.maximum(0.6 - y, 0)
        w = np.sqrt(ax * ax + ay * ay) + 1e-12
        c = np.maximum(r - w, 0)
        gr += 2 * mu * c
        gx += 2 * mu * c * ax / w
        gy += 2 * mu * c * ay / w
        out = []
        for k, (val, gv) in enumerate([(x, gx), (y, gy), (r, gr)]):
            m, s = st[2 * k], st[2 * k + 1]
            m[:] = b1 * m + (1 - b1) * gv
            s[:] = b2 * s + (1 - b2) * gv * gv
            mh = m / (1 - b1 ** (t + 1))
            sh = s / (1 - b2 ** (t + 1))
            out.append(val - lr * mh / (np.sqrt(sh) + eps))
        x = np.clip(out[0], 0, 1)
        y = np.clip(out[1], 0, 1)
        r = np.clip(out[2], 0, 0.5)
    return x, y, r


def lpts(rng, n):
    pts = []
    while len(pts) < n:
        u, v = rng.uniform(0.05, 0.95, 2)
        if u < 0.55 or v < 0.55:
            pts.append([u, v])
    return np.array(pts)


def f():
    n = 21
    rng = np.random.default_rng(0)
    # fallback: original grid
    p = np.zeros((n, 2))
    k = 0
    for j in range(5):
        for i in range(5 if j < 3 else 3):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1
    bp, bq = p, g(p, np.full(n, 0.06))
    p_grid = p.copy()
    cands = []
    for t in range(8):
        p0 = p_grid if t == 0 else lpts(rng, n)
        x, y, r = opt(p0[:, 0].copy(), p0[:, 1].copy(), np.full(n, 0.03), T=3000)
        cands.append((np.c_[x, y], r))
    for t in range(16):
        p0 = rng.random((n, 2))
        bad = (p0[:, 0] > 0.6) & (p0[:, 1] > 0.6)
        p0[bad] *= 0.6
        cands.append((p0, rng.uniform(0.01, 0.06, n)))
    for pp, r in cands:
        for pq, rq in ((pp, r), pol(pp, r.copy())):
            qq = g(pq, rq.copy())
            if qq.sum() > bq.sum():
                bp, bq = pq.copy(), qq
    return bp, bq


def g(p, q):
    n = p.shape[0]
    q = np.clip(np.asarray(q, dtype=float), 0, None)
    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = max(0.0, min(q[i], u, v, 1 - u, 1 - v, w))
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((p[i] - p[j]) ** 2))
                if q[i] + q[j] > d:
                    s = d / (q[i] + q[j]) * (1 - 1e-12)
                    q[i] *= s
                    q[j] *= s
    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
