# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def f():
    n = 21
    p = np.zeros((n, 2))
    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1
    bz = np.concatenate([p[:, 0], p[:, 1], np.full(n, 0.06)])
    best = 0.06 * n

    iu, ju = np.triu_indices(n, 1)
    npair = len(iu)

    def obj(z):
        return -z[2 * n:].sum()

    def jac(z):
        gr = np.zeros(3 * n)
        gr[2 * n:] = -1.0
        return gr

    def con(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.sqrt((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2)
        return np.concatenate([d - r[iu] - r[ju], x - r, 1 - x - r, y - r, 1 - y - r])

    def cjac(z):
        x, y = z[:n], z[n:2 * n]
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        d = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
        J = np.zeros((npair + 4 * n, 3 * n))
        a = np.arange(npair)
        J[a, iu] = dx / d
        J[a, ju] = -dx / d
        J[a, n + iu] = dy / d
        J[a, n + ju] = -dy / d
        J[a, 2 * n + iu] = -1
        J[a, 2 * n + ju] = -1
        e = np.arange(n)
        o = npair
        J[o + e, e] = 1; J[o + e, 2 * n + e] = -1
        o += n
        J[o + e, e] = -1; J[o + e, 2 * n + e] = -1
        o += n
        J[o + e, n + e] = 1; J[o + e, 2 * n + e] = -1
        o += n
        J[o + e, n + e] = -1; J[o + e, 2 * n + e] = -1
        return J

    def cjac2(z):
        x, y = z[:n], z[n:2 * n]
        J = np.vstack([cjac(z), np.zeros((n, 3 * n))])
        ax, ay = np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0)
        w = np.hypot(ax, ay) + 1e-12
        e = np.arange(n)
        o = npair + 4 * n
        J[o + e, e] = -ax / w
        J[o + e, n + e] = -ay / w
        J[o + e, 2 * n + e] = -1
        return J

    def wd(x, y):
        return np.hypot(np.maximum(0.6 - x, 0), np.maximum(0.6 - y, 0))

    def con2(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.sqrt((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2 + 1e-18)
        return np.concatenate([d - r[iu] - r[ju], x - r, 1 - x - r,
                               y - r, 1 - y - r, wd(x, y) - r])

    def consi(x):
        P = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1) + 1e-18)[iu, ju]
        return np.concatenate([d - r[iu] - r[ju], P[:, 0] - r, P[:, 1] - r,
                               1 - P[:, 0] - r, 1 - P[:, 1] - r,
                               wd(P[:, 0], P[:, 1]) - r])

    bp = p.copy()
    bq = g(bp, np.full(n, 0.5))
    rng = np.random.default_rng(0)
    bnds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    T = 30
    for t in range(T + 20):
        if (t >= T // 2 and t % 2) or t >= T:
            sc = 0.03 if t < T else (0.01, 0.02, 0.04)[t % 3]
            pts = np.clip(bp + rng.normal(0, sc, (n, 2)), 0.02, 0.98)
            r0 = bq * 0.9
        else:
            pts = []
            while len(pts) < n:
                c = rng.uniform(0.05, 0.95, 2)
                if wd(c[0], c[1]) > 0.05:
                    pts.append(c)
            pts = np.array(pts)
            r0 = np.full(n, 0.04)
        z0 = np.stack([pts[:, 0], pts[:, 1]], 1).ravel()
        z0 = np.concatenate([z0, r0])
        try:
            res = minimize(lambda x: -x[2 * n:].sum(), z0,
                           jac=lambda x: np.concatenate([np.zeros(2 * n), -np.ones(n)]),
                           constraints=[{"type": "ineq", "fun": consi}],
                           bounds=bnds, method="SLSQP",
                           options={"maxiter": 1000, "ftol": 1e-10})
        except Exception:
            continue
        z = res.x
        if not np.all(np.isfinite(z)):
            continue
        pp = np.clip(z[:2 * n].reshape(n, 2), 0.0, 1.0)
        qq = g(pp, z[2 * n:].copy())
        if qq.sum() > bq.sum():
            bp, bq = pp, qq
    return bp, bq


def g(p, q):
    n = p.shape[0]
    q = np.maximum(q, 0.0)
    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = min(q[i], u, v, 1 - u, 1 - v, w)

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
