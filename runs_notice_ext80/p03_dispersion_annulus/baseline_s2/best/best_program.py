# EVOLVE-BLOCK-START
import numpy as np


def _md(q):
    D = np.linalg.norm(q[:, None] - q[None], axis=2)
    np.fill_diagonal(D, 9.0)
    return D.min()


def _vg(q):
    d = np.hypot(q[:, 0], q[:, 1])
    d = np.maximum(d, 1e-12)
    return q * (np.clip(d, 0.4, 1.0) / d)[:, None]


def _ring(no, ni, ri, off):
    a = 2 * np.pi * np.arange(no) / no
    b = 2 * np.pi * (np.arange(ni) + off) / ni
    return np.vstack([np.c_[np.cos(a), np.sin(a)],
                      ri * np.c_[np.cos(b), np.sin(b)]])


def _opt(q, it=1500, s0=0.01, beta=80.0):
    n = q.shape[0]
    E = np.eye(n)
    M = ~np.eye(n, dtype=bool)
    best, bs = q.copy(), _md(q)
    for k in range(it):
        D = q[:, None] - q[None]
        L = np.linalg.norm(D, axis=2) + E
        m = L[M].min()
        w = np.exp(-(L - m) * beta) * (1 - E)
        step = s0 * (1 - k / it) + s0 * 0.01
        F = ((w / L)[:, :, None] * D).sum(1)
        F /= np.abs(F).max() + 1e-12
        q = _vg(q + step * F)
        s = _md(q)
        if s > bs:
            bs, best = s, q.copy()
    return best, bs


def _pol(q):
    try:
        from scipy.optimize import minimize
        n = q.shape[0]
        I, J = np.triu_indices(n, 1)
        K = len(I)
        r = np.arange(K)

        def cd(x):
            p = x[:-1].reshape(n, 2)
            return np.sum((p[I] - p[J]) ** 2, 1) - x[-1] ** 2

        def cdj(x):
            p = x[:-1].reshape(n, 2)
            d = 2 * (p[I] - p[J])
            Jm = np.zeros((K, 2 * n + 1))
            Jm[r, 2 * I] = d[:, 0]
            Jm[r, 2 * I + 1] = d[:, 1]
            Jm[r, 2 * J] = -d[:, 0]
            Jm[r, 2 * J + 1] = -d[:, 1]
            Jm[:, -1] = -2 * x[-1]
            return Jm

        def rr(x):
            p = x[:-1].reshape(n, 2)
            return np.sum(p * p, 1)

        def rj(x, sg):
            p = x[:-1].reshape(n, 2)
            Jm = np.zeros((n, 2 * n + 1))
            Jm[np.arange(n), 2 * np.arange(n)] = 2 * p[:, 0] * sg
            Jm[np.arange(n), 2 * np.arange(n) + 1] = 2 * p[:, 1] * sg
            return Jm

        cons = [{'type': 'ineq', 'fun': cd, 'jac': cdj},
                {'type': 'ineq', 'fun': lambda x: 1.0 - rr(x),
                 'jac': lambda x: rj(x, -1.0)},
                {'type': 'ineq', 'fun': lambda x: rr(x) - 0.16,
                 'jac': lambda x: rj(x, 1.0)}]
        x0 = np.r_[q.ravel(), _md(q)]
        res = minimize(lambda x: -x[-1], x0, method='SLSQP',
                       jac=lambda x: np.r_[np.zeros(2 * n), -1.0],
                       constraints=cons,
                       options={'maxiter': 400, 'ftol': 1e-15})
        p = _vg(res.x[:-1].reshape(n, 2))
        if np.all(np.isfinite(p)) and _md(p) > _md(q):
            return p
    except Exception:
        pass
    return q


def f():
    n = 18
    p = np.zeros((n, 2))

    for i in range(n):
        a = 2.0 * np.pi * i / n
        p[i] = [0.55 * np.cos(a), 0.55 * np.sin(a)]

    best, bs = g(p), _md(g(p))
    rng = np.random.default_rng(1)
    starts = [_ring(11, 7, 0.5, 0.5), _ring(12, 6, 0.45, 0.5),
              _ring(10, 8, 0.55, 0.25), _ring(13, 5, 0.5, 0.5),
              _ring(11, 7, 0.55, 0.0)]
    starts += [_vg(rng.uniform(-1, 1, (n, 2))) for _ in range(4)]
    for s0 in starts:
        q, s = _opt(_vg(s0))
        if s > bs:
            bs, best = s, q
    for s0, beta in ((2e-3, 300.0), (3e-4, 1500.0)):
        q, s = _opt(best, it=2500, s0=s0, beta=beta)
        if s > bs:
            bs, best = s, q
    qp = _pol(best)
    if _md(qp) > bs:
        best = qp
    return g(best)


def g(p):
    q = p.copy()

    for i in range(q.shape[0]):
        d = np.hypot(q[i, 0], q[i, 1])
        if d < 0.4:
            q[i] *= 0.4 / d
        elif d > 1.0:
            q[i] *= 1.0 / d

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
