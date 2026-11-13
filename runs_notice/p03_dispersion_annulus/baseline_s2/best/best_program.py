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


def _opt(q, it=1500):
    n = q.shape[0]
    E = np.eye(n)
    M = ~np.eye(n, dtype=bool)
    best, bs = q.copy(), _md(q)
    for k in range(it):
        D = q[:, None] - q[None]
        L = np.linalg.norm(D, axis=2) + E
        m = L[M].min()
        w = np.exp(-(L - m) * 80.0) * (1 - E)
        step = 0.01 * (1 - k / it) + 1e-4
        F = ((w / L)[:, :, None] * D).sum(1)
        F /= np.abs(F).max() + 1e-12
        q = _vg(q + step * F)
        s = _md(q)
        if s > bs:
            bs, best = s, q.copy()
    return best, bs


def f():
    n = 18
    p = np.zeros((n, 2))

    for i in range(n):
        a = 2.0 * np.pi * i / n
        p[i] = [0.55 * np.cos(a), 0.55 * np.sin(a)]

    best, bs = g(p), _md(g(p))
    rng = np.random.default_rng(1)
    starts = [_ring(11, 7, 0.5, 0.5), _ring(12, 6, 0.45, 0.5),
              _ring(10, 8, 0.55, 0.25)]
    starts += [_vg(rng.uniform(-1, 1, (n, 2))) for _ in range(3)]
    for s0 in starts:
        q, s = _opt(_vg(s0))
        if s > bs:
            bs, best = s, q
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
