# EVOLVE-BLOCK-START
import numpy as np


def f():
    # Two staggered rings filling the annulus 0.4 <= r <= 1.0
    a_out = 2.0 * np.pi * np.arange(12) / 12
    a_in = 2.0 * np.pi * np.arange(6) / 6 + np.pi / 12
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)]) * 1.0
    inner = np.column_stack([np.cos(a_in), np.sin(a_in)]) * 0.5
    p0 = g(np.vstack([outer, inner]))
    rng = np.random.default_rng(1)
    n = p0.shape[0]
    iu = np.triu_indices(n, 1)

    def md(p):
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        return d[iu].min()

    best, bm = p0, md(p0)
    for t in range(40):
        if t == 0:
            p = p0.copy()
        else:
            r = np.sqrt(rng.uniform(0.16, 1.0, n))
            a = rng.uniform(0, 2 * np.pi, n)
            p = np.column_stack([r * np.cos(a), r * np.sin(a)])
        for k in range(600):
            diff = p[:, None] - p[None]
            d = np.sqrt((diff ** 2).sum(-1)) + np.eye(n)
            m = d[iu].min()
            beta = 20.0 + k * 0.5
            w = np.exp(-beta * (d - m)) * (1 - np.eye(n))
            step = 0.02 * (1 - k / 600) + 1e-4
            gvec = (w[:, :, None] * diff / d[:, :, None]).sum(1)
            nrm = np.abs(gvec).max() + 1e-12
            p = g(p + step * gvec / nrm)
            cm = md(p)
            if cm > bm:
                bm, best = cm, p.copy()
    return best


def g(p):
    d = np.hypot(p[:, 0], p[:, 1])
    s = np.clip(d, 0.4, 1.0) / np.maximum(d, 1e-12)
    return p * s[:, None]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
