# EVOLVE-BLOCK-START
import numpy as np


def f():
    # 13 points on the outer boundary (r=1) and 5 on an inner ring (r=0.46),
    # half a step apart. Smallest gap overall is the outer one, 2*sin(pi/13) ~ 0.4786.
    # r=0.46 roughly balances the inner gap (2r*sin36 ~ 0.540) against the
    # closest inner-outer gap (~0.541), so the inner points have the most room.
    ao = 2.0 * np.pi * np.arange(13) / 13
    ai = 2.0 * np.pi * (np.arange(5) + 0.5) / 5
    p = np.vstack([np.c_[np.cos(ao), np.sin(ao)],
                   0.46 * np.c_[np.cos(ai), np.sin(ai)]])
    p = g(p)
    try:
        p = h(p)
    except Exception:
        pass
    return p


def md(p):
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    return d[np.triu_indices(len(p), 1)].min()


def h(p0):
    # Maximise t subject to all pairwise distances >= t and 0.16 <= |p|^2 <= 1.
    from scipy.optimize import minimize
    n = len(p0)
    iu = np.triu_indices(n, 1)

    def cons(x):
        p = x[:2 * n].reshape(n, 2)
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))[iu]
        r = (p ** 2).sum(1)
        return np.r_[d - x[-1], 1.0 - r, r - 0.16]

    a12 = 2 * np.pi * np.arange(12) / 12
    a6 = 2 * np.pi * np.arange(6) / 6 + np.pi / 12
    starts = [p0, np.vstack([np.c_[np.cos(a12), np.sin(a12)],
                             0.5176 * np.c_[np.cos(a6), np.sin(a6)]])]
    rng = np.random.default_rng(1)
    for _ in range(12):
        r = np.sqrt(rng.uniform(0.16, 1.0, n))
        a = rng.uniform(0, 2 * np.pi, n)
        starts.append(np.c_[r * np.cos(a), r * np.sin(a)])
    best, bm = p0, md(p0)
    for s in starts:
        x0 = np.r_[s.ravel(), 0.9 * md(s)]
        res = minimize(lambda x: -x[-1], x0, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': cons}],
                       options={'maxiter': 400, 'ftol': 1e-12})
        q = g(res.x[:2 * n].reshape(n, 2))
        m = md(q)
        if np.all(np.isfinite(q)) and m > bm:
            best, bm = q, m
    return best


def g(p):
    d = np.maximum(np.hypot(p[:, 0], p[:, 1]), 1e-12)
    return p * (np.clip(d, 0.4, 1.0) / d)[:, None]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
