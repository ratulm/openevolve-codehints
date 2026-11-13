# EVOLVE-BLOCK-START
import numpy as np


def _md(p):
    d = p[:, None] - p[None]
    return (np.hypot(d[..., 0], d[..., 1]) + np.eye(p.shape[0]) * 9).min()


def _polish(p):
    try:
        from scipy.optimize import minimize
    except Exception:
        return p
    n = p.shape[0]
    iu = np.triu_indices(n, 1)

    def con(x):
        q = x[:-1].reshape(n, 2)
        d = q[:, None] - q[None]
        rr = (q ** 2).sum(1)
        return np.r_[np.hypot(d[..., 0], d[..., 1])[iu] - x[-1],
                     rr - 0.16, 1.0 - rr]

    try:
        res = minimize(lambda x: -x[-1], np.r_[p.ravel(), _md(p)],
                       jac=lambda x: np.r_[np.zeros(2 * n), -1.0],
                       constraints=[{'type': 'ineq', 'fun': con}],
                       method='SLSQP',
                       options={'maxiter': 500, 'ftol': 1e-14})
        q = g(res.x[:-1].reshape(n, 2))
        return q if _md(q) > _md(p) else p
    except Exception:
        return p


def f():
    n = 18
    rng = np.random.default_rng(0)
    best, bs = None, -1.0
    for t in range(30):
        k = 10 + t % 5
        m = n - k
        a = np.r_[2 * np.pi * np.arange(k) / k,
                  2 * np.pi * (np.arange(m) + 0.5) / m + rng.uniform(0, 1)]
        r = np.r_[np.ones(k), np.full(m, 0.4 + 0.15 * rng.uniform())]
        p = g(np.c_[r * np.cos(a), r * np.sin(a)] + rng.normal(0, 0.03, (n, 2)))
        if t % 3 == 2:
            p = g(rng.uniform(-1, 1, (n, 2)))
        step = 0.003
        for it in range(700):
            d = p[:, None] - p[None]
            D = np.hypot(d[..., 0], d[..., 1]) + np.eye(n) * 9
            w = np.exp(-(D - D.min()) * 60)
            p = g(p + step * (d * (w / D)[..., None]).sum(1))
            step *= 0.996
        p = _polish(p)
        s = _md(p)
        if s > bs:
            bs, best = s, p.copy()
    return best


def g(p):
    d = np.hypot(p[:, 0], p[:, 1])[:, None]
    d = np.maximum(d, 1e-12)
    return p * np.clip(d, 0.4, 1.0) / d


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
