# EVOLVE-BLOCK-START
import numpy as np


def m(p):
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return d[np.triu_indices(len(p), 1)].min()


def f():
    o, k = 11, 7
    a = 2 * np.pi * np.arange(o) / o
    b = 2 * np.pi * (np.arange(k) + 0.5) / k
    p0 = np.vstack([np.c_[np.cos(a), np.sin(a)],
                    0.45 * np.c_[np.cos(b), np.sin(b)]])
    rng = np.random.default_rng(1)
    best, bs = g(p0), m(g(p0))
    T = 600
    for s in range(24):
        p = g(p0 if s == 0 else rng.uniform(-1, 1, (18, 2)))
        R = max(bs, 0.5) * 1.05
        for t in range(T):
            v = p[:, None] - p[None]
            d = np.linalg.norm(v, axis=-1) + np.eye(18)
            w = np.clip(R - d, 0, None) / d
            np.fill_diagonal(w, 0)
            p = g(p + 0.15 * (1 - t / T) * (w[..., None] * v).sum(1))
            sc = m(p)
            if sc > bs:
                best, bs = p.copy(), sc
                R = bs * 1.05
    try:
        from scipy.optimize import minimize
    except Exception:
        return best
    n = 18
    iu = np.triu_indices(n, 1)
    P = lambda x: x[:-1].reshape(n, 2)
    c = [{'type': 'ineq', 'fun': lambda x: np.linalg.norm(
              P(x)[:, None] - P(x)[None], axis=-1)[iu] - x[-1]},
         {'type': 'ineq', 'fun': lambda x: 1 - (P(x) ** 2).sum(1)},
         {'type': 'ineq', 'fun': lambda x: (P(x) ** 2).sum(1) - 0.16}]

    def opt(s0, it):
        x = minimize(lambda x: -x[-1], np.r_[s0.ravel(), m(s0)],
                     constraints=c, method='SLSQP',
                     options={'maxiter': it, 'ftol': 1e-15}).x
        return g(P(x))

    starts = [best]
    for o in (11, 12, 13):
        for ri in (0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
            for ph in (0.0, 0.5):
                ao = 2 * np.pi * np.arange(o) / o
                ai = 2 * np.pi * (np.arange(n - o) + ph) / (n - o)
                s0 = np.vstack([np.c_[np.cos(ao), np.sin(ao)],
                                ri * np.c_[np.cos(ai), np.sin(ai)]])
                starts.append(s0 + rng.normal(0, 0.02, s0.shape))
    for _ in range(150):
        rr = np.sqrt(rng.uniform(0.16, 1, n))
        aa = rng.uniform(0, 2 * np.pi, n)
        starts.append(np.c_[rr * np.cos(aa), rr * np.sin(aa)])
    for i, s0 in enumerate(starts + [None]):
        try:
            q = opt(best if s0 is None else s0, 3000 if s0 is None else 500)
            sc = m(q)
            if np.isfinite(sc) and sc > bs:
                best, bs = q, sc
        except Exception:
            pass
    return best


def g(p):
    r = np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
    return p * np.clip(r, 0.4, 1.0) / r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
