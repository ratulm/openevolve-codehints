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
    def tryq(s0, it):
        nonlocal best, bs
        try:
            q = opt(s0, it)
            sc = m(q)
            if np.isfinite(sc) and sc > bs:
                best, bs = q, sc
        except Exception:
            pass

    for s0 in starts:
        tryq(s0, 500)
    tryq(best, 3000)
    # rotate the outer or inner ring alone, then re-optimise
    for _ in range(4):
        rad = np.linalg.norm(best, axis=1)
        for thr in (0.6, 0.7, 0.8):
            for side in (0, 1):
                mask = (rad > thr) if side else (rad <= thr)
                if mask.sum() in (0, n):
                    continue
                for th in rng.uniform(-0.3, 0.3, 4):
                    c_, s_ = np.cos(th), np.sin(th)
                    q = best.copy()
                    q[mask] = q[mask] @ np.array([[c_, s_], [-s_, c_]])
                    tryq(q, 500)
    # shrink toward the middle of the band, then polish for longer
    for lam in (0.98, 0.995):
        r = np.linalg.norm(best, axis=1, keepdims=True)
        tryq(best * (0.7 + lam * (r - 0.7)) / r, 4000)
    tryq(best, 4000)
    return best


def g(p):
    r = np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
    return p * np.clip(r, 0.4, 1.0) / r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
