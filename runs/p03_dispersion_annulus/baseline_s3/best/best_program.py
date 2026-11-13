# EVOLVE-BLOCK-START
import numpy as np


def f():
    # 12 points on outer circle, 6 staggered on inner circle; radius chosen so
    # all nearest-neighbour distances are equal (~0.5176).
    r = 1.0 / (2.0 * np.cos(np.pi / 12))
    a_out = 2.0 * np.pi * np.arange(12) / 12
    a_in = 2.0 * np.pi * np.arange(6) / 6 + np.pi / 12
    p = np.vstack([
        np.c_[np.cos(a_out), np.sin(a_out)],
        r * np.c_[np.cos(a_in), np.sin(a_in)],
    ])
    best = g(p)
    bs = m(best)
    try:
        from scipy.optimize import minimize
    except Exception:
        return best
    n = len(p)
    iu = np.triu_indices(n, 1)
    P = lambda x: x[:-1].reshape(n, 2)
    cons = [
        {'type': 'ineq', 'fun': lambda x: np.linalg.norm(
            P(x)[:, None] - P(x)[None], axis=-1)[iu] - x[-1]},
        {'type': 'ineq', 'fun': lambda x: 1 - np.sum(P(x) ** 2, 1)},
        {'type': 'ineq', 'fun': lambda x: np.sum(P(x) ** 2, 1) - 0.16},
    ]
    rng = np.random.default_rng(0)
    starts = [best]
    # structured two-ring starts with varied splits / radii / phase offsets
    for k in (11, 12, 13):
        for ri in (0.40, 0.45, 0.5, 0.55, 0.6):
            for ph in (0.0, 0.5):
                ao = 2 * np.pi * np.arange(k) / k
                ai = 2 * np.pi * (np.arange(n - k) + ph) / (n - k)
                s = np.vstack([np.c_[np.cos(ao), np.sin(ao)],
                               ri * np.c_[np.cos(ai), np.sin(ai)]])
                starts.append(s + rng.normal(0, 0.02, s.shape))
    for _ in range(80):
        rr = np.sqrt(rng.uniform(0.16, 1, n))
        aa = rng.uniform(0, 2 * np.pi, n)
        starts.append(np.c_[rr * np.cos(aa), rr * np.sin(aa)])

    def run_opt(s0, it):
        res = minimize(lambda x: -x[-1], np.r_[s0.ravel(), m(s0)],
                       constraints=cons, method='SLSQP',
                       options={'maxiter': it, 'ftol': 1e-14})
        return g(P(res.x))

    for s0 in starts:
        try:
            q = run_opt(s0, 400)
            s = m(q)
            if np.isfinite(s) and s > bs:
                best, bs = q, s
        except Exception:
            pass
    # final polish of the best configuration
    try:
        q = run_opt(best, 2000)
        s = m(q)
        if np.isfinite(s) and s > bs:
            best, bs = q, s
    except Exception:
        pass
    return best


def m(p):
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return d[np.triu_indices(len(p), 1)].min()


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
