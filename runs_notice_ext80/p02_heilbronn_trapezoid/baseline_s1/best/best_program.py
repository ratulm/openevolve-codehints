# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 13
    p = np.zeros((n, 2))

    for i in range(n):
        t = 2 * np.pi * i / n
        p[i] = [0.56 + 0.375 * np.cos(t), 0.25 + 0.24 * np.sin(t)]

    q = h(g(p))
    # second seed: rotated, slightly shrunk ellipse; keep the better result
    t2 = 2 * np.pi * np.arange(n) / n + 0.2
    p2 = np.c_[0.56 + 0.36 * np.cos(t2), 0.25 + 0.23 * np.sin(t2)]
    q2 = h(g(p2), seed=7)
    T = _tri(n)
    if m(q2, T).min() > m(q, T).min():
        q = q2
    best = m(q, T).min()
    k0 = np.arange(n)
    seeds = []
    for ph, a, b in ((0.1, 0.37, 0.235), (0.35, 0.35, 0.22), (0.45, 0.38, 0.24)):
        tt = 2 * np.pi * k0 / n + ph
        seeds.append(np.c_[0.56 + a * np.cos(tt), 0.25 + b * np.sin(tt)])
    to, ti = 2 * np.pi * np.arange(9) / 9, 2 * np.pi * np.arange(4) / 4 + 0.4
    seeds.append(np.r_[np.c_[0.58 + 0.37 * np.cos(to), 0.25 + 0.235 * np.sin(to)],
                       np.c_[0.58 + 0.15 * np.cos(ti), 0.25 + 0.1 * np.sin(ti)]])
    seeds.append(np.r_[np.c_[0.58 + 0.37 * np.cos(to + 0.3), 0.25 + 0.235 * np.sin(to + 0.3)],
                       np.c_[0.58 + 0.13 * np.cos(ti), 0.25 + 0.09 * np.sin(ti)]])
    improved = False
    for k, s in enumerate(seeds):
        qk = h(g(s), seed=20 + k, R=6)
        vk = m(qk, T).min()
        if np.isfinite(vk) and vk > best:
            best, q, improved = vk, qk, True
    if improved:
        qf = h(g(q), seed=5, R=8)
        if m(qf, T).min() > best:
            q = qf
    return g(q)


def _tri(n):
    import itertools
    return np.array(list(itertools.combinations(range(n), 3)))


def m(q, T):
    a, b, c = q[T[:, 0]], q[T[:, 1]], q[T[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def h(p0, seed=1, R=16):
    # Perturbed-restart SLSQP on epigraph form of max-min triangle area.
    try:
        from scipy.optimize import minimize
    except Exception:
        return p0
    n = p0.shape[0]
    T = _tri(n)
    best, bp = m(p0, T).min(), p0.copy()
    rng = np.random.default_rng(seed)
    cons = [
        {"type": "ineq", "fun": lambda x: m(x[:-1].reshape(n, 2), T) - x[-1]},
        {"type": "ineq", "fun": lambda x: x[1:-1:2]},
        {"type": "ineq", "fun": lambda x: 0.5 - x[1:-1:2]},
        {"type": "ineq", "fun": lambda x: x[0:-1:2] - 0.6 * x[1:-1:2]},
        {"type": "ineq", "fun": lambda x: 1.0 - 0.2 * x[1:-1:2] - x[0:-1:2]},
    ]

    def run(s0):
        x0 = np.r_[s0.ravel(), m(s0, T).min()]
        try:
            s = minimize(lambda x: -x[-1], x0, constraints=cons,
                         method="SLSQP",
                         options={"maxiter": 400, "ftol": 1e-13})
            q = g(s.x[:-1].reshape(n, 2))
            v = m(q, T).min()
            return (v, q) if np.isfinite(v) else (-1, q)
        except Exception:
            return -1, s0

    for r in range(R):
        sig = 0.04 if r < R // 2 else 0.01
        s0 = bp if r == 0 else g(bp + rng.normal(0, sig, bp.shape))
        v, q = run(s0)
        if v > best:
            best, bp = v, q
    return bp


def g(p):
    q = np.array(p, dtype=float)
    v = np.clip(q[:, 1], 0.0, 0.5)
    u = np.minimum(np.maximum(q[:, 0], 0.6 * v), 1.0 - 0.2 * v)
    return np.c_[u, v]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
