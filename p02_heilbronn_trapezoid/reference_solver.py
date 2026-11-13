"""Offline solver for the trapezoid point-placement instance (not shown to the LLM).

Region: trapezoid with corners (0,0), (1,0), (0.9,0.5), (0.3,0.5), i.e.
0 <= y <= 0.5 and 0.6*y <= x <= 1 - 0.2*y.  N points.
Objective family: (smallest triangle area)^a * (median triangle area)^(1-a).
"""
import itertools
import numpy as np
from scipy.optimize import minimize

N = 13
H = 0.5
SL = 0.6   # left edge:  x >= SL * y
SR = 0.2   # right edge: x <= 1 - SR * y
AREA = 0.4
TRI = np.array(list(itertools.combinations(range(N), 3)))
I, J, K = TRI.T
M = len(TRI)          # 286
HALF = M // 2         # 143


def signed_areas(p):
    a, b, c = p[I], p[J], p[K]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def stats(p):
    ar = np.abs(signed_areas(p))
    return float(ar.min()), float(np.median(ar))


def mix(mn, md, a):
    """Weighted geometric mean min^a * median^(1-a); a = 1 is the plain minimum."""
    return max(mn, 0.0) ** a * max(md, 0.0) ** (1.0 - a)


def value(p, a):
    return mix(*stats(p), a)


def clip(p):
    p = np.array(p, dtype=float)
    p[:, 1] = np.clip(p[:, 1], 0.0, H)
    p[:, 0] = np.clip(p[:, 0], SL * p[:, 1], 1.0 - SR * p[:, 1])
    return p


def random_points(rng):
    y = rng.uniform(0, H, N)
    u = rng.uniform(0, 1, N)
    return np.c_[SL * y + u * (1 - SR * y - SL * y), y]


def anneal(a, rng, iters=60000, t0=0.002, s0=0.15, p=None):
    p = random_points(rng) if p is None else p.copy()
    v = value(p, a)
    bp, bv = p.copy(), v
    for it in range(iters):
        f = 1.0 - it / iters
        q = p.copy()
        i = rng.integers(N)
        q[i] += rng.normal(0, s0 * f + 0.002, 2)
        q = clip(q)
        w = value(q, a)
        if w >= v or rng.random() < np.exp((w - v) / (t0 * f * f + 1e-12)):
            p, v = q, w
            if v > bv:
                bp, bv = p.copy(), v
    return bp, bv


def polish(p, a, rounds=8):
    """SLSQP on an active-set model: signs of all triangles fixed, the set of
    triangles above the median fixed; maximise mix(t, u, a) with
    area_k >= t for all k and area_k >= u for the upper half."""
    best, bv = clip(p), value(clip(p), a)
    for _ in range(rounds):
        sa = signed_areas(best)
        sg = np.where(sa >= 0, 1.0, -1.0)
        top = np.argsort(np.abs(sa))[HALF - 1:]   # ranks 143..286: all >= u  =>  median >= u
        mn, md = stats(best)
        z0 = np.concatenate([best.ravel(), [mn, np.abs(sa)[top].min()]])

        def cons(z):
            q = z[:2 * N].reshape(N, 2)
            ar = sg * signed_areas(q)
            x, y = q[:, 0], q[:, 1]
            return np.concatenate([ar - z[-2], ar[top] - z[-1], y, H - y, x - SL * y, 1 - SR * y - x])

        res = minimize(lambda z: -mix(z[-2], z[-1], a), z0, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": cons}],
                       options={"maxiter": 200, "ftol": 1e-12})
        q = clip(res.x[:2 * N].reshape(N, 2))
        v = value(q, a)
        if v > bv + 1e-12:
            best, bv = q, v
        else:
            break
    return best, bv


def solve(a, starts, seed=0, iters=60000):
    rng = np.random.default_rng(seed)
    best, bv = None, -np.inf
    for _ in range(starts):
        p, _ = anneal(a, rng, iters)
        p, v = polish(p, a)
        if v > bv:
            best, bv = p, v
    return best, bv


def refine(p, a, rng, rounds=30, iters=4000):
    """Basin hopping: kick a few points, greedy descent, SLSQP polish, keep if better."""
    best, bv = polish(p, a)
    for _ in range(rounds):
        q = best.copy()
        idx = rng.choice(N, rng.integers(1, 4), replace=False)
        q[idx] += rng.normal(0, 0.05, (len(idx), 2))
        q, _ = anneal(a, rng, iters, t0=0.0, s0=0.02, p=clip(q))
        q, v = polish(q, a)
        if v > bv:
            best, bv = q, v
    return best, bv


def job(args):
    a, seed, starts, iters, rounds = args
    rng = np.random.default_rng(seed)
    p, _ = solve(a, starts, seed, iters)
    return refine(p, a, rng, rounds)


def job2(args):
    p, a, seed, rounds = args
    return refine(p, a, np.random.default_rng(1000 + seed), rounds)
