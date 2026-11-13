# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Edge-weighted (U-shaped) profile instead of a flat one.
    x = (np.arange(m) + 0.5) / m
    base = 1.0 + 0.6 * (np.abs(2 * x - 1) ** 4)
    h[:m] = base

    # Light deterministic local search minimizing max autoconvolution / mass^2
    def score(v):
        v = np.maximum(v, 0)
        s = v.sum()
        if s <= 0:
            return np.inf
        return np.max(np.convolve(v, v)) / (s * s)

    rng = np.random.default_rng(0)
    best = h.copy()
    best_s = score(best)
    step = 0.1
    for it in range(4000):
        cand = best + step * rng.standard_normal(n) * best.mean()
        cand = np.maximum(cand, 0)
        cs = score(cand)
        if cs < best_s:
            best, best_s = cand, cs
        if it % 1000 == 999:
            step *= 0.5
    h = best / best.sum()

    h = g(h)

    return h


def g(h):
    n = h.shape[0]
    q = np.zeros(n)

    for k in range(n):
        q[k] = min(max(h[k], -2.0), 2.0)

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
