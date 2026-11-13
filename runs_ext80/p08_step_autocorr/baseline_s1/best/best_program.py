# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    x = (np.arange(n) + 0.5) / n

    # Arcsine-like start (known good for low autoconvolution peaks).
    a = 1.0 / np.sqrt(x * (1.0 - x))
    a = g(0.6 * a / a.mean() + 0.4)

    a = mirror_descent(a, [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
    a = refine(a)
    a = mirror_descent(a, [3000.0, 10000.0, 30000.0, 100000.0], lr0=0.02)
    a = refine(a, iters=6000, seed=1, step0=0.05)
    return a


def peak(p):
    return np.max(np.convolve(p, p))


def mirror_descent(a, betas, steps=1500, lr0=0.05):
    # Exponentiated-gradient on a softmax-smoothed max of the
    # autoconvolution; keeps positivity and unit mass automatically.
    n = a.shape[0]
    best, best_val = a.copy(), peak(a)
    for beta in betas:
        w = best.copy()
        lr = lr0
        for t in range(steps):
            c = np.convolve(w, w)
            z = np.exp(beta * n * (c - c.max()))
            z /= z.sum()
            gr = 2.0 * np.correlate(z, w, 'valid')
            gr = (gr - gr @ w) / (np.abs(gr).max() + 1e-15)
            w = g(w * np.exp(-lr * gr))
            v = peak(w)
            if v < best_val:
                best, best_val = w.copy(), v
            if t % 500 == 499:
                lr *= 0.6
    return best


def refine(a, iters=6000, seed=0, step0=0.1):
    # Seeded local search: single-bin rescale or neighbour mass swap,
    # accepting only strict improvements of the peak.
    rng = np.random.default_rng(seed)
    best = a.copy()
    best_val = peak(best)
    n = best.shape[0]
    step = step0
    for t in range(iters):
        cand = best.copy()
        i = rng.integers(n)
        if t % 2 == 0:
            cand[i] *= np.exp(step * rng.standard_normal())
        else:
            j = (i + 1) % n
            d = step * 0.5 * min(cand[i], cand[j]) * rng.standard_normal()
            cand[i] += d
            cand[j] -= d
        cand = g(cand)
        v = peak(cand)
        if v < best_val:
            best, best_val = cand, v
        if t % 1500 == 1499:
            step *= 0.6
    return best


def g(a):
    a = np.maximum(a, 0.0)
    s = np.sum(a)

    if s <= 0.0:
        a = np.ones(a.shape[0])
        s = np.sum(a)

    return a / s


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
