# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Sampled 1/sqrt(x) profile: its autoconvolution is nearly flat, so its peak
    # is lower than the triangular peak of a flat step function.
    for k in range(m):
        h[k] = 1.0 / np.sqrt(k + 0.5)

    h = h / np.sum(h)

    h = opt(h)

    # Basin-hopping: perturb the best solution multiplicatively and re-optimise.
    rng = np.random.default_rng(0)
    bv = peak(h)
    for r in range(12):
        s = 0.3 * (0.85 ** r)
        x0 = h * np.exp(s * rng.standard_normal(n))
        x0 /= x0.sum()
        y = opt(x0, lr0=0.02)
        v = peak(y)
        if v < bv:
            bv, h = v, y

    h = g(h)

    return h


def peak(x):
    s = np.sum(x)
    return np.max(np.convolve(x, x)) / (s * s) if s > 0 else np.inf


def opt(h0, lr0=0.05):
    # Smoothed-max (softmax) projected gradient descent on autoconvolution peak.
    x = h0.copy()
    best, bv = x.copy(), peak(x)
    for beta in [20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]:
        x = best.copy()
        lr = lr0
        for _ in range(1500):
            c = np.convolve(x, x)
            w = np.exp(beta * (c / c.max() - 1.0))
            w /= w.sum()
            gr = 2.0 * np.correlate(w, x, 'valid')
            gr -= gr.mean()
            gm = np.abs(gr).max()
            if gm <= 0:
                break
            x = np.maximum(x - lr * gr / gm * x.mean(), 0.0)
            x /= x.sum()
            v = peak(x)
            if v < bv:
                bv, best = v, x.copy()
            lr *= 0.998
    return best


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
