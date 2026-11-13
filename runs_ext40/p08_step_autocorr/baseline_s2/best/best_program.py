# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    # Points at the centre of each of the n equal intervals of [0, 1]
    x = (np.arange(n) + 0.5) / n

    # Smooth hump (beta-like) with a small rise at the left edge.
    # These values set the shape and can be tuned.
    alpha, beta = 1.5, 2.3
    hump = x ** (alpha - 1.0) * (1.0 - x) ** (beta - 1.0)
    # Spikes at both ends tend to flatten the self-convolution
    edge = 0.35 * np.exp(-x / 0.08) + 0.2 * np.exp(-(1.0 - x) / 0.06)
    base = 0.15  # small constant floor: keeps every weight above zero

    a = g(base + hump / np.max(hump) + edge)

    return g(h(a))


def h(a, iters=6000):
    # Minimise the peak of the self-convolution with gradient descent on a
    # smoothed peak (log-sum-exp). Writing a = softmax(w) keeps a positive.
    n = len(a)
    w = np.log(np.maximum(a, 1e-12))
    best, best_v = a.copy(), np.max(np.convolve(a, a))
    for t in range(iters):
        p = np.exp(w - w.max())
        p /= p.sum()
        c = np.convolve(p, p) * (2 * n)
        v = c.max() / (2 * n)
        if v < best_v:
            best, best_v = p.copy(), v
        b = 10.0 * (1.0 + t / 60.0)
        s = np.exp(b * (c - c.max()))
        s /= s.sum()
        gp = 2.0 * np.correlate(s, p, 'valid')
        gw = p * (gp - np.dot(p, gp))
        w -= 0.4 / np.sqrt(1.0 + t / 30.0) * gw / (np.linalg.norm(gw) + 1e-15)
    return best


def g(a):
    a = np.maximum(np.nan_to_num(np.asarray(a, float), posinf=0.0, neginf=0.0), 0.0)
    s = a.sum()
    return a / s if s > 0 else np.ones(len(a)) / len(a)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
