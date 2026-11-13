# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Edge-heavy (arcsine-like) profile instead of a flat one: a flat step
    # function gives a sharply peaked triangular autoconvolution.
    x = (np.arange(m) + 0.5) / m * 2.0 - 1.0
    w = 1.0 / np.sqrt(np.maximum(1.0 - x * x, 1e-3))
    w = 0.6 * w / w.mean() + 0.4
    h[:m] = w / w.sum()

    # Short multiplicative refinement that lowers max(h*h) / (sum h)^2.
    best = h.copy()
    best_val = np.max(np.convolve(h, h)) / (h.sum() ** 2)
    eta = 0.05
    for _ in range(3000):
        c = np.convolve(h, h)
        p = int(np.argmax(c))
        grad = np.zeros(n)
        for i in range(n):
            j = p - i
            if 0 <= j < n:
                grad[i] = 2.0 * h[j]
        grad = grad / (np.max(grad) + 1e-12)
        h = h * np.exp(-eta * grad)
        h = h / h.sum()
        val = np.max(np.convolve(h, h)) / (h.sum() ** 2)
        if val < best_val:
            best_val = val
            best = h.copy()
        else:
            eta *= 0.995

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
