# EVOLVE-BLOCK-START
import numpy as np


def _peak(a):
    a = g(a)
    c = np.convolve(a, a)
    return 2.0 * a.shape[0] * np.max(c)


def _optimize(a0, iters=4000):
    n = a0.shape[0]
    a = g(a0.copy())
    best = a.copy()
    best_val = _peak(a)
    beta = 50.0
    eta = 0.5
    for t in range(iters):
        c = np.convolve(a, a) * (2.0 * n)
        z = beta * (c - c.max())
        w = np.exp(z)
        w /= w.sum()
        grad = 2.0 * (2.0 * n) * np.correlate(w, a, mode="valid")
        grad = grad - np.dot(a, grad)
        a = a * np.exp(-eta * grad / (np.abs(grad).max() + 1e-12))
        a = np.maximum(a, 1e-15)
        a /= a.sum()
        v = _peak(a)
        if v < best_val:
            best_val = v
            best = a.copy()
        if (t + 1) % 500 == 0:
            beta = min(beta * 1.6, 5000.0)
            eta *= 0.7
            a = best.copy()
    return best, best_val


def f():
    n = 50
    x = (np.arange(n) + 0.5) / n
    init = 1.0 / np.sqrt(x * (1.0 - x))

    cands = []
    ramp = g(1.0 + np.arange(n, dtype=float))
    cands.append((_peak(ramp), ramp))
    flat = g(np.ones(n))
    cands.append((_peak(flat), flat))
    try:
        for start in (init, np.ones(n)):
            b, v = _optimize(start)
            if np.all(np.isfinite(b)):
                cands.append((v, b))
    except Exception:
        pass

    cands.sort(key=lambda p: p[0])
    a = g(cands[0][1])

    return a


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
