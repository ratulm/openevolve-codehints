# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    a = np.zeros(n)

    # Symmetric profile, flat-ish in the middle with mildly raised edges
    # (between uniform and arcsine), which lowers the autoconvolution peak.
    for i in range(n):
        x = -1.0 + (2.0 * i + 1.0) / n
        a[i] = (1.0 - x * x) ** (-0.25)

    a = h(g(a))
    a = g(a)

    return a


def peak(a):
    s = np.sum(a)
    return 2.0 * len(a) * np.max(np.convolve(a, a)) / (s * s)


def h(a0):
    # Minimise smoothed max of autoconvolution with Adam on a = w^2.
    n = len(a0)
    w = np.sqrt(np.maximum(a0, 1e-12) * n)
    best = a0.copy()
    best_v = peak(a0)
    m = np.zeros(n)
    v = np.zeros(n)
    t = 0
    lr = 0.01
    for beta in (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0):
        for _ in range(3000):
            t += 1
            a = w * w
            s = np.sum(a)
            c = 2.0 * n * np.convolve(a, a) / (s * s)
            z = beta * (c - np.max(c))
            p = np.exp(z)
            p /= np.sum(p)
            M = np.dot(p, c)
            ga = 2.0 * n * 2.0 * np.correlate(p, a, 'valid') / (s * s) \
                - 2.0 * M / s
            gw = ga * 2.0 * w
            m = 0.9 * m + 0.1 * gw
            v = 0.999 * v + 0.001 * gw * gw
            mh = m / (1 - 0.9 ** t)
            vh = v / (1 - 0.999 ** t)
            w = w - lr * mh / (np.sqrt(vh) + 1e-12)
            w = w / np.sqrt(np.mean(w * w) + 1e-300)
            a = w * w
            val = peak(a)
            if np.isfinite(val) and val < best_v:
                best_v = val
                best = a / np.sum(a)
        lr *= 0.7
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
