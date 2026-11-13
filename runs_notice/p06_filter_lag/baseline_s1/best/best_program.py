# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Arcsine-like profile (heavier near the edges), blended with uniform.
    x = (np.arange(m) + 0.5) / m * 2.0 - 1.0  # bin centres in (-1, 1)
    w = (1.0 - x * x) ** -0.6
    w = 0.1 * np.ones(m) + 0.9 * w / w.mean()
    w = w / w.sum()  # keep the same total mass as the uniform baseline

    for k in range(m):
        h[k] = w[k]

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
