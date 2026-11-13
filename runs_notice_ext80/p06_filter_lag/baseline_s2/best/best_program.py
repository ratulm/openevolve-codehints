# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Bin-averaged arcsine density (exact via its CDF), normalised to unit mass.
    edges = np.linspace(-1.0, 1.0, m + 1)
    w = np.diff(np.arcsin(edges) / np.pi + 0.5)
    alpha = 5.0
    w = np.maximum(alpha * w + (1.0 - alpha) / m, 0.0)
    h[:m] = w / np.sum(w)

    h = g(h)

    return h


def g(h):
    n = h.shape[0]
    return np.clip(np.asarray(h, dtype=float).reshape(n), -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
