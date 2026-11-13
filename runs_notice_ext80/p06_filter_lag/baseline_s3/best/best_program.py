# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    # Edge-heavy arcsine (Chebyshev) profile, sharpened by an exponent.
    # Gamma scan: 3.5:.4106, 5.0:.4158, 7.0:.4157, 8.0:.4156 -> peak ~5-6
    edges = np.linspace(-1.0, 1.0, n + 1)
    w = np.arcsin(edges[1:]) - np.arcsin(edges[:-1])
    w = w ** 5.5
    h = w / np.sum(w)

    h = g(h)

    return h


def g(h):
    return np.clip(h, -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
