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

    a = base + hump / np.max(hump) + edge

    a = g(a)

    return a


def g(a):
    a = np.maximum(np.nan_to_num(np.asarray(a, float), posinf=0.0, neginf=0.0), 0.0)
    s = a.sum()
    return a / s if s > 0 else np.ones(len(a)) / len(a)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
