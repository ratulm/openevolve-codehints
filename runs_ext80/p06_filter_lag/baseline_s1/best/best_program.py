# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    # Pure end-heavy taper, steeper again: the score has risen at each
    # step (-0.65 -> -0.85 -> -1.6 -> -2.0), so keep going in that direction.
    p = -3.0
    eps = 0.0  # optional flat floor so the centre doesn't drop to ~0
    x = (np.arange(n) + 0.5) / n
    w = (x * (1.0 - x)) ** p
    w = w / w.sum()
    w = (1.0 - eps) * w + eps / n
    return g(w / w.sum())


def g(h):
    return np.clip(h, -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
