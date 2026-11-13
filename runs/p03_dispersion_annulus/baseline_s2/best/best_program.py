# EVOLVE-BLOCK-START
import numpy as np


def f():
    # Two staggered rings: 12 on the outer boundary, 6 on an inner ring
    # offset by 15 deg so inner points sit between outer neighbours.
    a = 2.0 * np.pi * np.arange(12) / 12
    b = 2.0 * np.pi * np.arange(6) / 6 + np.pi / 12
    r = 0.505
    p = np.vstack([np.c_[np.cos(a), np.sin(a)],
                   r * np.c_[np.cos(b), np.sin(b)]])
    return g(p)


def g(p):
    # Project points into the annulus 0.4 <= |p| <= 1.0
    d = np.hypot(p[:, 0], p[:, 1])[:, None]
    return p * (np.clip(d, 0.4, 1.0) / d)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
