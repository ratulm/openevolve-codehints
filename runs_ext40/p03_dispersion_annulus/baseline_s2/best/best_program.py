# EVOLVE-BLOCK-START
import numpy as np


def f():
    # Two interleaved rings: 12 on outer boundary, 6 on inner ring offset by 15 deg
    a = np.pi * np.arange(12) / 6
    b = np.pi * np.arange(6) / 3 + np.pi / 12
    # Score rose monotonically as r shrank (0.5176 -> 0.48); continue toward 0.4 limit.
    r = 0.44
    p = np.vstack([np.c_[np.cos(a), np.sin(a)], r * np.c_[np.cos(b), np.sin(b)]])
    return g(p)


def g(p):
    d = np.hypot(p[:, 0], p[:, 1])[:, None]
    return p * np.clip(d, 0.4, 1.0) / d


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
