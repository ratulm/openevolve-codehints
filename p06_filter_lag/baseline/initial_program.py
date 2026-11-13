# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    for k in range(m):
        h[k] = 1.0 / m

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
