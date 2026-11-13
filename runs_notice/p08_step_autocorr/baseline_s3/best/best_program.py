# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    a = np.zeros(n)

    # Equal weights: maximally spread out, no single entry dominates
    for i in range(n):
        a[i] = 1.0

    a = g(a)

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
