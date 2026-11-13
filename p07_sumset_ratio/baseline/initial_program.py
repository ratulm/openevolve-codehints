# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 300
    k = 60
    n = 40
    d = 7

    a = np.arange(n) * d

    a = np.unique(a)
    a = a[(a >= 0) & (a < m)]
    a = a[:k]

    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
