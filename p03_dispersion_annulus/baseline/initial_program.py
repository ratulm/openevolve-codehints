# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 18
    p = np.zeros((n, 2))

    for i in range(n):
        a = 2.0 * np.pi * i / n
        p[i] = [0.55 * np.cos(a), 0.55 * np.sin(a)]

    return g(p)


def g(p):
    q = p.copy()

    for i in range(q.shape[0]):
        d = np.hypot(q[i, 0], q[i, 1])
        if d < 0.4:
            q[i] *= 0.4 / d
        elif d > 1.0:
            q[i] *= 1.0 / d

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
