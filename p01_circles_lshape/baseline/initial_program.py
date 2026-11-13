# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 21
    p = np.zeros((n, 2))

    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1

    q = g(p)

    return p, q


def g(p):
    n = p.shape[0]
    q = np.zeros(n)

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = min(u, v, 1 - u, 1 - v, w)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

            if q[i] + q[j] > d:
                s = d / (q[i] + q[j])
                q[i] *= s
                q[j] *= s

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
