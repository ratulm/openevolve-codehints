# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 13
    p = np.zeros((n, 2))

    for i in range(n):
        t = 2 * np.pi * i / n
        p[i] = [0.52 + 0.2 * np.cos(t), 0.25 + 0.12 * np.sin(t)]

    return h(g(p))


def a(q):
    # smallest triangle area over all triples of points
    import itertools
    T = np.array(list(itertools.combinations(range(q.shape[0]), 3)))
    x, y, z = q[T[:, 0]], q[T[:, 1]], q[T[:, 2]]
    return np.min(np.abs((y[:, 0] - x[:, 0]) * (z[:, 1] - x[:, 1])
                         - (z[:, 0] - x[:, 0]) * (y[:, 1] - x[:, 1]))) / 2


def h(q, it=30000):
    # deterministic hill-climb that only accepts non-worsening moves
    r = np.random.default_rng(1)
    b, s = a(q), 0.08
    for k in range(it):
        c = q.copy()
        i = r.integers(q.shape[0])
        c[i] += r.normal(0, s, 2)
        c = g(c)
        m = a(c)
        if m >= b:
            q, b = c, m
        if k % 3000 == 2999:
            s *= 0.6
    return q


def g(p):
    q = np.array(p, dtype=float)

    for i in range(q.shape[0]):
        v = min(max(q[i, 1], 0.0), 0.5)
        u = min(max(q[i, 0], 0.6 * v), 1.0 - 0.2 * v)
        q[i] = [u, v]

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
