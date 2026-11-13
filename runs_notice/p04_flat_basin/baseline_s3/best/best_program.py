# EVOLVE-BLOCK-START
import numpy as np


def f():
    r = np.random.default_rng(0)
    starts = [np.array(x, dtype=float) for x in c]
    starts += list(r.uniform(-4.0, 4.0, (300, 3)))
    starts.sort(key=h)
    bp, bv = None, np.inf
    for p0 in starts[:8]:
        p = np.clip(p0, -4.0, 4.0)
        v = h(p)
        st = 0.2
        while st > 1e-10:
            imp = False
            for _ in range(30):
                q = np.clip(p + r.normal(0.0, st, 3), -4.0, 4.0)
                u = h(q)
                if u < v:
                    p, v, imp = q, u, True
            if not imp:
                st *= 0.5
        for k in range(3):
            for e in (1e-6, -1e-6, 1e-8, -1e-8):
                q = p.copy()
                q[k] = np.clip(q[k] + e, -4.0, 4.0)
                u = h(q)
                if u < v:
                    p, v = q, u
        if v < bv:
            bp, bv = p, v
    return bp


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
import numpy as np

c = np.array([
    [1.8, -1.6, 1.2],
    [-2.0, 1.5, -1.0],
    [-1.85, 1.4, -0.9],
    [2.5, 2.4, -2.6],
    [-2.6, -2.4, 2.2],
])
a = np.array([9.0, 6.5, 4.5, 6.0, 4.0])
s = np.array([1.0, 0.8, 0.15, 0.3, 0.8])


def h(x):
    x = np.asarray(x, dtype=float)
    d = np.sum((x - c) ** 2, axis=1)
    w = np.sum(a * np.exp(-d / (2.0 * s ** 2)))
    t = (0.5 * np.sin(2.3 * x[0] + 1.1 * x[1]) * np.cos(1.7 * x[2] - 0.6 * x[0])
         + 0.3 * np.cos(2.9 * x[1] - 1.3 * x[2]))
    b = 0.02 * np.sum(x ** 2)
    o = 50.0 * np.sum(np.maximum(np.abs(x) - 4.0, 0.0))
    return 13.0 - w + t + b + o


def run():
    return f()
