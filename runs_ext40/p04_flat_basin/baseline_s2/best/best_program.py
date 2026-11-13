# EVOLVE-BLOCK-START
import numpy as np


def f():
    r = np.random.default_rng(0)
    C = np.asarray(c, dtype=float)
    pts = [p for p in C] + [0.5 * (C[1] + C[2])]
    pts += [r.uniform(-4.0, 4.0, 3) for _ in range(400)]
    pts.sort(key=h)

    best, bv = None, np.inf
    for x in pts[:10]:
        x = np.array(x, dtype=float)
        v = h(x)
        st = 1.0
        while st > 1e-10:
            moved = False
            for k in range(3):
                for sg in (1.0, -1.0):
                    y = x.copy()
                    y[k] = np.clip(y[k] + sg * st, -4.0, 4.0)
                    u = h(y)
                    if u < v:
                        x, v, moved = y, u, True
            if not moved:
                st *= 0.5
        if v < bv:
            best, bv = x, v
    return best


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
