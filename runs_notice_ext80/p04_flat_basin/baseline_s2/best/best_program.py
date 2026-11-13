# EVOLVE-BLOCK-START
import numpy as np


def _ps(x, fx, st, tol=1e-10, cap=4000):
    x = np.array(x, dtype=float)
    n = 0
    while st > tol and n < cap:
        moved = False
        for k in range(3):
            for sg in (1.0, -1.0):
                y = x.copy()
                y[k] = np.clip(y[k] + sg * st, -4.0, 4.0)
                fy = h(y)
                n += 1
                if fy < fx:
                    x, fx, moved = y, fy, True
                    break
        if not moved:
            st *= 0.5
    return x, fx


def _polish(x, fx, st=1e-3, tol=1e-13, cap=20000):
    n = 0
    while st > tol and n < cap:
        best, bv = None, fx
        for k in range(3):
            for sg in (1.0, -1.0):
                y = x.copy()
                y[k] = np.clip(y[k] + sg * st, -4.0, 4.0)
                fy = h(y)
                n += 1
                if fy < bv:
                    best, bv = y, fy
        if best is None:
            st *= 0.5
        else:
            x, fx = best, bv
    return x, fx


def f():
    r = np.random.default_rng(0)
    starts = [np.array(ci, dtype=float) for ci in c]
    starts += [c[2] + t * (c[1] - c[2]) for t in (0.1, 0.25, 0.5)]
    starts += list(c[2] + r.normal(0, 0.05, (6, 3)))
    starts += list(r.uniform(-4.0, 4.0, (24, 3)))
    vals = [h(q) for q in starts]
    order = np.argsort(vals)[:12]
    p, v = None, np.inf
    for i in order:
        x, fx = _ps(starts[i], vals[i], 0.03, tol=1e-8)
        if fx < v:
            p, v = x, fx
    p, v = _polish(p, v)
    return np.clip(p, -4.0, 4.0)


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
