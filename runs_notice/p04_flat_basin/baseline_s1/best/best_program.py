# EVOLVE-BLOCK-START
import numpy as np


def _ref(x, fx, st=0.5):
    x = np.array(x, dtype=float)
    E = np.eye(3)
    while st > 1e-9:
        imp = False
        for e in E:
            for sg in (1.0, -1.0):
                y = np.clip(x + sg * st * e, -4.0, 4.0)
                fy = h(y)
                if fy < fx:
                    x, fx, imp = y, fy, True
        if not imp:
            st *= 0.5
    return x, fx


def f():
    r = np.random.default_rng(0)
    S = [np.array(z, dtype=float) for z in c]
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            S.append(0.5 * (c[i] + c[j]))
    S += list(r.uniform(-4.0, 4.0, (3000, 3)))
    V = np.array([h(z) for z in S])
    best, bv = None, np.inf
    for k in np.argsort(V)[:12]:
        x, fx = _ref(S[k], V[k])
        if fx < bv:
            best, bv = x, fx
    return np.clip(best, -4.0, 4.0)


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
