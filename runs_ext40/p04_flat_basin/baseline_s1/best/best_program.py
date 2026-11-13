# EVOLVE-BLOCK-START
import numpy as np


def f():
    r = np.random.default_rng(0)
    g = globals()
    C = g['c']
    S = [C[i] for i in range(len(C))]
    S += [(C[i] + C[j]) / 2 for i in range(len(C)) for j in range(i + 1, len(C))]
    S += list(r.uniform(-4.0, 4.0, (40, 3)))
    S = sorted(S, key=lambda q: h(q))[:10]

    p, v = None, np.inf
    for q in S:
        q = np.clip(np.array(q, float), -4, 4)
        u = h(q)
        st = 0.5
        while st > 1e-9:
            ok = False
            D = list(np.eye(3)) + list(r.normal(size=(6, 3)))
            for d in D:
                d = d / np.linalg.norm(d)
                for sg in (1, -1):
                    z = np.clip(q + sg * st * d, -4, 4)
                    w = h(z)
                    if w < u:
                        q, u, ok = z, w, True
            if not ok:
                st *= 0.5
        if u < v:
            p, v = q, u
    return p


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
