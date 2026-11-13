# EVOLVE-BLOCK-START
import numpy as np


D = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
              for k in (-1, 0, 1) if i or j or k], float)
D /= np.linalg.norm(D, axis=1)[:, None]


def g(q, st, tol=1e-13):
    q = np.clip(np.asarray(q, float), -4.0, 4.0)
    u = h(q)
    while st > tol:
        Y = np.clip(q + st * D, -4.0, 4.0)
        F = np.array([h(y) for y in Y])
        k = F.argmin()
        if F[k] < u:
            q, u = Y[k], F[k]
        else:
            st *= 0.5
    return q, u


def f():
    r = np.random.default_rng(0)
    S = [c[2] + r.normal(0, 0.04, 3) for _ in range(8)]
    S += [c[1] + t * (c[2] - c[1]) for t in np.linspace(0.5, 1.2, 8)]
    S += list(c) + list(r.uniform(-4.0, 4.0, (40, 3)))
    S.sort(key=h)
    p, v = min((g(q, 0.05) for q in S[:4]), key=lambda t: t[1])
    x, u = g(p, 1e-4, 1e-15)
    return np.clip(x if u < v else p, -4.0, 4.0)


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
