# EVOLVE-BLOCK-START
import numpy as np


# 6 axis + 8 diagonal directions (diagonals follow tilted valleys)
E = np.vstack([np.eye(3), -np.eye(3),
               np.array([[i, j, k] for i in (1, -1) for j in (1, -1)
                         for k in (1, -1)]) / np.sqrt(3.0)])


def _ls(x, fx, st, tol):
    # best-improvement pattern search: grow step on success, shrink on failure
    x = np.clip(np.asarray(x, float), -4.0, 4.0)
    while st > tol:
        Y = np.clip(x + st * E, -4.0, 4.0)
        F = np.array([h(y) for y in Y])
        k = int(np.argmin(F))
        if F[k] < fx:
            x, fx, st = Y[k], F[k], st * 1.5
        else:
            st *= 0.6
    return x, fx


def _newton(x, fx, e=1e-4, it=6):
    # finite-difference Newton polish; accept only improving steps
    I = np.eye(3) * e
    for _ in range(it):
        g = np.array([(h(x + d) - h(x - d)) / (2 * e) for d in I])
        H = np.array([[(h(x + di + dj) - h(x + di - dj) - h(x - di + dj)
                        + h(x - di - dj)) / (4 * e * e) for dj in I] for di in I])
        H = 0.5 * (H + H.T)
        try:
            np.linalg.cholesky(H)
            y = x - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            y = x - 1e-3 * g / (np.linalg.norm(g) + 1e-300)
        y = np.clip(y, -4.0, 4.0)
        fy = h(y)
        if not fy < fx:
            break
        x, fx = y, fy
    return x, fx


def f():
    # c[2] (narrow, deep) nested in c[1] is the global basin; keep a few others as a guard
    S = [c[0], c[1], c[2], c[3], 0.5 * (c[1] + c[2]),
         c[2] + 0.1 * (c[1] - c[2]), c[2] - 0.1 * (c[1] - c[2])]
    R = [_ls(q, h(q), 0.1, 1e-6) for q in S]
    p, v = min(R, key=lambda t: t[1])
    p, v = _ls(p, v, 1e-3, 1e-13)
    p, v = _newton(p, v)
    p, v = _ls(p, v, 1e-5, 1e-14)
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
