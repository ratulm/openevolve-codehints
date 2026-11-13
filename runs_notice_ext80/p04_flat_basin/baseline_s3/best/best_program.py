# EVOLVE-BLOCK-START
import numpy as np


def _ls(x, fx, step, tol, it):
    # pattern search: axis moves plus diagonal moves; shrink step on failure
    D = [np.eye(3)[i] * sg for i in range(3) for sg in (1, -1)]
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                D.append(np.array([sx, sy, sz]) / np.sqrt(3))
    for _ in range(it):
        imp = False
        for d in D:
            y = np.clip(x + step * d, -4.0, 4.0)
            fy = h(y)
            if fy < fx:
                x, fx, imp = y, fy, True
        if not imp:
            step *= 0.5
            if step < tol:
                break
    return x, fx


def f():
    r = np.random.default_rng(0)
    seeds = [np.array(ci, dtype=float) for ci in c]
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            seeds.append(0.5 * (c[i] + c[j]))
    seeds += [r.uniform(-4.0, 4.0, 3) for _ in range(40)]

    # cheap screening: rank seeds by raw value, only refine the best few
    pre = sorted(((h(q), q) for q in seeds), key=lambda z: z[0])[:8]
    cand = []
    for fq, q in pre:
        x, fx = _ls(np.array(q, float), fq, 0.1, 1e-4, 200)
        cand.append((fx, x))
    cand.sort(key=lambda z: z[0])

    p, v = None, np.inf
    for fx, x in cand[:3]:
        x, fx = _ls(x, fx, 0.01, 1e-13, 5000)
        if fx < v:
            p, v = x, fx

    # Newton polish with finite-difference gradient/Hessian (accept only improvements)
    e = 1e-5
    I = np.eye(3)
    for _ in range(8):
        g = np.array([(h(p + e * I[i]) - h(p - e * I[i])) / (2 * e) for i in range(3)])
        H = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                H[i, j] = (h(p + e * I[i] + e * I[j]) - h(p + e * I[i] - e * I[j])
                           - h(p - e * I[i] + e * I[j]) + h(p - e * I[i] - e * I[j])) / (4 * e * e)
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        y = np.clip(p + dx, -4.0, 4.0)
        fy = h(y)
        if fy < v:
            p, v = y, fy
        else:
            break
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
