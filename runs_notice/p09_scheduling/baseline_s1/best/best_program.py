# EVOLVE-BLOCK-START
import numpy as np


def _cost(s, a, b, c):
    t = 0
    r = 0
    for j in s:
        t += a[j]
        if t > b[j]:
            r += c[j] * (t - b[j])
    return r


def _atc(a, b, c, K):
    n = len(a)
    pbar = float(np.mean(a))
    left = list(range(n))
    t = 0
    s = []
    while left:
        best, bv = None, -1.0
        for j in left:
            v = (c[j] / a[j]) * np.exp(-max(0.0, b[j] - a[j] - t) / (K * pbar))
            if v > bv:
                bv, best = v, j
        s.append(best)
        left.remove(best)
        t += a[best]
    return s


def _local(s, a, b, c):
    s = list(s)
    n = len(s)
    best = _cost(s, a, b, c)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                t = s[:]
                x = t.pop(i)
                t.insert(j, x)
                v = _cost(t, a, b, c)
                if v < best:
                    best, s, improved = v, t, True
        for i in range(n - 1):
            for j in range(i + 1, n):
                t = s[:]
                t[i], t[j] = t[j], t[i]
                v = _cost(t, a, b, c)
                if v < best:
                    best, s, improved = v, t, True
    return s, best


def f(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(a)
    starts = [list(np.argsort(b, kind="stable")),
              list(np.argsort(a / c, kind="stable"))]
    for K in (0.5, 1.0, 2.0, 3.0, 5.0):
        starts.append(_atc(a, b, c, K))
    rng = np.random.RandomState(0)
    gs, gv = None, None
    for st in starts:
        s, v = _local(st, a, b, c)
        if gv is None or v < gv:
            gs, gv = s, v
    for it in range(150):
        if gv == 0:
            break
        t = gs[:]
        m = it % 3
        if m == 0:
            i, j = sorted(rng.randint(0, n, 2))
            t[i:j + 1] = t[i:j + 1][::-1]
        elif m == 1:
            i = rng.randint(n); L = rng.randint(2, 6)
            seg = t[i:i + L]; del t[i:i + L]
            k = rng.randint(len(t) + 1)
            t[k:k] = seg
        else:
            for _ in range(2 + it % 4):
                i, j = rng.randint(n), rng.randint(n)
                t[i], t[j] = t[j], t[i]
        s, v = _local(t, a, b, c)
        if v <= gv:
            gs, gv = s, v
    return np.array(gs, dtype=int)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
a = np.array([6, 11, 12, 5, 3, 8, 9, 10, 9, 9, 12, 12, 12, 11, 9, 12, 2, 2, 10, 6, 10, 7, 11, 2, 9, 2, 3, 11, 5,
              12, 5, 10, 12, 5, 12, 9])
b = np.array([22, 170, 23, 40, 66, 110, 138, 13, 164, 135, 56, 33, 105, 105, 47, 29, 135, 119, 141, 38, 128, 13,
              1, 31, 138, 95, 109, 104, 112, 57, 127, 2, 66, 9, 22, 160])
c = np.array([9, 3, 5, 7, 9, 3, 5, 8, 8, 9, 2, 9, 2, 8, 8, 9, 8, 7, 6, 5, 1, 9, 9, 1, 6, 4, 7, 6, 6, 5, 2, 8,
              8, 2, 5, 1])


def run():
    return f(a, b, c)
