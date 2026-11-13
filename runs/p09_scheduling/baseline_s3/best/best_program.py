# EVOLVE-BLOCK-START
import numpy as np


def _cost(s, a, b, c):
    t = np.cumsum(a[s])
    return float(np.sum(c[s] * np.maximum(0, t - b[s])))


def _atc(a, b, c, K=2.0):
    n = len(a)
    left = list(range(n))
    pbar = a.mean()
    t, s = 0, []
    while left:
        best = max(left, key=lambda j: c[j] / a[j] *
                   np.exp(-max(0, b[j] - a[j] - t) / (K * pbar)))
        s.append(best)
        left.remove(best)
        t += a[best]
    return s


def _improve(s, a, b, c):
    s = list(s)
    best = _cost(s, a, b, c)
    n = len(s)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                t = s[:i] + s[i + 1:]
                t.insert(j, s[i])
                v = _cost(t, a, b, c)
                if v < best:
                    s, best, improved = t, v, True
        for i in range(n - 1):
            t = s[:]
            t[i], t[i + 1] = t[i + 1], t[i]
            v = _cost(t, a, b, c)
            if v < best:
                s, best, improved = t, v, True
    return s, best


def f(a, b, c):
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    starts = [list(np.argsort(b, kind="stable")),
              list(np.argsort(a / c, kind="stable"))]
    for K in (0.5, 1.0, 2.0, 4.0):
        starts.append(_atc(a, b, c, K))
    best_s, best_v = None, float("inf")
    for s0 in starts:
        s, v = _improve(s0, a, b, c)
        if v < best_v:
            best_s, best_v = s, v
    return np.array(best_s)


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
