# EVOLVE-BLOCK-START
import numpy as np


def _cost(s, a, b, c):
    t = 0
    tot = 0
    for j in s:
        t += a[j]
        if t > b[j]:
            tot += c[j] * (t - b[j])
    return tot


def _atc(a, b, c, k):
    n = len(a)
    pbar = float(np.mean(a))
    rem = list(range(n))
    t = 0
    s = []
    while rem:
        best, bv = None, -1.0
        for j in rem:
            v = (c[j] / a[j]) * np.exp(-max(b[j] - a[j] - t, 0) / (k * pbar))
            if v > bv:
                bv, best = v, j
        s.append(best)
        rem.remove(best)
        t += a[best]
    return s


def _mdd(a, b, c):
    rem = list(range(len(a)))
    t = 0
    s = []
    while rem:
        j = min(rem, key=lambda j: (max(t + a[j], b[j]) / c[j], b[j]))
        s.append(j)
        rem.remove(j)
        t += a[j]
    return s


def _local(s, a, b, c):
    s = list(s)
    best = _cost(s, a, b, c)
    n = len(s)
    improved = True
    while improved:
        improved = False
        # insertion moves
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                t = s[:]
                x = t.pop(i)
                t.insert(j, x)
                v = _cost(t, a, b, c)
                if v < best:
                    s, best, improved = t, v, True
        # pairwise swaps
        for i in range(n - 1):
            for j in range(i + 1, n):
                t = s[:]
                t[i], t[j] = t[j], t[i]
                v = _cost(t, a, b, c)
                if v < best:
                    s, best, improved = t, v, True
    return s, best


def f(a, b, c):
    n = len(a)
    cands = [
        list(np.argsort(b, kind="stable")),
        list(np.argsort(a / c, kind="stable")),
        _mdd(a, b, c),
    ]
    for k in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        cands.append(_atc(a, b, c, k))
    cands.sort(key=lambda s: _cost(s, a, b, c))
    best_s, best_v = None, None
    for s in cands[:3]:
        s2, v = _local(s, a, b, c)
        if best_v is None or v < best_v:
            best_s, best_v = s2, v
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
