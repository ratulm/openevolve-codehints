# EVOLVE-BLOCK-START
import numpy as np


def _cost(s, a, b, c):
    t = 0.0
    tot = 0.0
    for j in s:
        t += a[j]
        if t > b[j]:
            tot += c[j] * (t - b[j])
    return tot


def _mdd(a, b, c):
    rem = list(range(len(a)))
    t, s = 0.0, []
    while rem:
        j = min(rem, key=lambda j: (max(t + a[j], b[j]) / c[j], b[j]))
        s.append(j)
        rem.remove(j)
        t += a[j]
    return s


def _atc(a, b, c, k):
    n = len(a)
    pbar = a.mean()
    left = list(range(n))
    t, s = 0, []
    while left:
        best, bv = None, -1.0
        for j in left:
            v = c[j] / a[j] * np.exp(-max(0, b[j] - a[j] - t) / (k * pbar))
            if v > bv:
                bv, best = v, j
        s.append(best)
        left.remove(best)
        t += a[best]
    return s


def _local(s, a, b, c):
    s = list(s)
    best = _cost(s, a, b, c)
    n = len(s)
    improved = True
    while improved:
        improved = False
        # pairwise swaps
        for i in range(n - 1):
            for j in range(i + 1, n):
                s[i], s[j] = s[j], s[i]
                v = _cost(s, a, b, c)
                if v < best - 1e-9:
                    best, improved = v, True
                else:
                    s[i], s[j] = s[j], s[i]
        # insertions (best position per element)
        for i in range(n):
            x = s[i]
            r = s[:i] + s[i + 1:]
            bj, bvv = -1, best
            for j in range(n):
                if j == i:
                    continue
                r.insert(j, x)
                v = _cost(r, a, b, c)
                r.pop(j)
                if v < bvv - 1e-9:
                    bj, bvv = j, v
            if bj >= 0:
                r.insert(bj, x)
                s, best, improved = r, bvv, True
    return s, best


def f(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    starts = [list(np.argsort(b, kind="stable")),
              list(np.argsort(-c / a, kind="stable")),
              list(np.argsort(b - a, kind="stable"))]
    for k in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
        starts.append(_atc(a, b, c, k))
    al, bl, cl = a.tolist(), b.tolist(), c.tolist()
    starts.append(_mdd(al, bl, cl))
    bs, bv = None, float("inf")
    for st in starts:
        s, v = _local([int(x) for x in st], al, bl, cl)
        if v < bv:
            bs, bv = s, v
    return np.array(bs, dtype=int)


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
