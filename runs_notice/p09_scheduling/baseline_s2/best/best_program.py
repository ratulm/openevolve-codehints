# EVOLVE-BLOCK-START
import numpy as np


def f(a, b, c):
    n = len(a)
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)

    def cost(s):
        t = np.cumsum(a[s])
        return float((c[s] * np.maximum(0.0, t - b[s])).sum())

    def atc(K):
        u = list(range(n)); s = []; t = 0.0; m = a.mean()
        while u:
            k = max(u, key=lambda j: c[j] / a[j] * np.exp(-max(0.0, b[j] - a[j] - t) / (K * m)))
            s.append(k); u.remove(k); t += a[k]
        return s

    cands = [list(np.argsort(b, kind='stable'))] + [atc(K) for K in (0.5, 1.0, 2.0, 3.0)]
    s = min(cands, key=cost)
    best = cost(s)
    imp = True
    while imp:
        imp = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                q = s[:]; x = q.pop(i); q.insert(j, x)
                v = cost(q)
                if v < best:
                    best, s, imp = v, q, True
        for i in range(n):
            for j in range(i + 1, n):
                q = s[:]; q[i], q[j] = q[j], q[i]
                v = cost(q)
                if v < best:
                    best, s, imp = v, q, True

    return np.array(s)


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
