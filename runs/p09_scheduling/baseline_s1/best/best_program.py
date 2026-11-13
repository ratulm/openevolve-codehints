# EVOLVE-BLOCK-START
import numpy as np


def f(a, b, c):
    n = len(a)
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)

    def T(s):
        t = np.cumsum(a[s])
        return float(np.sum(c[s] * np.maximum(0.0, t - b[s])))

    cands = [list(np.argsort(b, kind="stable")), list(np.argsort(-c / a, kind="stable"))]
    pm = a.mean()
    for K in (0.5, 1, 2, 3, 5):
        u = set(range(n)); t = 0.0; s = []
        while u:
            k = max(u, key=lambda j: c[j] / a[j] * np.exp(-max(0.0, b[j] - a[j] - t) / (K * pm)))
            s.append(k); u.remove(k); t += a[k]
        cands.append(s)
    best = min(cands, key=T)
    bv = T(best)

    imp = True
    while imp:
        imp = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = best[:i] + best[i + 1:]
                s.insert(j, best[i])
                v = T(s)
                if v < bv - 1e-9:
                    best, bv, imp = s, v, True
        for i in range(n - 1):
            s = best[:]
            s[i], s[i + 1] = s[i + 1], s[i]
            v = T(s)
            if v < bv - 1e-9:
                best, bv, imp = s, v, True

    return np.array(best)


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
