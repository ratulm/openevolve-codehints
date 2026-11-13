# EVOLVE-BLOCK-START
import numpy as np


def f(a, b, c):
    p, d, w = list(a), list(b), list(c)
    n = len(p)
    m = sum(p) / n

    def cost(s):
        t = r = 0
        for j in s:
            t += p[j]
            r += w[j] * max(t - d[j], 0)
        return r

    def atc(K):
        u, t, s = set(range(n)), 0, []
        while u:
            k = max(u, key=lambda j: w[j] / p[j] * np.exp(-max(d[j] - p[j] - t, 0) / K / m))
            s.append(k); u.remove(k); t += p[k]
        return s

    s = min([sorted(range(n), key=d.__getitem__)] + [atc(K) for K in (1, 2, 3)], key=cost)
    best, imp = cost(s), True
    while imp:
        imp = False
        for i in range(n):
            for j in range(n):
                q = s[:i] + s[i + 1:]
                q.insert(j, s[i])
                v = cost(q)
                if v < best:
                    s, best, imp = q, v, True
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
