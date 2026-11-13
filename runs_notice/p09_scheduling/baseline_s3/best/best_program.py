# EVOLVE-BLOCK-START
import numpy as np


def f(a, b, c):
    n = len(a)
    p = [int(x) for x in a]
    d = [int(x) for x in b]
    w = [int(x) for x in c]

    def cost(s):
        t = 0
        z = 0
        for j in s:
            t += p[j]
            if t > d[j]:
                z += w[j] * (t - d[j])
        return z

    pbar = sum(p) / n
    starts = [sorted(range(n), key=lambda j: d[j])]
    for K in (0.5, 1.0, 2.0, 3.0, 4.0):
        s, u, t = [], set(range(n)), 0
        while u:
            k = max(u, key=lambda j: w[j] / p[j] *
                    np.exp(-max(0, d[j] - p[j] - t) / (K * pbar)))
            s.append(k)
            u.remove(k)
            t += p[k]
        starts.append(s)

    def ls(s):
        cur = cost(s)
        imp = True
        while imp:
            imp = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    t = s[:]
                    x = t.pop(i)
                    t.insert(j, x)
                    v = cost(t)
                    if v < cur:
                        s, cur, imp = t, v, True
                        continue
                    t = s[:]
                    t[i], t[j] = t[j], t[i]
                    v = cost(t)
                    if v < cur:
                        s, cur, imp = t, v, True
        return s, cur

    best, bc = None, None
    for s in starts:
        s, v = ls(s)
        if bc is None or v < bc:
            best, bc = s, v

    rng = np.random.default_rng(0)
    for _ in range(8):
        s = best[:]
        for _ in range(3):
            i, j = rng.integers(n, size=2)
            s[i], s[j] = s[j], s[i]
        s, v = ls(s)
        if v < bc:
            best, bc = s, v

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
