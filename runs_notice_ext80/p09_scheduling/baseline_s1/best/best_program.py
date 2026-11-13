# EVOLVE-BLOCK-START
import numpy as np


def f(a, b, c):
    p, d, w = (np.asarray(x, float) for x in (a, b, c))
    n = len(p)

    def cost(s):
        return float((w[s] * np.maximum(np.cumsum(p[s]) - d[s], 0)).sum())

    def ls(s):
        s = np.array(s)
        best = cost(s)
        imp = True
        while imp:
            imp = False
            t0 = 0.0
            for i in range(n - 1):
                x, y = s[i], s[i + 1]
                cx, cy = t0 + p[x], t0 + p[x] + p[y]
                old = w[x] * max(cx - d[x], 0) + w[y] * max(cy - d[y], 0)
                ny = t0 + p[y]
                new = w[y] * max(ny - d[y], 0) + w[x] * max(cy - d[x], 0)
                if new < old - 1e-9:
                    s[i], s[i + 1] = y, x
                    imp = True
                    t0 = ny
                else:
                    t0 = cx
            for i in range(n):
                for j in range(max(0, i - 10), min(n, i + 11)):
                    if i == j:
                        continue
                    t = s.copy()
                    if j > i:
                        t[i:j + 1] = np.roll(t[i:j + 1], -1)
                    else:
                        t[j:i + 1] = np.roll(t[j:i + 1], 1)
                    v = cost(t)
                    if v < best - 1e-9:
                        s, best, imp = t, v, True
            best = cost(s)
        return list(s), best

    starts = [list(np.argsort(d, kind="stable"))]
    pm = p.mean()
    for K in (0.5, 1, 2, 3, 4, 6):
        s, t, u = [], 0.0, np.zeros(n, bool)
        for _ in range(n):
            r = np.where(u, -1.0,
                         w / p * np.exp(-np.maximum(d - p - t, 0) / (K * pm)))
            k = int(np.argmax(r))
            s.append(k)
            u[k] = True
            t += p[k]
        starts.append(s)

    rng = np.random.default_rng(0)
    bs, bv = None, np.inf
    for s in starts:
        s, v = ls(s)
        if v < bv:
            bs, bv = s, v
    for _ in range(40):
        t = bs[:]
        i, j = rng.choice(n, 2, replace=False)
        t.insert(j, t.pop(i))
        t, v = ls(t)
        if v < bv:
            bs, bv = t, v

    return np.array(bs)


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
