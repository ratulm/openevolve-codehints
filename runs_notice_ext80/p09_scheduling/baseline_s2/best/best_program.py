# EVOLVE-BLOCK-START
import numpy as np


def f(a, b, c):
    p = np.asarray(a); d = np.asarray(b); w = np.asarray(c)
    n = len(p)
    P = [int(x) for x in p]; D = [int(x) for x in d]; W = [int(x) for x in w]

    def cost(s):
        t = 0; v = 0
        for j in s:
            t += P[j]
            if t > D[j]:
                v += W[j] * (t - D[j])
        return v

    def ls(s):
        s = [int(x) for x in s]; best = cost(s); imp = True
        while imp:
            imp = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    q = s[:]; q.insert(j, q.pop(i))
                    v = cost(q)
                    if v < best:
                        s, best, imp = q, v, True
                    if j > i:
                        r = s[:]; r[i], r[j] = r[j], r[i]
                        v = cost(r)
                        if v < best:
                            s, best, imp = r, v, True
        return s, best

    starts = [list(np.argsort(d, kind='stable')), list(np.argsort(p / w, kind='stable'))]
    pm = p.mean()
    for k in (0.5, 1, 2):
        u = np.ones(n, bool); t = 0.0; s = []
        for _ in range(n):
            idx = np.where(u)[0]
            pr = w[idx] / p[idx] * np.exp(-np.maximum(0, d[idx] - p[idx] - t) / (k * pm))
            j = int(idx[np.argmax(pr)]); s.append(j); u[j] = False; t += p[j]
        starts.append(s)

    bs, bv = None, np.inf
    for s in starts:
        s, v = ls(s)
        if v < bv:
            bs, bv = s, v

    rng = np.random.default_rng(0)
    cs, cv = bs[:], bv
    for it in range(30):
        s = cs[:]
        L = int(rng.integers(2, 6)); i = int(rng.integers(0, n - L + 1))
        if it % 2:
            s[i:i + L] = s[i:i + L][::-1]
        else:
            blk = s[i:i + L]; del s[i:i + L]
            j = int(rng.integers(0, len(s) + 1)); s[j:j] = blk
        s, v = ls(s)
        if v <= cv:
            cs, cv = s, v
        if v < bv:
            bs, bv = s[:], v

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
