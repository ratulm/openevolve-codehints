# EVOLVE-BLOCK-START
import numpy as np


def f(a, b, c):
    p = [int(x) for x in a]
    d = [int(x) for x in b]
    w = [int(x) for x in c]
    n = len(p)

    def cost(s):
        t = 0
        z = 0
        for j in s:
            t += p[j]
            if t > d[j]:
                z += w[j] * (t - d[j])
        return z

    def ins(r, x):
        m = len(r)
        T = [0] * (m + 1)
        Z = [0] * (m + 1)
        for idx, j in enumerate(r):
            T[idx + 1] = T[idx] + p[j]
            Z[idx + 1] = Z[idx] + w[j] * max(0, T[idx + 1] - d[j])
        px, wx, dx = p[x], w[x], d[x]
        bk, bv = -1, None
        for k in range(m + 1):
            t = T[k] + px
            v = Z[k] + wx * max(0, t - dx)
            if bv is not None and v >= bv:
                continue
            for j in r[k:]:
                t += p[j]
                if t > d[j]:
                    v += w[j] * (t - d[j])
                    if bv is not None and v >= bv:
                        break
            if bv is None or v < bv:
                bk, bv = k, v
        return bk, bv

    def ls(s):
        s = list(s)
        best = cost(s)
        imp = True
        while imp:
            imp = False
            for i in range(n):
                x = s[i]
                r = s[:i] + s[i + 1:]
                bk, bv = ins(r, x)
                if bv < best:
                    s, best, imp = r[:bk] + [x] + r[bk:], bv, True
        return s, best

    pm = sum(p) / n
    starts = [sorted(range(n), key=lambda j: d[j]),
              sorted(range(n), key=lambda j: p[j] / w[j])]
    for K in (1.0, 3.0):
        u = set(range(n))
        t = 0
        s = []
        while u:
            j = max(u, key=lambda j: w[j] / p[j] *
                    np.exp(-max(0, d[j] - p[j] - t) / (K * pm)))
            s.append(j)
            u.remove(j)
            t += p[j]
        starts.append(s)

    import random
    rng = random.Random(7)
    best, bz = min((ls(s) for s in starts), key=lambda t: t[1])
    cur, cz, stall = best[:], bz, 0
    last = 0
    for it in range(250):
        if it - last > 120:
            break
        q = cur[:]
        if rng.random() < 0.25:
            i = rng.randrange(n - 3)
            L = rng.randint(3, min(7, n - i))
            q[i:i + L] = q[i:i + L][::-1]
        else:
            for _ in range(rng.randint(2, 5)):
                i, k = rng.sample(range(n), 2)
                q.insert(k, q.pop(i))
        q, vq = ls(q)
        if vq <= cz:
            cur, cz = q, vq
        stall += 1
        if vq < bz:
            best, bz, stall, last = q[:], vq, 0, it
        if stall > 60:
            cur, cz, stall = best[:], bz, 0
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
