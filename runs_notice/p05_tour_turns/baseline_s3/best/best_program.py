# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))

    def L(t):
        return D[t, np.roll(t, -1)].sum()

    def opt(t):
        imp = True
        while imp:
            imp = False
            # 2-opt
            for i in range(n - 1):
                for j in range(i + 2, n if i else n - 1):
                    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        imp = True
            if imp:
                continue
            # Or-opt: move segments of length 1..3 (optionally reversed)
            base = L(t)
            for s in (1, 2, 3):
                for i in range(n):
                    seg = [t[(i + k) % n] for k in range(s)]
                    r = [x for x in t if x not in seg]
                    for k in range(len(r)):
                        for sg in (seg, seg[::-1]):
                            u = r[:k + 1] + sg + r[k + 1:]
                            lu = L(u)
                            if lu < base - 1e-9:
                                t[:] = u
                                base = lu
                                imp = True
                    if imp:
                        break
                if imp:
                    break
        return t

    def nn(s):
        t, left = [s], set(range(n)) - {s}
        while left:
            x = min(left, key=lambda y: D[t[-1], y])
            t.append(x)
            left.remove(x)
        return t

    c = p.mean(axis=0)
    best, bl = None, 1e18
    for it in range(12):
        if it == 0:
            t = [int(i) for i in np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
        else:
            t = nn((it * 7) % n)
        t = opt(t)
        l = L(t)
        if l < bl:
            bl, best = l, t[:]
    return np.array(best)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
P = np.array([
    [38, 12], [46, 57], [10, 34], [66, 67], [86, 96], [28, 22],
    [30, 72], [26, 79], [6, 79], [59, 69], [60, 54], [62, 94],
    [93, 16], [76, 65], [71, 60], [91, 40], [59, 86], [17, 78],
    [27, 48], [23, 15], [67, 27], [64, 37], [45, 48], [6, 72],
    [45, 77], [33, 7], [4, 61], [42, 29], [26, 90], [86, 62],
    [65, 83], [49, 23], [38, 77], [81, 34], [96, 71], [7, 5],
    [82, 70], [34, 43], [13, 23], [34, 90], [77, 12], [97, 8],
    [80, 21], [71, 36], [85, 78], [60, 7],
], dtype=float)


def run():
    o = f(P)
    return [int(i) for i in o]
