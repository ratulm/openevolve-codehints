# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))

    def length(t):
        return sum(D[t[i - 1], t[i]] for i in range(n))

    def improve(t):
        imp = True
        while imp:
            imp = False
            # 2-opt
            for i in range(n - 1):
                a, b = t[i], t[i + 1]
                for j in range(i + 2, n if i else n - 1):
                    c, d = t[j], t[(j + 1) % n]
                    if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        b = t[i + 1]
                        imp = True
            # Or-opt: move segments of length 1..3
            for k in (1, 2, 3):
                i = 0
                while i < n:
                    if i + k > n:
                        break
                    seg = t[i:i + k]
                    prv, nxt = t[i - 1], t[(i + k) % n]
                    gain = D[prv, seg[0]] + D[seg[-1], nxt] - D[prv, nxt]
                    rest = t[:i] + t[i + k:]
                    m = len(rest)
                    best, bj, brev = 1e-9, -1, False
                    for j in range(m):
                        x, y = rest[j], rest[(j + 1) % m]
                        base = D[x, y]
                        c1 = D[x, seg[0]] + D[seg[-1], y] - base
                        c2 = D[x, seg[-1]] + D[seg[0], y] - base
                        if gain - c1 > best:
                            best, bj, brev = gain - c1, j, False
                        if gain - c2 > best:
                            best, bj, brev = gain - c2, j, True
                    if bj >= 0:
                        s = seg[::-1] if brev else seg
                        t[:] = rest[:bj + 1] + s + rest[bj + 1:]
                        imp = True
                    i += 1
        return t

    best_t, best_l = None, float("inf")
    for s in range(0, n, max(1, n // 8)):
        t = [s]
        left = set(range(n)) - {s}
        while left:
            j = min(left, key=lambda k: D[t[-1], k])
            t.append(j)
            left.remove(j)
        t = improve(t)
        L = length(t)
        if L < best_l:
            best_l, best_t = L, t[:]
    return best_t


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
