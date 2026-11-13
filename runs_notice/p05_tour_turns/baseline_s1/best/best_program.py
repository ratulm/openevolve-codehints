# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)).tolist()

    def length(t):
        return sum(D[t[i - 1]][t[i]] for i in range(n))

    def improve(t):
        imp = True
        while imp:
            imp = False
            # 2-opt
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
                    if D[a][c] + D[b][d] < D[a][b] + D[c][d] - 1e-9:
                        t[i:j + 1] = t[i:j + 1][::-1]
                        imp = True
            # Or-opt (segments of 1..3)
            for k in (1, 2, 3):
                i = 1
                while i + k <= n:
                    seg = t[i:i + k]
                    a, b = t[i - 1], t[(i + k) % n]
                    s0, se = seg[0], seg[-1]
                    gain = D[a][s0] + D[se][b] - D[a][b]
                    r = t[:i] + t[i + k:]
                    m = len(r)
                    best, bj, brev = 1e-9, -1, False
                    for j in range(m):
                        c, d = r[j], r[(j + 1) % m]
                        base = D[c][d]
                        d1 = gain - (D[c][s0] + D[se][d] - base)
                        d2 = gain - (D[c][se] + D[s0][d] - base)
                        if d1 > best:
                            best, bj, brev = d1, j, False
                        if d2 > best:
                            best, bj, brev = d2, j, True
                    if bj >= 0:
                        ins = seg[::-1] if brev else seg
                        t[:] = r[:bj + 1] + ins + r[bj + 1:]
                        imp = True
                    i += 1
        return t

    best_t, best_l = None, float("inf")
    for s in range(n):
        t = [s]
        u = set(range(n)) - {s}
        while u:
            last = t[-1]
            j = min(u, key=lambda k: D[last][k])
            t.append(j)
            u.remove(j)
        t = improve(t)
        L = length(t)
        if L < best_l:
            best_l, best_t = L, t[:]
    return np.array(best_t)


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
