# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))

    def length(t):
        return sum(D[t[i - 1], t[i]] for i in range(n))

    def improve(t):
        imp = True
        while imp:
            imp = False
            # 2-opt
            for i in range(n - 1):
                for j in range(i + 2, n if i > 0 else n - 1):
                    a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        imp = True
            # Or-opt: move segments of length 1..3
            for l in (1, 2, 3):
                i = 0
                while i < n:
                    if i + l >= n:
                        break
                    seg = t[i:i + l]
                    pr, nx = t[i - 1], t[(i + l) % n]
                    rem = D[pr, seg[0]] + D[seg[-1], nx] - D[pr, nx]
                    r = t[:i] + t[i + l:]
                    best, bk, brev = 1e-9, -1, False
                    for k in range(len(r)):
                        a, b = r[k], r[(k + 1) % len(r)]
                        if a == pr and b == nx:
                            continue
                        g1 = rem - (D[a, seg[0]] + D[seg[-1], b] - D[a, b])
                        g2 = rem - (D[a, seg[-1]] + D[seg[0], b] - D[a, b])
                        if g1 > best:
                            best, bk, brev = g1, k, False
                        if g2 > best:
                            best, bk, brev = g2, k, True
                    if bk >= 0:
                        s = seg[::-1] if brev else seg
                        t[:] = r[:bk + 1] + s + r[bk + 1:]
                        imp = True
                    i += 1
        return t

    best_t, best_L = None, float("inf")
    for s in range(0, n, max(1, n // 12)):
        t = [s]
        left = set(range(n)) - {s}
        while left:
            j = min(left, key=lambda k: D[t[-1], k])
            t.append(j)
            left.remove(j)
        t = improve(t)
        L = length(t)
        if L < best_L:
            best_t, best_L = t[:], L
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
