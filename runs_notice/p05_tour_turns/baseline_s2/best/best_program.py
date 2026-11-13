# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))

    def L(t):
        return sum(D[t[i - 1], t[i]] for i in range(n))

    def two_opt(t):
        imp = True
        while imp:
            imp = False
            for i in range(n - 1):
                a, b = t[i], t[i + 1]
                for j in range(i + 2, n if i > 0 else n - 1):
                    c, d = t[j], t[(j + 1) % n]
                    if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-10:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        a, b = t[i], t[i + 1]
                        imp = True
        return t

    def or_opt(t):
        imp = False
        for k in (1, 2, 3):
            i = 0
            while i < n:
                idx = [(i + q) % n for q in range(k)]
                seg = [t[q] for q in idx]
                a = t[(i - 1) % n]
                b = t[(i + k) % n]
                gain = D[a, seg[0]] + D[seg[-1], b] - D[a, b]
                s = set(idx)
                r = [t[q] for q in range(n) if q not in s]
                m = len(r)
                best = (1e-10, None, None)
                for j in range(m):
                    x, y = r[j - 1], r[j]
                    c1 = D[x, seg[0]] + D[seg[-1], y] - D[x, y]
                    c2 = D[x, seg[-1]] + D[seg[0], y] - D[x, y]
                    if gain - c1 > best[0]:
                        best = (gain - c1, j, False)
                    if gain - c2 > best[0]:
                        best = (gain - c2, j, True)
                if best[1] is not None:
                    j, rv = best[1], best[2]
                    sg = seg[::-1] if rv else seg
                    t[:] = r[:j] + sg + r[j:]
                    imp = True
                i += 1
        return imp

    best_t, best_l = None, float("inf")
    for s0 in range(n):
        t = [s0]
        u = set(range(n)) - {s0}
        while u:
            j = min(u, key=lambda k: D[t[-1], k])
            t.append(j)
            u.remove(j)
        while True:
            two_opt(t)
            if not or_opt(t):
                break
        l = L(t)
        if l < best_l:
            best_l, best_t = l, t[:]
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
