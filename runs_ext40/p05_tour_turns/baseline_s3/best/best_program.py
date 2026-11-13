# EVOLVE-BLOCK-START
import numpy as np


def _len(t, D):
    return sum(D[t[i - 1], t[i]] for i in range(len(t)))


def _improve(t, D):
    n = len(t)
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
        for L in (1, 2, 3):
            i = 0
            while i < n:
                seg = [t[(i + k) % n] for k in range(L)]
                if i + L > n:
                    i += 1
                    continue
                p_, nx = t[i - 1], t[(i + L) % n]
                gain = D[p_, seg[0]] + D[seg[-1], nx] - D[p_, nx]
                rest = t[:i] + t[i + L:]
                best, bj, rev = 1e-9, -1, False
                m = len(rest)
                for j in range(m):
                    u, v = rest[j], rest[(j + 1) % m]
                    c1 = D[u, seg[0]] + D[seg[-1], v] - D[u, v]
                    c2 = D[u, seg[-1]] + D[seg[0], v] - D[u, v]
                    if gain - c1 > best:
                        best, bj, rev = gain - c1, j, False
                    if gain - c2 > best:
                        best, bj, rev = gain - c2, j, True
                if bj >= 0:
                    s = seg[::-1] if rev else seg
                    t[:] = rest[:bj + 1] + s + rest[bj + 1:]
                    imp = True
                i += 1
    return t


def f(p):
    n = len(p)
    D = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    best, bl = None, float("inf")
    for s in range(n):
        t, used = [s], {s}
        while len(t) < n:
            r = D[t[-1]].copy()
            r[list(used)] = np.inf
            k = int(np.argmin(r))
            t.append(k)
            used.add(k)
        t = _improve(t, D)
        L = _len(t, D)
        if L < bl:
            bl, best = L, t[:]
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
