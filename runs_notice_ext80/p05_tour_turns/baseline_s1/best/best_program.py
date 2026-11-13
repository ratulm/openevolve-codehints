# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.hypot(*(p[:, None] - p[None]).T)
    t = list(np.argsort(np.arctan2(*(p - p.mean(0)).T[::-1])))
    D = D.tolist()
    t = [int(x) for x in t]
    L = lambda t: sum(D[t[i - 1]][t[i]] for i in range(n))

    def ls(t):
        b, g = L(t), 1
        while g:
            g = 0
            for i in range(n):
                for j in range(i + 2, n + 1):
                    for s in range(4):
                        r = t[:i] + t[i + s:]
                        u = r[:j - s] + t[i:i + s][::-1] + r[j - s:] if s else t[:i] + t[i:j][::-1] + t[j:]
                        l = L(u)
                        if l < b - 1e-9:
                            t, b, g = u, l, 1
        return t, b

    t, b = ls(t)
    R = np.random.RandomState(0)
    for _ in range(30):
        i, j, k = sorted(R.choice(range(1, n), 3, replace=False))
        u, l = ls(t[:i] + t[j:k] + t[i:j] + t[k:])
        if l < b - 1e-9:
            t, b = u, l
    return np.array(t)


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
