# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    L = lambda t: sum(D[t[i - 1], t[i]] for i in range(n))

    def opt(t):  # 2-opt until no improving move
        imp = True
        while imp:
            imp = False
            for i in range(n - 1):
                for j in range(i + 2, n - (i == 0)):
                    a, b, x, y = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    if D[a, x] + D[b, y] < D[a, b] + D[x, y] - 1e-9:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        imp = True
        return t

    nn = [0]  # nearest-neighbour start
    while len(nn) < n:
        nn.append(min((k for k in range(n) if k not in nn), key=lambda k: D[nn[-1], k]))
    ang = [int(i) for i in np.argsort(np.arctan2(*(p - p.mean(0)).T[::-1]))]
    best = min((opt(s) for s in (ang, nn)), key=L)
    bl, rng = L(best), np.random.default_rng(0)
    for _ in range(400):  # ILS with segment-swap kicks
        a, b, e = sorted(rng.choice(range(1, n), 3, replace=False))
        t = opt(best[:a] + best[b:e] + best[a:b] + best[e:])
        l = L(t)
        if l < bl - 1e-9:
            best, bl = t, l
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
