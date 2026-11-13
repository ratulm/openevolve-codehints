# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    L = lambda t: d[t, np.roll(t, -1)].sum()

    def opt(t):
        t, imp = list(t), 1
        while imp:
            imp = 0
            for i in range(n - 1):
                for j in range(i + 2, n if i else n - 1):
                    a, b, c, e = t[i], t[i + 1], t[j], t[(j + 1) % n]
                    if d[a, c] + d[b, e] < d[a, b] + d[c, e] - 1e-9:
                        t[i + 1:j + 1], imp = t[i + 1:j + 1][::-1], 1
            for s in (1, 2, 3):
                for i in range(n - s + 1):
                    sg, a, b = t[i:i + s], t[i - 1], t[(i + s) % n]
                    g = d[a, sg[0]] + d[sg[-1], b] - d[a, b]
                    r = t[:i] + t[i + s:]
                    for k in range(len(r)):
                        x, y = r[k], r[(k + 1) % len(r)]
                        q = min((sg, sg[::-1]), key=lambda z: d[x, z[0]] + d[z[-1], y])
                        if d[x, q[0]] + d[q[-1], y] - d[x, y] < g - 1e-9:
                            t, imp = r[:k + 1] + q + r[k + 1:], 1
                            break
        return t

    o, u = [0], set(range(1, n))  # greedy nearest-neighbour start
    while u:
        k = min(u, key=lambda x: d[o[-1], x])
        o.append(k)
        u.remove(k)
    best = opt(o)
    bl = L(best)
    rng = np.random.default_rng(12345)
    stall = 0
    for _ in range(600):
        # localized double bridge: segments within a window of <=16
        w = int(rng.integers(6, min(16, n - 2)))
        s0 = int(rng.integers(1, n - w))
        i, j, k = sorted(rng.choice(np.arange(s0, s0 + w), 3, replace=False))
        t = opt(best[:i] + best[j:k] + best[i:j] + best[k:])
        l = L(t)
        if l < bl - 1e-9:
            best, bl, stall = t, l, 0
        else:
            stall += 1
            if stall > 200:  # converged: stop early
                break
    return np.array(best, dtype=int)


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
