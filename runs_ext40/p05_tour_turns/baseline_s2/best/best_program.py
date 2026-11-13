# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.hypot(*(p[:, None] - p[None]).T).tolist()
    L = lambda t: sum(D[t[i - 1]][t[i]] for i in range(n))

    def opt(t):
        l = L(t)
        while 1:
            for i in range(1, n):
                for j in range(i + 1, n + 1):
                    for k in range(i, min(i + 3, j)):
                        for u in (t[:i] + t[i:j][::-1] + t[j:],
                                  t[:i] + t[k:j] + t[i:k] + t[j:],
                                  t[:i] + t[k:j] + t[i:k][::-1] + t[j:]):
                            v = L(u)
                            if v < l - 1e-9: t, l = u, v; break
                        else: continue
                        break
                    else: continue
                    break
                else: continue
                break
            else: return t

    c = p.mean(0)
    b = opt([int(x) for x in np.argsort(np.arctan2(*(p - c).T[::-1]))])
    r = np.random.default_rng(1)
    for _ in range(100):
        i, j, k = sorted(r.choice(range(1, n), 3, replace=False).tolist())
        t = opt(b[:i] + b[j:k] + b[i:j] + b[k:])
        if L(t) < L(b) - 1e-9: b = t
    return b


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
