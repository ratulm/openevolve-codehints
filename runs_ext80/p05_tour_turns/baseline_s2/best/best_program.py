# EVOLVE-BLOCK-START
import numpy as np


def f(p):
    n = len(p)
    D = np.linalg.norm(p[:, None] - p[None], axis=2)
    L = lambda t: sum(D[t[k - 1], t[k]] for k in range(n))

    def ls(o):
        l, imp = L(o), True
        while imp:
            imp = False
            for i in range(1, n):
                for j in range(i + 1, n + 1):
                    for q in (o[:i] + o[i:j][::-1] + o[j:],
                              o[:i] + o[i + 1:j] + [o[i]] + o[j:],
                              o[:i] + [o[j - 1]] + o[i:j - 1] + o[j:]):
                        m = L(q)
                        if m < l - 1e-9:
                            o, l, imp = q, m, True
        return o

    c = p.mean(0)
    b = ls([int(i) for i in np.argsort(np.arctan2(*(p - c).T[::-1]))])
    r = np.random.default_rng(0)
    for _ in range(150):
        x, y, z = sorted(r.choice(range(1, n), 3, replace=False))
        for q in (b[:x] + b[y:z] + b[x:y] + b[z:], b[:x] + b[z:] + b[y:z] + b[x:y]):
            q = ls(q)
            if L(q) < L(b) - 1e-9:
                b = q
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
