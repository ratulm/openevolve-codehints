# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 300
    k = 60
    n = 40
    d = 7

    import time

    rng = np.random.default_rng(2024)

    def triples(sset, x):
        # all 3-APs that adding x to sset would create (as the other two elements)
        out = []
        for y in sset:
            if y == x:
                continue
            z = 2 * y - x          # x, y, z  (y middle)
            if z != x and z != y and z in sset and y < z if False else (z in sset and z != y and z != x and x < z):
                out.append((y, z))
            z = 2 * x - y          # y, x, z  (x middle)
            if z in sset and z != x and z != y and y < z:
                out.append((y, z))
            if (x + y) % 2 == 0:   # x, mid, y
                md = (x + y) // 2
                if md in sset and md != x and md != y and x < y:
                    out.append((y, md))
        return out

    def ok(sset, x):
        for y in sset:
            z = 2 * y - x
            if z in sset and z != y:
                return False
            z = 2 * x - y
            if z in sset and z != x:
                return False
            if (x + y) % 2 == 0:
                md = (x + y) // 2
                if md in sset and md != x and md != y:
                    return False
        return True

    def fill(S, order):
        S = set(S)
        for x in order:
            if x not in S and ok(S, x):
                S.add(x)
        return S

    def no2(v):
        while v:
            if v % 3 == 2:
                return False
            v //= 3
        return True

    base = [v for v in range(m) if no2(v)]
    cur = fill(set(), base + list(range(m)))
    best = set(cur)
    t0 = time.time()
    while time.time() - t0 < 8.0 and len(best) < k:
        x = int(rng.integers(m))
        if x in cur:
            continue
        tr = triples(cur, x)
        if len(tr) == 0:
            cur.add(x)
        elif len(tr) == 1:
            y, z = tr[0]
            cur.discard(y if rng.random() < 0.5 else z)
            cur.add(x)
            cur = fill(cur, [int(v) for v in rng.permutation(m)])
        else:
            continue
        if len(cur) > len(best):
            best = set(cur)

    a = np.array(sorted(best), dtype=int)
    a = np.unique(a)
    a = a[(a >= 0) & (a < m)]
    a = a[:k]

    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
