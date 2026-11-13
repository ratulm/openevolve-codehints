# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 300
    k = 60
    n = 40
    d = 7

    # Build a large 3-AP-free subset of [0, m): start from the base-3
    # "digits in {0,1}" (Salem-Spencer / Stanley) set, then try to grow it
    # with randomized greedy restarts, keeping the best valid set found.
    def ok_add(S, sset, x):
        for y in S:
            # x as endpoint: 2y - x, or x as middle: (x+y)/2 ...
            z = 2 * y - x
            if z in sset and z != y:
                return False
            z = 2 * x - y
            if z in sset and z != x:
                return False
            if (x + y) % 2 == 0:
                mid = (x + y) // 2
                if mid in sset and mid != x and mid != y:
                    return False
        return True

    def greedy(order):
        S, sset = [], set()
        for x in order:
            if ok_add(S, sset, x):
                S.append(x)
                sset.add(x)
        return S

    def no2(v):
        while v:
            if v % 3 == 2:
                return False
            v //= 3
        return True

    import time

    def conf(s, x):
        out = set()
        for y in s:
            if y == x:
                continue
            for z in (2 * y - x, 2 * x - y):
                if z in s and z != x and z != y:
                    out.add((min(y, z), max(y, z)))
            if (x + y) % 2 == 0:
                md = (x + y) // 2
                if md in s and md != x and md != y:
                    out.add((min(y, md), max(y, md)))
        return list(out)

    def fill(s, order):
        for x in order:
            x = int(x)
            if x not in s and not conf(s, x):
                s.add(x)
        return s

    base = [v for v in range(m) if no2(v)]
    rng = np.random.default_rng(2024)
    cur = fill(set(base), range(m))
    best = set(cur)
    t0 = time.time()
    while time.time() - t0 < 8.0 and len(best) < k:
        x = int(rng.integers(m))
        if x in cur:
            continue
        rem = set()
        for p in conf(cur, x):
            if p[0] in rem or p[1] in rem:
                continue
            rem.add(p[0] if rng.random() < 0.5 else p[1])
        if len(rem) > 2:
            continue
        cur -= rem
        cur.add(x)
        cur = fill(cur, rng.permutation(m))
        if len(cur) > len(best):
            best = set(cur)
        elif len(cur) < len(best) - 2:
            cur = set(best)

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
