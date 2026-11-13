# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 300
    k = 60
    n = 40
    d = 7

    import random, time

    def ok(S, x):
        if x in S:
            return False
        for y in S:
            z = 2 * y - x
            if z in S:
                return False
            z = 2 * x - y
            if z in S:
                return False
            if not (x + y) & 1 and ((x + y) >> 1) in S:
                return False
        return True

    def fill(S, order):
        for x in order:
            if ok(S, x):
                S.add(x)
        return S

    def swap12(S, rng):
        while len(S) < k:
            done = False
            el = list(S)
            rng.shuffle(el)
            for r in el:
                S.remove(r)
                adds = [x for x in range(m) if x != r and ok(S, x)]
                if len(adds) >= 2:
                    for i, a1 in enumerate(adds):
                        S.add(a1)
                        b = next((b for b in adds[i + 1:] if ok(S, b)), None)
                        if b is not None:
                            S.add(b)
                            done = True
                            break
                        S.remove(a1)
                if done:
                    break
                S.add(r)
                if len(S) >= k:
                    return S
            if not done:
                break
        return S

    def tern(v):
        while v:
            if v % 3 == 2:
                return False
            v //= 3
        return True

    rng = random.Random(2024)
    base = [v for v in range(m) if tern(v)]
    starts = []
    for sh in range(0, 40):
        starts.append({v + sh for v in base if v + sh < m})
        starts.append({m - 1 - v - sh for v in base if m - 1 - v - sh >= 0})

    best = set()
    t0 = time.time()
    it = 0
    TL = 4.5
    while time.time() - t0 < TL and len(best) < k:
        S = set(starts[it]) if it < len(starts) else set()
        it += 1
        order = list(range(m))
        rng.shuffle(order)
        S = swap12(fill(S, order), rng)
        loc = set(S)
        for _ in range(60):
            if time.time() - t0 > TL:
                break
            T = set(S)
            kk = rng.choice((2, 3, 3, 3, 5))
            for r in rng.sample(sorted(T), min(kk, len(T))):
                T.remove(r)
            rng.shuffle(order)
            T = swap12(fill(T, order), rng)
            if len(T) >= len(S) or (len(T) == len(S) - 1 and rng.random() < 0.08):
                S = T
            if len(S) > len(loc):
                loc = set(S)
        if len(loc) > len(best):
            best = set(loc)

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
