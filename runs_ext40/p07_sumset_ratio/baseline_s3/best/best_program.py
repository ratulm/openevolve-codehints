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
            if 2 * y - x in S or 2 * x - y in S:
                return False
            if (x + y) % 2 == 0 and (x + y) // 2 in S:
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
        starts.append({m - 1 - v - sh for v in base if 0 <= m - 1 - v - sh})
    best = set()
    t0 = time.time()
    it = 0
    while time.time() - t0 < 4.0 and len(best) < k:
        S = set(starts[it]) if it < len(starts) else set()
        it += 1
        order = list(range(m))
        rng.shuffle(order)
        S = swap12(fill(S, order), rng)
        # perturbation: kick out a few elements, refill, keep if not worse
        for _ in range(30):
            if time.time() - t0 > 4.0:
                break
            T = set(S)
            for r in rng.sample(sorted(T), min(3, len(T))):
                T.remove(r)
            rng.shuffle(order)
            T = swap12(fill(T, order), rng)
            if len(T) >= len(S):
                S = T
        if len(S) > len(best):
            best = set(S)

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
