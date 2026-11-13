# EVOLVE-BLOCK-START
import numpy as np


def _ok(S, x):
    for a in S:
        if (2 * a - x) in S or (2 * x - a) in S:
            return False
        if (a + x) % 2 == 0 and (a + x) // 2 in S:
            return False
    return True


def _valid(lst):
    s = set(lst)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if (lst[i] + lst[j]) % 2 == 0 and (lst[i] + lst[j]) // 2 in s:
                return False
    return True


def f():
    import random
    m = 300
    k = 60

    # Baseline: numbers with only digits 0/1 in base 3 (provably 3-AP-free)
    def base3_ok(v):
        while v:
            if v % 3 == 2:
                return False
            v //= 3
        return True
    best = [v for v in range(m) if base3_ok(v)]

    rng = random.Random(12345)
    for t in range(150):
        if t == 0:
            S = set(best)
        else:
            order = list(range(m))
            rng.shuffle(order)
            S = set()
            for x in order:
                if _ok(S, x):
                    S.add(x)
        # local search: remove one element, try to add >= 2
        for _ in range(400):
            y = rng.choice(sorted(S))
            S.discard(y)
            cand = list(range(m))
            rng.shuffle(cand)
            added = []
            for x in cand:
                if x != y and x not in S and _ok(S, x):
                    S.add(x)
                    added.append(x)
            if len(added) == 0:
                S.add(y)
            elif len(added) == 1:
                # neutral move accepted (plateau walk) -- keeps size
                pass
        if len(S) > len(best) and _valid(sorted(S)):
            best = sorted(S)

    a = np.array(sorted(best)[:k], dtype=int)
    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
