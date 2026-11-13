# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 300
    k = 60
    n = 40
    d = 7

    rng = np.random.default_rng(12345)

    # Ternary digits in {0,1}: a classic 3-AP-free (Salem–Spencer) set.
    base = [x for x in range(m) if set(np.base_repr(x, 3)) <= set("01")]

    def ok(S, x):
        for y in S:
            if (2 * y - x) in S or (2 * x - y) in S:
                return False
            if (x + y) % 2 == 0 and (x + y) // 2 in S:
                return False
        return True

    best = sorted(base)
    for _ in range(300):
        S = set(best)
        cnt = int(rng.integers(1, 6))
        for r in rng.choice(best, size=min(cnt, len(best)), replace=False):
            S.discard(int(r))
        for x in rng.permutation(m):
            x = int(x)
            if x not in S and ok(S, x):
                S.add(x)
        if len(S) >= len(best):
            best = sorted(S)

    # Final safety check: no 3-term arithmetic progressions.
    S = set(best)
    for i in best:
        for j in best:
            if j > i and (2 * j - i) in S:
                best = sorted(base)
                break

    a = np.unique(np.array(best, dtype=int))
    a = a[(a >= 0) & (a < m)]
    a = a[:k]

    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
