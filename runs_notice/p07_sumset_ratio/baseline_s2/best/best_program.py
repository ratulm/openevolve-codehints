# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 300
    k = 60
    n = 40
    d = 7

    rng = np.random.default_rng(12345)

    def ok_add(x, S, Sset):
        # adding x must not create a 3-term AP with any two members
        for y in S:
            # x as an endpoint: midpoint (x+y)/2 or the third point 2y-x / 2x-y
            if (x + y) % 2 == 0 and (x + y) // 2 in Sset:
                return False
            if 2 * y - x in Sset and 2 * y - x != x and 2 * y - x != y:
                return False
            if 2 * x - y in Sset and 2 * x - y != x and 2 * x - y != y:
                return False
        return True

    def complete(S):
        Sset = set(S)
        S = list(S)
        cand = [x for x in rng.permutation(m) if x not in Sset]
        for x in cand:
            x = int(x)
            if len(S) >= k:
                break
            if ok_add(x, S, Sset):
                S.append(x)
                Sset.add(x)
        return S

    # Salem–Spencer seed: ternary digits in {0,1}
    seed = [x for x in range(m) if all(c in "01" for c in np.base_repr(x, 3))]
    best = complete(seed)

    cur = list(best)
    for it in range(400):
        if len(best) >= k:
            break
        cur_list = list(cur)
        r = int(rng.integers(1, 5))
        drop = set(int(v) for v in rng.choice(cur_list, size=min(r, len(cur_list)), replace=False))
        trial = complete([x for x in cur_list if x not in drop])
        if len(trial) >= len(cur):
            cur = trial
        if len(trial) > len(best):
            best = list(trial)

    # final verification
    S = sorted(set(best))
    Sset = set(S)
    good = []
    gset = set()
    for x in S:
        if ok_add(x, good, gset):
            good.append(x)
            gset.add(x)

    a = np.array(sorted(good), dtype=int)
    a = a[(a >= 0) & (a < m)]
    a = a[:k]

    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
