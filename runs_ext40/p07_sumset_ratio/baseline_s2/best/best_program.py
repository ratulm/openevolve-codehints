# EVOLVE-BLOCK-START
import numpy as np


def _ap_free_greedy(order, m, k, start=()):
    """Greedily add elements from `order`, keeping the set free of 3-term APs."""
    chosen = [int(v) for v in start]
    inset = np.zeros(m, dtype=bool)
    for v in chosen:
        inset[v] = True
    for x in order:
        x = int(x)
        if len(chosen) >= k:
            break
        if inset[x]:
            continue
        ok = True
        for y in chosen:
            z = 2 * y - x          # x, y, z  (y is the midpoint)
            if 0 <= z < m and inset[z]:
                ok = False
                break
            z = 2 * x - y          # y, x, z  (x is the midpoint)
            if 0 <= z < m and inset[z]:
                ok = False
                break
            if (x + y) % 2 == 0 and inset[(x + y) // 2]:  # x, mid, y
                ok = False
                break
        if ok:
            chosen.append(x)
            inset[x] = True
    return chosen


def f():
    m = 300
    k = 60

    # Natural-order greedy = Stanley sequence (ternary digits 0/1), AP-free.
    best = _ap_free_greedy(range(m), m, k)

    # Randomized restarts plus perturbation of the current best.
    rng = np.random.default_rng(12345)
    for _ in range(300):
        cand = _ap_free_greedy(rng.permutation(m), m, k)
        if len(cand) > len(best):
            best = cand
        if len(best) > 4:
            keep = rng.choice(best, size=len(best) - 3, replace=False)
            cand = _ap_free_greedy(rng.permutation(m), m, k, start=keep)
            if len(cand) >= len(best):
                best = cand
        if len(best) >= k:
            break

    a = np.array(sorted(set(int(v) for v in best)), dtype=int)
    a = a[(a >= 0) & (a < m)]
    a = a[:k]
    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
