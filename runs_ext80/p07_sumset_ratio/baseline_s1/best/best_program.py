# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 300
    k = 60
    n = 40
    d = 7

    # Greedy: add the element contributing the most new distinct
    # differences and sums (Sidon-like start, then dense filling).
    chosen = []
    diffs = set()
    sums = set()
    for _ in range(k):
        best, best_gain = None, -1
        for x in range(m):
            if x in chosen:
                continue
            nd = len({abs(x - y) for y in chosen} - diffs)
            ns = len({x + y for y in chosen + [x]} - sums)
            # favour new sums (doubling differences hurt), ties -> fewer diffs
            gain = (nd + 2 * ns, -nd)
            if best is None or gain > best_gain:
                best, best_gain = x, gain
        if best is None or best_gain[0] <= 0:
            break
        diffs |= {abs(best - y) for y in chosen}
        chosen.append(best)
        sums |= {best + y for y in chosen}

    a = np.unique(np.array(chosen, dtype=int))
    a = a[(a >= 0) & (a < m)]
    a = a[:k]

    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
