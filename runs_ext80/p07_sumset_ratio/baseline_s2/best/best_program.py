# EVOLVE-BLOCK-START
import numpy as np


def f():
    m, k = 300, 60

    rng = np.random.default_rng(1)

    def full(t):
        v = np.array(sorted(t))
        return (len(np.unique(v[:, None] - v[None, :])) == 2 * m - 1 and
                len(np.unique(v[:, None] + v[None, :])) == 2 * m - 1)

    # Symmetric set A = (m-1) - A: then A+A = (m-1) + (A-A), so covering
    # all differences automatically covers all sums. Greedily add mirrored
    # pairs {c, m-1-c} that add the most new differences (random ties).
    def greedy(noise):
        s, D = {0, m - 1}, {m - 1}
        while len(D) < m - 1:
            best, bg = None, -1
            for c in range(1, (m + 1) // 2):
                if c in s:
                    continue
                p = {c, m - 1 - c}
                g = len({abs(x - y) for x in p for y in s | p} - D - {0})
                g += noise * rng.random()
                if g > bg:
                    best, bg = c, g
            p = {best, m - 1 - best}
            D |= {abs(x - y) for x in p for y in s | p} - {0}
            s |= p
        return s

    # Prune elements whose removal keeps full coverage (to fixpoint).
    def prune(s, shuffle):
        a, changed = sorted(s), True
        while changed:
            changed = False
            order = list(a)
            if shuffle:
                rng.shuffle(order)
            for x in order:
                t = [y for y in a if y != x]
                if full(t):
                    a, changed = t, True
        return a

    a = prune(greedy(0), False)
    cands = []
    # Wichmann-like symmetric rulers: dense block plus arithmetic marks.
    for w in range(3, 40):
        for st in range(w, 120):
            B = set(range(w)) | set(range(w - 1, m, st)) | {m - 1}
            B |= {m - 1 - x for x in B}
            if len(B) < len(a) + 6 and full(B):
                cands.append(B)
    for _ in range(40):
        cands.append(greedy(0.99))
    for B in cands:
        for sh in (False, True):
            t = prune(B, sh)
            if len(t) < len(a):
                a = t

    a = np.unique(np.array(a, dtype=int))
    a = a[(a >= 0) & (a < m)]
    a = a[:k]

    return a


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
