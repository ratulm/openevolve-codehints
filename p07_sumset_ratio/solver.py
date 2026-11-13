"""Offline simulated-annealing solver for the sumset problem (not shown to the LLM)."""
import math
import numpy as np

M = 300      # elements must lie in [0, M)
KMIN = 5
KMAX = 60


def counts(a):
    a = np.asarray(a)
    s = np.count_nonzero(np.bincount((a[:, None] + a[None, :]).ravel(), minlength=2 * M))
    d = np.count_nonzero(np.bincount((a[:, None] - a[None, :]).ravel() + M, minlength=2 * M))
    return s, d


def anneal(obj, seed, steps=60000, t0=0.05, t1=0.0005, start=None, mirror=None):
    """Maximise obj(S, D, n) over subsets of [0, M) with KMIN <= n <= KMAX. Works on log(obj).

    With mirror=c the search is restricted to sets symmetric about c/2: the state is
    the lower half B and the set is B | (c - B).
    """
    rng = np.random.default_rng(seed)
    M = globals()["M"] if mirror is None else mirror // 2 + 1
    full = (lambda st: st) if mirror is None else (lambda st: st | {mirror - x for x in st})
    KMIN, KMAX = (globals()["KMIN"], globals()["KMAX"]) if mirror is None else (3, 30)
    if start is None:
        n = int(rng.integers(KMIN, (KMAX if mirror is None else 14) + 1))
        cur = set(rng.choice(M, n, replace=False).tolist())
    else:
        cur = set(start)

    def val(st):
        a = sorted(full(st))
        return math.log(obj(*counts(a), len(a)))

    fc = val(cur)
    best, fb = set(cur), fc
    for i in range(steps):
        t = t0 * (t1 / t0) ** (i / steps)
        new = set(cur)
        r = rng.random()
        if r < 0.25 and len(new) < KMAX:
            new.add(int(rng.integers(M)))
        elif r < 0.5 and len(new) > KMIN:
            new.discard(list(new)[int(rng.integers(len(new)))])
        else:
            x = list(new)[int(rng.integers(len(new)))]
            new.discard(x)
            y = int(rng.integers(M)) if rng.random() < 0.5 else x + int(rng.integers(-3, 4))
            new.add(min(max(y, 0), M - 1))
        if len(new) < KMIN or len(new) > KMAX or new == cur:
            continue
        fn = val(new)
        if fn >= fc or rng.random() < math.exp((fn - fc) / t):
            cur, fc = new, fn
            if fc > fb:
                best, fb = set(cur), fc
    return sorted(full(best)), math.exp(fb)
