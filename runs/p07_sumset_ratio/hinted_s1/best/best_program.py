# EVOLVE-BLOCK-START
# Goal: choose a set A of distinct integers. With A+A the set of all sums a+b
# and A-A the set of all differences a-b (a, b in A, a = b allowed), make the
# sumset large relative to the difference set while keeping A small.
#
# Score: the evaluator computes
#     |A+A|**2 / (|A-A| * |A|**1.5)
# and divides it by a reference value of about 2.5, so 1.0 means matching the
# reference (higher is possible). It is NOT the plain ratio |A+A| / |A-A| and
# NOT its logarithmic form: the sumset size is squared and the score is divided
# by the number of elements to the power 1.5, so a set with a good plain ratio
# but many elements scores poorly.
#
# Validity: run() must return a list of between 1 and 60 integers, all
# distinct, each in the range 0..299 (there is no tolerance; values are exact
# integers). Any violation, an exception, or a run longer than 30 seconds
# scores 0.
import numpy as np


def _score(A):
    S = {a + b for a in A for b in A}
    D = {a - b for a in A for b in A}
    return len(S) ** 2 / (len(D) * len(A) ** 1.5)


def _greedy_sidon(limit=300):
    # Mian-Chowla style greedy Sidon set (all pairwise sums distinct).
    A, sums = [], set()
    for x in range(limit):
        new = {x + a for a in A} | {2 * x}
        if not (new & sums) and len(new) == len(A) + 1:
            A.append(x)
            sums |= new
    return A


def _basis(N, k):
    # Symmetric set (k | N): A+A = N + (A-A), and A-A covers [-N, N].
    A = set(range(0, N + 1, k)) | set(range(k)) | set(range(N - k + 1, N + 1))
    return sorted(A)


def _fast_score(x):
    n = x.sum()
    if n == 0:
        return 0.0
    s = np.count_nonzero(np.convolve(x, x) > 0.5)
    d = np.count_nonzero(np.correlate(x, x, mode="full") > 0.5)
    return s * s / (d * n ** 1.5)


def _search(start, sym, budget, seed, limit=300):
    import time
    rng = np.random.default_rng(seed)
    x = np.zeros(limit)
    x[list(start)] = 1.0
    N = max(start)
    cur = _fast_score(x)
    best, bx = cur, x.copy()
    t0 = time.time()
    it = 0
    while time.time() - t0 < budget:
        it += 1
        y = x.copy()
        if sym:
            for _ in range(1 + (rng.random() < 0.3)):
                i = int(rng.integers(0, N // 2 + 1))
                v = 1.0 - y[i]
                y[i] = v
                y[N - i] = v
        else:
            ones = np.flatnonzero(y > 0.5)
            zeros = np.flatnonzero(y < 0.5)
            r = rng.random()
            if r < 0.4 and len(ones) > 2:
                y[rng.choice(ones)] = 0.0
            elif r < 0.7 and len(ones) < 60:
                y[rng.choice(zeros)] = 1.0
            else:
                y[rng.choice(ones)] = 0.0
                y[rng.choice(zeros)] = 1.0
        n = y.sum()
        if not (1 <= n <= 60):
            continue
        s = _fast_score(y)
        if s >= cur or rng.random() < 0.002:
            x, cur = y, s
            if s > best:
                best, bx = s, y.copy()
        if it % 3000 == 0:
            x, cur = bx.copy(), best
    return [int(v) for v in np.flatnonzero(bx > 0.5)]


def _valid(c):
    return 1 <= len(set(c)) <= 60 and all(0 <= v < 300 for v in c)


def f():
    upper_limit = 300
    ruler = [0, 1, 8, 11, 68, 77, 94, 116, 121, 156, 158, 179,
             194, 208, 212, 228, 240, 253, 259, 283]
    candidates = [_greedy_sidon(upper_limit), list(range(2)), ruler]
    for N in range(200, 300):
        for k in range(5, 30):
            if N % k == 0:
                candidates.append(_basis(N, k))
    candidates = [c for c in candidates if _valid(c)]
    best = max(candidates, key=_score)
    for sym, budget, seed in ((True, 9.0, 1), (False, 7.0, 2)):
        try:
            ls = _search(best, sym, budget, seed)
            if _valid(ls) and _score(ls) > _score(best):
                best = ls
        except Exception:
            pass
    return np.array(sorted(set(best))[:60])


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
