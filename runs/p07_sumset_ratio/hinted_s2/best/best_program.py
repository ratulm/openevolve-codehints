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
    # Bitset-based sumset/difference-set sizes: O(n) big-int ops instead of O(n^2).
    m = 0
    for a in A:
        m |= 1 << a
    S = 0
    D = 0
    for a in A:
        S |= m << a
        D |= m << (300 - a)
    s = bin(S).count('1')
    d = bin(D).count('1')
    return s * s / (d * len(A) ** 1.5)


def f():
    import random, time
    upper_limit = 300  # every element must be below this
    max_set_size = 60  # at most this many elements

    # Conway's MSTD set: |A+A|=26 > |A-A|=25 with only 8 elements.
    best = [0, 2, 3, 4, 7, 11, 12, 14]
    best_s = _score(best)

    import math
    rng = random.Random(12345)
    t0 = time.time()
    T_TOTAL = 20.0
    sizes = [26, 30, 22, 34, 28, 24, 32, 20, 36, 18]
    run_idx = 0
    while time.time() - t0 < T_TOTAL:
        n = sizes[run_idx % len(sizes)]
        run_idx += 1
        cur = rng.sample(range(upper_limit), n)
        cur_s = _score(cur)
        t1 = time.time()
        run_len = 2.0
        temp0 = 0.02
        while True:
            el = time.time() - t1
            if el > run_len or time.time() - t0 > T_TOTAL:
                break
            temp = temp0 * (1 - el / run_len) + 1e-5
            B = list(cur)
            r = rng.random()
            if r < 0.05 and len(B) > 6:
                B.pop(rng.randrange(len(B)))
            elif r < 0.10 and len(B) < 40:
                v = rng.randrange(upper_limit)
                if v in B:
                    continue
                B.append(v)
            else:
                i = rng.randrange(len(B))
                if rng.random() < 0.5:
                    v = rng.randrange(upper_limit)
                else:
                    v = B[i] + rng.choice((-3, -2, -1, 1, 2, 3))
                if v < 0 or v >= upper_limit or v in B:
                    continue
                B[i] = v
            s = _score(B)
            if s >= cur_s or rng.random() < math.exp((s - cur_s) / temp):
                cur, cur_s = B, s
                if s > best_s:
                    best, best_s = sorted(B), s

    candidate_set = np.array(best)

    # Enforce the validity rules: distinct values, inside the range, not too many.
    candidate_set = np.unique(candidate_set)
    candidate_set = candidate_set[(candidate_set >= 0) & (candidate_set < upper_limit)]
    candidate_set = candidate_set[:max_set_size]

    return candidate_set


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
