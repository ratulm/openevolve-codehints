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


import time
import random


def _score(a):
    a = np.asarray(a)
    s = np.zeros(600, dtype=bool)
    s[(a[:, None] + a).ravel()] = True
    d = np.zeros(600, dtype=bool)
    d[(a[:, None] - a).ravel() + 299] = True
    S = s.sum()
    return S * S / (d.sum() * len(a) ** 1.5)


def _anneal(rng, A, deadline, iters, N=300, M=60, sym=False):
    A = set(A)
    cur = _score(sorted(A))
    best, bs = sorted(A), cur
    T = 0.08
    for _ in range(iters):
        if time.time() > deadline:
            break
        B = set(A)
        m = rng.random()
        if sym:
            # operate on mirrored pairs {x, N-1-x}; keeps |A+A| == |A-A|
            def rem():
                x = rng.choice(tuple(B))
                B.discard(x)
                B.discard(N - 1 - x)
                return x

            def add(x):
                x = min(N - 1, max(0, x))
                B.add(x)
                B.add(N - 1 - x)

            if m < 0.3 and len(B) > 4:
                rem()
            elif m < 0.55 and len(B) < M - 1:
                add(rng.randrange(N))
            elif m < 0.8 and len(B) > 2:
                x = rem()
                add(x + rng.choice((-3, -2, -1, 1, 2, 3)))
            else:
                if len(B) > 2:
                    rem()
                add(rng.randrange(N))
            if len(B) > M:
                continue
        elif m < 0.3 and len(B) > 2:
            B.remove(rng.choice(tuple(B)))
        elif m < 0.55 and len(B) < M:
            B.add(rng.randrange(N))
        elif m < 0.8 and len(B) > 2:
            x = rng.choice(tuple(B))
            B.remove(x)
            B.add(min(N - 1, max(0, x + rng.choice((-2, -1, 1, 2)))))
        else:
            if len(B) > 2:
                B.remove(rng.choice(tuple(B)))
            B.add(rng.randrange(N))
        if len(B) < 2 or B == A:
            continue
        v = _score(sorted(B))
        if v >= cur or rng.random() < np.exp((v - cur) / T):
            A, cur = B, v
            if cur > bs:
                bs, best = cur, sorted(A)
        T = max(T * 0.99985, 1e-4)
    return best, bs


def f():
    upper_limit = 300  # every element must be below this
    max_set_size = 60  # at most this many elements

    rng = random.Random(12345)
    t0 = time.time()
    deadline = t0 + 15.0

    best, bs = list(range(0, 280, 7)), 0.0
    bs = _score(best)
    # symmetric phase: A = 299 - A, so score reduces to |A+A| / |A|^1.5
    while time.time() < deadline:
        k = rng.randint(10, 20)
        half = rng.sample(range(upper_limit // 2), k)
        start = set(half) | {upper_limit - 1 - x for x in half} | {0, upper_limit - 1}
        cand, v = _anneal(rng, start, deadline, 25000, sym=True)
        if v > bs:
            bs, best = v, cand
    # general polish from the best found (may break symmetry if it helps)
    deadline2 = t0 + 22.5
    while time.time() < deadline2:
        cand, v = _anneal(rng, best, deadline2, 15000)
        if v > bs:
            bs, best = v, cand

    candidate_set = np.unique(np.array(best, dtype=int))
    candidate_set = candidate_set[(candidate_set >= 0) & (candidate_set < upper_limit)]
    candidate_set = candidate_set[:max_set_size]
    if len(candidate_set) == 0:
        candidate_set = np.array([0])
    return candidate_set


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
