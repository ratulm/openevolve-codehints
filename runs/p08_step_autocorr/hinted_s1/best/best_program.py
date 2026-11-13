# EVOLVE-BLOCK-START
# Goal: choose the 50 non-negative heights a_0..a_49 of a step function (50
# equal-width steps on an interval) so that a position-weighted peak of its
# autoconvolution is as small as possible relative to its squared total.
#
# Score: the evaluator computes the discrete autoconvolution
#   c_k = sum_i a_i * a_(k-i)   for k = 0..98   (same as np.convolve(a, a)),
# the weights w_k = 1 + 3*k/98 (rising linearly from 1 at k = 0 to 4 at
# k = 98), and the value
#   V = max_k (w_k * c_k) / (sum_i a_i)**2.
# The score is 0.0287461148884 / V, so lower V is better, 1.0 means matching
# the reference and higher is possible. The weight is asymmetric: the same
# autoconvolution value costs four times more at the largest shift than at
# shift 0. It is NOT the plain unweighted maximum of the autoconvolution, so
# reversing the vector changes the score. V does not change when all heights
# are multiplied by the same positive number.
#
# Validity: run() must return a one-dimensional vector of exactly 50 finite
# numbers, every one >= 0 (no tolerance: any negative value is invalid), and
# not all zero. Any violation, an exception, or a run longer than 30 seconds
# scores 0.
import numpy as np


import time

N = 50
M = 2 * N - 1
W = 1.0 + 3.0 * np.arange(M) / (M - 1)
# Index pattern for the Jacobian of the autoconvolution: J[i+j, i] = 2 a_j.
_JR = (np.arange(N)[None, :] + np.arange(N)[:, None])  # rows i+j, [j, i]
_JC = np.tile(np.arange(N), (N, 1))                    # cols i


def value(a):
    s = np.sum(a)
    if not np.all(np.isfinite(a)) or s <= 0:
        return np.inf
    return np.max(W * np.convolve(a, a)) / s ** 2


def smooth_descent(a0, iters, deadline):
    # a = x^2 keeps heights non-negative; minimise log-sum-exp smoothed max.
    x = np.sqrt(np.maximum(a0, 1e-12))
    x /= np.linalg.norm(x)
    m = np.zeros(N); v = np.zeros(N)
    lr, b1, b2 = 0.01, 0.9, 0.999
    best, bv = a0.copy(), value(a0)
    for t in range(1, iters + 1):
        if t % 100 == 0 and time.time() > deadline:
            break
        a = x * x
        s = a.sum()
        r = W * np.convolve(a, a) / s ** 2
        val = r.max()
        if val < bv:
            bv, best = val, a.copy()
        beta = 50.0 + 4000.0 * t / iters
        z = beta * (r / val - 1.0)
        p = np.exp(z - z.max()); p /= p.sum()
        F = np.dot(p, r)
        ga = 2.0 * np.correlate(p * W, a, 'valid') / s ** 2 - 2.0 * F / s
        g = ga * 2.0 * x / val
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-12)
        x /= np.linalg.norm(x)
    return best, bv


def lp_polish(a, deadline):
    # Sequential linear programming on the exact max objective.
    try:
        from scipy.optimize import linprog
    except Exception:
        return a
    a = a / a.sum()
    cur = value(a)
    step = 0.2 * a.max()
    cost = np.r_[np.zeros(N), 1.0]
    aeq = np.r_[np.ones(N), 0.0][None, :]
    for _ in range(400):
        if time.time() > deadline or step < 1e-10:
            break
        c = np.convolve(a, a)
        A = np.zeros((M, N))
        A[_JR, _JC] = 2.0 * a[:, None]
        A = W[:, None] * A
        aub = np.hstack([A, -np.ones((M, 1))])
        bub = -(W * c)
        lo = np.maximum(-a, -step)
        bounds = list(zip(lo, np.full(N, step))) + [(None, None)]
        try:
            res = linprog(cost, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=[0.0],
                          bounds=bounds, method='highs')
        except Exception:
            break
        if res.status != 0 or res.x is None:
            break
        na = np.maximum(a + res.x[:N], 0.0)
        if na.sum() <= 0:
            step *= 0.5
            continue
        na /= na.sum()
        nv = value(na)
        if nv < cur:
            a, cur = na, nv
            step *= 1.5
        else:
            step *= 0.5
    return a


def f():
    t0 = time.time()
    idx = np.arange(N, dtype=float)
    rng = np.random.default_rng(12345)
    starts = [np.exp(-idx / 10.0), np.exp(-idx / 25.0), np.ones(N),
              (N - idx), 1.0 + idx]
    starts += [np.minimum(idx + 1, N - idx), 1.0 / np.sqrt(idx + 1.0),
               np.exp(-idx / 40.0), np.linspace(2.0, 1.0, N)]
    starts += [rng.random(N) + 0.1 for _ in range(30)]
    best = make_valid_and_normalise(1.0 + idx)
    bv = value(best)
    try:
        for s in starts:
            if time.time() - t0 > 10.0:
                break
            a, v = smooth_descent(make_valid_and_normalise(s), 1500, t0 + 10.0)
            if v < bv:
                bv, best = v, make_valid_and_normalise(a)
    except Exception:
        pass
    try:
        cand = make_valid_and_normalise(lp_polish(best.copy(), t0 + 13.0))
        if value(cand) < bv:
            best, bv = cand, value(cand)
    except Exception:
        pass
    # Basin hopping: perturb the incumbent and re-polish with SLP.
    try:
        fails = 0
        while time.time() - t0 < 24.0:
            # Small shakes normally; larger ones after repeated failures.
            if fails > 8 and rng.random() < 0.5:
                scale = rng.uniform(0.3, 0.6)
            else:
                scale = rng.uniform(0.03, 0.3)
            pert = best * np.exp(scale * rng.standard_normal(N))
            pert += 0.02 * scale * best.mean() * rng.random(N)
            cand = lp_polish(make_valid_and_normalise(pert),
                             min(time.time() + 1.5, t0 + 25.0))
            cand = make_valid_and_normalise(cand)
            cv = value(cand)
            if cv < bv:
                best, bv = cand, cv
                fails = 0
            else:
                fails += 1
    except Exception:
        pass
    return make_valid_and_normalise(best)


def make_valid_and_normalise(step_heights):
    # Clip negative heights to zero so the vector is valid.
    step_heights = np.maximum(step_heights, 0.0)
    total_height = np.sum(step_heights)

    # An all-zero (or non-finite) vector is invalid; fall back to constants.
    if not np.isfinite(total_height) or total_height <= 0.0:
        step_heights = np.ones(step_heights.shape[0])
        total_height = np.sum(step_heights)

    # Rescale so the heights sum to 1 (the score is unaffected by scale).
    return step_heights / total_height


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
