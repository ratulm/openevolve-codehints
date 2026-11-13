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
W = 1.0 + 3.0 * np.arange(2 * N - 1) / (2 * N - 2)


def V(a):
    s = a.sum()
    return np.max(W * np.convolve(a, a)) / (s * s) if s > 0 else np.inf


def jac(a):
    J = np.zeros((2 * N - 1, N))
    for i in range(N):
        J[i:i + N, i] = 2.0 * a
    return J


def lp_refine(a, deadline, iters=400):
    from scipy.optimize import linprog
    a = make_valid_and_normalise(a)
    best = V(a)
    r = 0.2 / N
    cost = np.r_[np.zeros(N), 1.0]
    Aeq = np.r_[np.ones(N), 0.0][None, :]
    for _ in range(iters):
        if time.time() > deadline or r < 1e-11:
            break
        c = np.convolve(a, a)
        A = np.hstack([W[:, None] * jac(a), -np.ones((2 * N - 1, 1))])
        b = -W * c
        bounds = [(max(-a[i], -r), r) for i in range(N)] + [(None, None)]
        res = linprog(cost, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[0.0],
                      bounds=bounds, method="highs")
        if not res.success:
            r *= 0.5
            continue
        na = make_valid_and_normalise(a + res.x[:N])
        v = V(na)
        if v < best:
            a, best = na, v
            r *= 1.3
        else:
            r *= 0.5
    return a


def grad_refine(a, deadline, iters=20000):
    # numpy-only fallback: softmax-smoothed max, parametrised by a = x^2
    x = np.sqrt(make_valid_and_normalise(a))
    best_a, best = make_valid_and_normalise(x * x), V(x * x)
    lr, beta = 0.02, 50.0
    for it in range(iters):
        if time.time() > deadline:
            break
        a = x * x
        s = a.sum()
        y = W * np.convolve(a, a) / (s * s)
        m = y.max()
        p = np.exp(beta * (y - m) / m)
        p /= p.sum()
        g_c = p * W / (s * s)
        g_a = 2.0 * np.correlate(g_c, a, mode="valid")[::-1][:N] if False else \
            2.0 * np.array([np.dot(g_c[i:i + N], a) for i in range(N)])
        g_a -= 2.0 * np.dot(p, y) / s
        x -= lr * g_a * 2.0 * x / (np.abs(g_a).max() + 1e-18) * 0.05
        x = np.abs(x)
        if V(x * x) < best:
            best, best_a = V(x * x), make_valid_and_normalise(x * x)
        if it % 2000 == 1999:
            beta = min(beta * 1.5, 2000.0)
    return best_a


def f():
    t0 = time.time()
    deadline = t0 + 20.0
    base = np.linspace(2.0, 1.0, N)  # front-loaded: late mass is costly
    best_a = make_valid_and_normalise(base)
    best = V(best_a)
    rng = np.random.default_rng(12345)
    starts = [base, np.linspace(3.0, 0.5, N), np.ones(N)]
    starts += [np.linspace(2.0, 0.5, N) + 0.5 * rng.random(N) for _ in range(6)]
    try:
        import scipy.optimize  # noqa: F401
        refine = lp_refine
    except Exception:
        refine = grad_refine
    ms_deadline = t0 + 13.0
    for s in starts:
        if time.time() > ms_deadline:
            break
        try:
            a = refine(s, min(ms_deadline, time.time() + 3.0))
        except Exception:
            continue
        v = V(a)
        if np.all(np.isfinite(a)) and v < best:
            best, best_a = v, a
    # Basin hopping: perturb the incumbent and re-refine until ~24s.
    hop_deadline = t0 + 24.0
    k = 0
    while time.time() < hop_deadline:
        k += 1
        scale = 0.15 if k % 3 else 0.4
        if k % 2:
            cand = best_a * (1.0 + scale * rng.standard_normal(N))
        else:
            cand = best_a + scale * best_a.mean() * rng.random(N) / (1 + k / 10)
        try:
            a = refine(cand, min(hop_deadline, time.time() + 1.5))
        except Exception:
            continue
        v = V(a)
        if np.all(np.isfinite(a)) and v < best:
            best, best_a = v, a
    try:
        a = refine(best_a, time.time() + 1.5)
        if np.all(np.isfinite(a)) and V(a) < best:
            best_a = a
    except Exception:
        pass
    return make_valid_and_normalise(best_a)


def make_valid_and_normalise(step_heights):
    # Clip negative heights to zero so the vector is valid.
    step_heights = np.nan_to_num(np.asarray(step_heights, dtype=float))
    step_heights = np.maximum(step_heights, 0.0)
    total_height = np.sum(step_heights)

    # An all-zero vector is invalid; fall back to constant heights.
    if total_height <= 0.0:
        step_heights = np.ones(step_heights.shape[0])
        total_height = np.sum(step_heights)

    # Rescale so the heights sum to 1 (the score is unaffected by scale).
    return step_heights / total_height


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
