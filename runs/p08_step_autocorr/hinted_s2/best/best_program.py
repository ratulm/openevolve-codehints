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


def score_v(a):
    s = a.sum()
    if s <= 0:
        return np.inf
    return np.max(W * np.convolve(a, a)) / s ** 2


def adam_opt(x, iters, deadline):
    m = np.zeros_like(x); v = np.zeros_like(x)
    lr, b1, b2 = 0.02, 0.9, 0.999
    best_a, best_v = x * x, score_v(x * x)
    for t in range(1, iters + 1):
        if t % 200 == 0 and time.time() > deadline:
            break
        beta = 30.0 * (100.0 ** (t / iters))
        a = x * x
        S = a.sum()
        y = W * np.convolve(a, a) / S ** 2
        z = beta * (y - y.max())
        p = np.exp(z); p /= p.sum()
        L = float(np.dot(p, y))
        ga = 2.0 * np.correlate(p * W, a, 'valid') / S ** 2 - 2.0 * L / S
        g = ga * 2.0 * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        step = lr * (1.0 - 0.9 * t / iters)
        x = x - step * mh / (np.sqrt(vh) + 1e-12)
        x *= 1.0 / np.sqrt(np.sum(x * x))  # keep scale fixed (sum a = 1)
        if t % 50 == 0:
            vv = score_v(x * x)
            if vv < best_v:
                best_v, best_a = vv, (x * x).copy()
    vv = score_v(x * x)
    if vv < best_v:
        best_v, best_a = vv, (x * x).copy()
    return best_a, best_v


def conv_jac(a):
    # d c_k / d a_j = 2 a_{k-j}
    M = np.zeros((2 * N - 1, N))
    for j in range(N):
        M[j:j + N, j] = 2.0 * a
    return M


def slsqp_polish(a, deadline=None, maxiter=400):
    try:
        from scipy.optimize import minimize
    except Exception:
        return a
    a = np.maximum(np.nan_to_num(a), 0.0)
    if a.sum() <= 0:
        return a
    a = a / a.sum()
    x0 = np.append(a, score_v(a))
    ones_col = np.ones((2 * N - 1, 1))
    cons = [dict(type='ineq',
                 fun=lambda x: x[-1] - W * np.convolve(x[:-1], x[:-1]),
                 jac=lambda x: np.hstack([-W[:, None] * conv_jac(x[:-1]), ones_col])),
            dict(type='eq', fun=lambda x: x[:-1].sum() - 1.0,
                 jac=lambda x: np.append(np.ones(N), 0.0))]
    try:
        r = minimize(lambda x: x[-1], x0, method='SLSQP',
                     jac=lambda x: np.append(np.zeros(N), 1.0),
                     constraints=cons, bounds=[(0, None)] * (N + 1),
                     options=dict(maxiter=maxiter, ftol=1e-15))
        b = np.maximum(np.nan_to_num(r.x[:-1]), 0.0)
        if b.sum() > 0 and np.all(np.isfinite(b)):
            return b / b.sum()
    except Exception:
        pass
    return a


def f():
    start = time.time()
    deadline = start + 9.0
    best = make_valid_and_normalise(1.0 + np.arange(N, dtype=float))
    best_v = score_v(best)
    rng = np.random.default_rng(12345)
    starts = [np.ones(N), np.linspace(2.0, 0.5, N), np.linspace(1.5, 1.0, N)]
    for _ in range(20):
        starts.append(rng.random(N) + 0.2)
    results = []
    for s in starts:
        if time.time() > deadline:
            break
        x = np.sqrt(s / s.sum())
        a, v = adam_opt(x, 1500, deadline)
        results.append((v, a / a.sum()))
        if v < best_v:
            best_v, best = v, a / a.sum()
    results.sort(key=lambda r: r[0])

    def try_cand(c):
        nonlocal best, best_v
        c = np.maximum(np.nan_to_num(c), 0.0)
        if c.sum() <= 0:
            return
        cv = score_v(c)
        if np.isfinite(cv) and cv < best_v:
            best_v, best = cv, c / c.sum()

    for v, a in results[:5]:
        if time.time() > start + 15.0:
            break
        try_cand(slsqp_polish(a))
    # Basin hopping around the incumbent with exact SLSQP polishing.
    while time.time() < start + 24.0:
        sigma = rng.choice([0.03, 0.1, 0.3])
        c = best * np.exp(sigma * rng.standard_normal(N))
        if rng.random() < 0.3:
            c[rng.integers(N, size=2)] = 0.0
        c = c + 1e-5 * rng.random(N)
        try_cand(slsqp_polish(c, maxiter=300))
    return make_valid_and_normalise(best)


def make_valid_and_normalise(step_heights):
    # Clip negative heights to zero so the vector is valid.
    step_heights = np.maximum(np.nan_to_num(step_heights), 0.0)
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
