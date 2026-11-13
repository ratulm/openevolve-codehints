# EVOLVE-BLOCK-START
# Goal: choose the 24 taps h[0..23] of a causal FIR filter that cleans a noisy
# signal: output[t] = sum_k h[k] * noisy[t - k], so h[0] multiplies the newest
# sample and h[23] the oldest. This program returns ONLY the taps. The evaluator
# applies them to its own fixed set of hidden test signals, which this program
# never sees.
#
# Hidden test signals: 64 signals of 400 samples each, noisy = clean + noise.
#   clean: a continuous piecewise-linear drift (random start level in [-1, 1],
#     slope redrawn uniformly from [-0.03, 0.03] per sample every 40 to 90
#     samples) plus 4 abrupt jumps per signal. Each jump has magnitude uniform
#     in [1, 3] with random sign, and jumps are at least 35 samples apart.
#   noise: coloured noise e[t] = 0.7 * e[t-1] + w[t] with w Gaussian of standard
#     deviation 0.45, plus independent white Gaussian noise of standard
#     deviation 0.45.
#
# Score: over samples t >= 23 of all signals the evaluator computes
#   cost = E_track + 8 * E_lag + 0.1 * E_rough
#   E_track = mean of (output - clean)**2 over all samples
#   E_lag   = mean, over every jump and over the 8 samples starting at the jump
#             (t = jump, ..., jump + 7), of ((output - clean) / jump_size)**2,
#             where clean already includes the jump
#   E_rough = mean of (output[t+1] - output[t])**2
# and the score is reference_cost / cost with reference_cost about 1.452, so 1.0
# means matching the reference (higher is possible). Being late after a jump is
# penalised heavily and roughness only lightly. The score is NOT the plain mean
# squared error to the clean signal and NOT a smoothness measure: a filter that
# smooths well but responds late to jumps scores badly.
#
# Validity: run() must return exactly 24 finite numbers (a one-dimensional
# array), each in [-2, 2] (no tolerance). Any violation, an exception, or a run
# longer than 30 seconds scores 0.
import numpy as np


def _gen(rng, N=400):
    # Simulate one signal following the stated generator.
    c = np.zeros(N)
    t = 0
    lv = rng.uniform(-1, 1)
    while t < N:
        L = int(rng.integers(40, 91))
        s = rng.uniform(-0.03, 0.03)
        m = min(L, N - t)
        c[t:t + m] = lv + s * np.arange(m)
        lv = lv + s * m
        t += m
    while True:
        j = np.sort(rng.choice(np.arange(23, N - 8), 4, replace=False))
        if np.all(np.diff(j) >= 35):
            break
    sz = rng.uniform(1, 3, 4) * rng.choice([-1.0, 1.0], 4)
    for jj, s in zip(j, sz):
        c[jj:] += s
    w = rng.normal(0, 0.45, N)
    e = np.zeros(N)
    acc = 0.0
    for i in range(N):
        acc = 0.7 * acc + w[i]
        e[i] = acc
    x = c + e + rng.normal(0, 0.45, N)
    return c, x, j, np.abs(sz)


def f():
    # Cost is quadratic in h -> solve the normal equations on simulated data.
    K = 24
    rng = np.random.default_rng(12345)
    A = np.zeros((K, K))
    b = np.zeros(K)
    for _ in range(1500):
        c, x, j, sz = _gen(rng)
        X = np.lib.stride_tricks.sliding_window_view(x, K)[:, ::-1]
        y = c[K - 1:]
        n = len(y)
        A += X.T @ X / n
        b += X.T @ y / n
        D = X[1:] - X[:-1]
        A += 0.1 * D.T @ D / (n - 1)
        rows = (j[:, None] - (K - 1) + np.arange(8)[None, :]).ravel()
        scale = np.repeat(1.0 / sz, 8)
        Xl = X[rows] * scale[:, None]
        yl = y[rows] * scale
        A += 8.0 * Xl.T @ Xl / len(rows)
        b += 8.0 * Xl.T @ yl / len(rows)
    A += 1e-9 * np.eye(K)
    h = np.linalg.solve(A, b)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return g(h)


def g(filter_taps):
    # Limit every tap to the allowed range [-2, 2].
    num_taps = filter_taps.shape[0]
    limited_taps = np.zeros(num_taps)

    for tap_index in range(num_taps):
        limited_taps[tap_index] = min(max(float(filter_taps[tap_index]), -2.0), 2.0)

    return limited_taps


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
