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


def _gen(rng, n=400):
    c = np.zeros(n)
    lvl = rng.uniform(-1, 1)
    t = 0
    while t < n:
        L = int(rng.integers(40, 91))
        s = rng.uniform(-0.03, 0.03)
        m = min(L, n - t)
        c[t:t + m] = lvl + s * np.arange(m)
        lvl = lvl + s * m
        t += m
    while True:
        j = np.sort(rng.integers(25, n - 8, 4))
        if np.all(np.diff(j) >= 35):
            break
    jumps = []
    for p in j:
        mag = rng.uniform(1, 3)
        c[p:] += mag * (1 if rng.random() < 0.5 else -1)
        jumps.append((int(p), mag))
    w = rng.normal(0, 0.45, n)
    e = np.zeros(n)
    e[0] = w[0] / np.sqrt(1 - 0.49)
    for i in range(1, n):
        e[i] = 0.7 * e[i - 1] + w[i]
    x = c + e + rng.normal(0, 0.45, n)
    return c, x, jumps


def f():
    # Directly minimise the (quadratic) evaluation cost over simulated signals.
    K = 24
    rng = np.random.default_rng(12345)
    HT = np.zeros((K, K)); gT = np.zeros(K); nT = 0
    HL = np.zeros((K, K)); gL = np.zeros(K); nL = 0
    HR = np.zeros((K, K)); nR = 0
    for _ in range(400):
        c, x, jumps = _gen(rng)
        # X[i, k] = x[t - k] with t = i + 23
        X = np.lib.stride_tricks.sliding_window_view(x, K)[:, ::-1]
        y = c[K - 1:]
        HT += X.T @ X; gT += X.T @ y; nT += len(y)
        D = X[1:] - X[:-1]
        HR += D.T @ D; nR += len(D)
        rows = [t - (K - 1) for p, _ in jumps for t in range(p, p + 8) if t >= K - 1]
        sc = np.array([1.0 / m for p, m in jumps for t in range(p, p + 8) if t >= K - 1])
        if rows:
            XL = X[rows] * sc[:, None]
            yl = y[rows] * sc
            HL += XL.T @ XL; gL += XL.T @ yl; nL += len(rows)
    H = HT / nT + 8.0 * HL / max(nL, 1) + 0.1 * HR / nR
    gv = gT / nT + 8.0 * gL / max(nL, 1)
    try:
        h = np.linalg.solve(H + 1e-9 * np.eye(K), gv)
    except Exception:
        h = np.ones(K) / K
    if not np.all(np.isfinite(h)):
        h = np.ones(K) / K
    return g(np.asarray(h, dtype=float))


def g(filter_taps):
    # Limit every tap to the allowed range [-2, 2].
    return np.clip(np.asarray(filter_taps, dtype=float), -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
