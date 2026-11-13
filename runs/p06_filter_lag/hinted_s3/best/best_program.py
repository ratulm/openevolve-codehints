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


def _sim(rng, n_sig, N=400):
    cl = np.zeros((n_sig, N))
    jumps = []
    for s in range(n_sig):
        while True:
            js = np.sort(rng.choice(np.arange(24, N - 1), 4, replace=False))
            if np.all(np.diff(js) >= 35):
                break
        c = np.zeros(N)
        lvl = rng.uniform(-1, 1)
        t = 0
        while t < N:
            L = int(rng.integers(40, 91))
            sl = rng.uniform(-0.03, 0.03)
            e = min(N, t + L)
            c[t:e] = lvl + sl * np.arange(e - t)
            lvl += sl * (e - t)
            t = e
        sz = rng.uniform(1, 3, 4) * rng.choice([-1.0, 1.0], 4)
        for j, a in zip(js, sz):
            c[j:] += a
        cl[s] = c
        jumps.append((js, sz))
    w = rng.normal(0, 0.45, (n_sig, N))
    e = np.zeros((n_sig, N))
    e[:, 0] = w[:, 0]
    for i in range(1, N):
        e[:, i] = 0.7 * e[:, i - 1] + w[:, i]
    x = cl + e + rng.normal(0, 0.45, (n_sig, N))
    return cl, x, jumps


def f():
    # Cost is quadratic in the taps -> solve the normal equations on
    # simulated data matching the described signal/noise model.
    K = 24
    rng = np.random.default_rng(12345)
    cl, x, jumps = _sim(rng, 1500)
    from numpy.lib.stride_tricks import sliding_window_view
    Mt = np.zeros((K, K)); vt = np.zeros(K); nt = 0
    Ml = np.zeros((K, K)); vl = np.zeros(K); nl = 0
    Mr = np.zeros((K, K)); nr = 0
    for s in range(x.shape[0]):
        X = sliding_window_view(x[s], K)[:, ::-1]  # row i -> t = i + 23
        y = cl[s, K - 1:]
        Mt += X.T @ X; vt += X.T @ y; nt += len(y)
        D = np.diff(X, axis=0)
        Mr += D.T @ D; nr += len(D)
        js, sz = jumps[s]
        for j, a in zip(js, sz):
            idx = np.arange(j, min(j + 8, x.shape[1])) - (K - 1)
            idx = idx[idx >= 0]
            Xl = X[idx] / abs(a); yl = y[idx] / abs(a)
            Ml += Xl.T @ Xl; vl += Xl.T @ yl; nl += len(idx)
    M = Mt / nt + 8 * Ml / nl + 0.1 * Mr / nr
    v = vt / nt + 8 * vl / nl
    try:
        h = np.linalg.solve(M + 1e-9 * np.eye(K), v)
        if not np.all(np.isfinite(h)):
            raise ValueError
    except Exception:
        h = np.ones(K) / K
    return g(h)


def g(filter_taps):
    # Limit every tap to the allowed range [-2, 2].
    num_taps = filter_taps.shape[0]
    limited_taps = np.zeros(num_taps)

    for tap_index in range(num_taps):
        limited_taps[tap_index] = min(max(filter_taps[tap_index], -2.0), 2.0)

    return limited_taps


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
