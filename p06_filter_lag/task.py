"""Problem 6: causal FIR filter scored on tracking, lag after jumps and roughness
(hidden from the blind arm)."""
import numpy as np
from scipy.signal import lfilter

NAME = "p06_filter_lag"
DESCRIPTION = (
    "Design the 24 taps h[0..23] of a causal FIR filter, y[t] = sum_k h[k] * x[t-k], each tap in "
    "[-2, 2]. The program returns only the taps; the evaluator applies them to its own fixed set of "
    "64 hidden synthetic signals of 400 samples. Each clean signal is a continuous piecewise-linear "
    "drift (slope redrawn uniformly in [-0.03, 0.03] every 40 to 90 samples) plus 4 abrupt jumps of "
    "magnitude 1 to 3 with random sign, at least 35 samples apart. The observed signal adds coloured "
    "noise (AR(1) with coefficient 0.7 and innovation standard deviation 0.45) and white noise "
    "(standard deviation 0.45). Over samples t >= 23 the evaluator computes cost = E_track + "
    "8 * E_lag + 0.1 * E_rough, where E_track is the mean squared error between the output and the "
    "clean signal, E_lag is the mean over all jumps and over the 8 samples starting at each jump of "
    "((output - clean) / jump size)^2, and E_rough is the mean squared first difference of the "
    "output. The score is reference_cost / cost, to be maximised. It is not plain mean squared "
    "error: a filter that smooths well but responds late to jumps scores badly."
)
TIME_LIMIT_S = 30

N_TAPS = 24
TAP_BOUND = 2.0
N_SIGNALS = 64
N_SAMPLES = 400
N_JUMPS = 4
LAG_WINDOW = 8
W_LAG = 8.0
W_ROUGH = 0.1
SEED = 60606
C_REF = 1.4524375427441734  # cost of the exact minimiser of the (quadratic) cost, see reference.py


def _make_signals():
    rng = np.random.default_rng(SEED)
    clean_all, noisy_all, jumps_all = [], [], []
    for _ in range(N_SIGNALS):
        # continuous piecewise-linear drift
        slope = np.zeros(N_SAMPLES)
        t = 0
        while t < N_SAMPLES:
            n = int(rng.integers(40, 91))
            slope[t:t + n] = rng.uniform(-0.03, 0.03)
            t += n
        clean = rng.uniform(-1.0, 1.0) + np.cumsum(slope)
        # abrupt jumps, at least 35 samples apart, away from both ends
        while True:
            where = np.sort(rng.integers(40, N_SAMPLES - 20, size=N_JUMPS))
            if np.all(np.diff(where) >= 35):
                break
        sizes = rng.uniform(1.0, 3.0, size=N_JUMPS) * rng.choice([-1.0, 1.0], size=N_JUMPS)
        for s, d in zip(where, sizes):
            clean[s:] += d
        coloured = lfilter([1.0], [1.0, -0.7], rng.normal(0.0, 0.45, N_SAMPLES + 100))[100:]
        white = rng.normal(0.0, 0.45, N_SAMPLES)
        clean_all.append(clean)
        noisy_all.append(clean + coloured + white)
        jumps_all.append((where, sizes))
    return clean_all, noisy_all, jumps_all


def _build():
    clean_all, noisy_all, jumps_all = _make_signals()
    first = N_TAPS - 1
    rows = np.arange(first, N_SAMPLES)
    lagged, target, lag_idx, lag_scale, starts = [], [], [], [], []
    offset = 0
    for clean, noisy, (where, sizes) in zip(clean_all, noisy_all, jumps_all):
        # column k holds x[t-k]
        lagged.append(noisy[rows[:, None] - np.arange(N_TAPS)[None, :]])
        target.append(clean[rows])
        for s, d in zip(where, sizes):
            for j in range(LAG_WINDOW):
                lag_idx.append(offset + s + j - first)
                lag_scale.append(1.0 / abs(d))
        starts.append(offset)
        offset += len(rows)
    return (np.vstack(lagged), np.concatenate(target), np.array(lag_idx), np.array(lag_scale),
            np.array(starts), len(rows))


X, CLEAN, LAG_IDX, LAG_SCALE, STARTS, ROWS_PER_SIGNAL = _build()
# first differences are taken within a signal only
_keep = np.ones(len(CLEAN) - 1, dtype=bool)
_keep[STARTS[1:] - 1] = False
DIFF_IDX = np.nonzero(_keep)[0]


def components(h):
    """Returns (E_track, E_lag, E_rough) for a valid tap vector."""
    y = X @ h
    err = y - CLEAN
    e_track = float(np.mean(err ** 2))
    e_lag = float(np.mean((err[LAG_IDX] * LAG_SCALE) ** 2))
    e_rough = float(np.mean((y[DIFF_IDX + 1] - y[DIFF_IDX]) ** 2))
    return e_track, e_lag, e_rough


def cost(h):
    e_track, e_lag, e_rough = components(h)
    return e_track + W_LAG * e_lag + W_ROUGH * e_rough


def score(output):
    try:
        h = np.asarray(output, dtype=float)
    except Exception:
        return 0.0, f"run() must return a sequence of {N_TAPS} numbers (the filter taps h[0..{N_TAPS - 1}])."
    if h.shape != (N_TAPS,):
        return 0.0, (f"Expected a one-dimensional array of exactly {N_TAPS} filter taps; got shape "
                     f"{h.shape}. Pad with zeros to use a shorter filter.")
    if not np.all(np.isfinite(h)):
        k = int(np.argmin(np.isfinite(h)))
        return 0.0, f"Tap {k} is {h[k]}; all taps must be finite numbers."
    if np.any(np.abs(h) > TAP_BOUND):
        k = int(np.argmax(np.abs(h)))
        return 0.0, f"Tap {k} is {h[k]:.4f}; every tap must lie in [-{TAP_BOUND}, {TAP_BOUND}]."
    c = cost(h)
    if not np.isfinite(c) or c <= 0:
        return 0.0, "The filter output is not finite on the test signals."
    return float(C_REF / c), None
