"""Problem 8: position-weighted autoconvolution of a step function (hidden from the blind arm)."""
import numpy as np

NAME = "p08_step_autocorr"
DESCRIPTION = (
    "Choose 50 non-negative heights a_0..a_49 of a step function (50 equal-width steps on an "
    "interval), not all zero. Let c_k = sum_i a_i * a_(k-i) for k = 0..98 be its discrete "
    "autoconvolution and let w_k = 1 + 3*k/98 be a position weight that grows linearly from 1 at "
    "shift k = 0 to 4 at shift k = 98. Minimise the weighted peak "
    "V = max_k (w_k * c_k) / (sum_i a_i)^2. This differs from the usual first autocorrelation "
    "inequality, which minimises the unweighted max_k c_k / (sum_i a_i)^2: here autoconvolution "
    "mass at large shifts costs up to four times more than at small shifts. "
    "The score is V_REF / V with V_REF = 0.0287461148884, so lower V is better."
)
TIME_LIMIT_S = 30
N = 50
W = 1.0 + 3.0 * np.arange(2 * N - 1) / (2 * N - 2)
V_REF = 0.028746114888427602  # best of 240 SLSQP + successive-LP starts, see reference.py


def score(output):
    try:
        a = np.asarray(output, dtype=float)
    except Exception:
        return 0.0, f"run() must return a sequence of {N} real numbers (the step heights)."
    if a.shape != (N,):
        return 0.0, f"Expected a one-dimensional vector of exactly {N} step heights; got shape {a.shape}."
    if not np.all(np.isfinite(a)):
        i = int(np.argmin(np.isfinite(a)))
        return 0.0, f"Height {i} is {a[i]}; all heights must be finite (no NaN or infinity)."
    if np.any(a < 0):
        i = int(np.argmin(a))
        return 0.0, f"Height {i} is {a[i]:.6g}; all heights must be non-negative."
    top = float(np.max(a))
    if top <= 0:
        return 0.0, "All heights are zero; at least one height must be positive."
    a = a / top  # the functional is scale-invariant; this avoids overflow and underflow
    v = float(np.max(W * np.convolve(a, a)) / np.sum(a) ** 2)
    return V_REF / v, None
