"""Problem 10: fit a two-input law from noisy data, scored on out-of-range queries only, asymmetrically
(hidden from the blind arm)."""
import numpy as np

from query_inputs import QUERY_INPUTS

NAME = "p10_curve_fit"
DESCRIPTION = (
    "A program is given 80 noisy measurements y of an unknown, invented two-input law y = F(x1, x2) "
    "(not a named physical law), taken at points with x1 in [0.5, 4] and x2 in [0, 5], with Gaussian "
    "noise of standard deviation 0.08. It must return predictions of y at a fixed list of 50 query "
    "inputs given in shuffled order. The 25 queries inside the measured region are not scored. The 25 "
    "queries outside it (x1 up to 7 and/or x2 up to 8.5) are the only ones scored, with an asymmetric "
    "loss: with d = prediction - true noise-free F, the error is sqrt(mean(w * d^2)) over those 25 "
    "queries, where w = 8 when the prediction is too low (d < 0) and w = 1 otherwise. The score is a "
    "reference error (that of the best model found offline from the same 80 points, without knowledge "
    "of F, for this asymmetric extrapolation loss) divided by this error. So the task is to "
    "extrapolate while erring high, not to minimise symmetric error on the measured points or on all "
    "queries."
)
TIME_LIMIT_S = 30
N_QUERIES = 50
UNDER_WEIGHT = 8.0  # weight on squared error where the prediction is below the truth
E_REF = 1.1182187272386428  # asymmetric extrapolation error of reference_output.pkl, see reference.py


def true_law(x1, x2):
    """The hidden noise-free law. Never shown to either arm."""
    return 3.2 * x1 ** 2 / (x1 ** 2 + 2.0) * np.cos(0.45 * x2 + 0.3) + 0.3 * (x1 + 0.4 * x2) ** 1.4


_Q = np.asarray(QUERY_INPUTS, dtype=float)
TRUE_VALUES = true_law(_Q[:, 0], _Q[:, 1])
# Only queries outside the training box (x1 in [0.5, 4], x2 in [0, 5]) carry weight.
SCORED = np.flatnonzero((_Q[:, 0] > 4.0) | (_Q[:, 1] > 5.0))


def asymmetric_loss(pred, truth):
    d = np.asarray(pred, dtype=float) - truth
    return float(np.sqrt(np.mean(np.where(d < 0, UNDER_WEIGHT, 1.0) * d ** 2)))


def extrapolation_error(pred):
    return asymmetric_loss(np.asarray(pred, dtype=float)[SCORED], TRUE_VALUES[SCORED])


def score(output):
    try:
        if isinstance(output, (str, bytes, dict)) or output is None:
            return 0.0, (f"run() must return a 1-D sequence of {N_QUERIES} numbers (one prediction per "
                         f"query input, in order); got {type(output).__name__}.")
        p = np.asarray(output)
        if p.dtype == object or not (np.issubdtype(p.dtype, np.number) or p.dtype == bool):
            return 0.0, (f"run() must return {N_QUERIES} real numbers; the returned value could not be "
                         f"read as a numeric array (dtype {p.dtype}).")
        if np.iscomplexobj(p):
            return 0.0, "Predictions must be real numbers; got complex values."
        p = p.astype(float)
        if p.shape != (N_QUERIES,):
            return 0.0, (f"Expected predictions of shape ({N_QUERIES},), one per query input in order; "
                         f"got shape {p.shape}.")
        bad = np.flatnonzero(~np.isfinite(p))
        if bad.size:
            return 0.0, (f"{bad.size} prediction(s) are NaN or infinite; the first is query {int(bad[0])} "
                         f"at (x1, x2) = ({_Q[bad[0], 0]:.3f}, {_Q[bad[0], 1]:.3f}) with value {p[bad[0]]}.")
        e = extrapolation_error(p)
        return float(E_REF / max(e, 1e-12)), None
    except Exception as exc:
        return 0.0, f"The value returned by run() could not be scored: {exc}"
