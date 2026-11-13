"""Problem 7: size-penalised sumset/difference-set score (hidden from the blind arm)."""
import numpy as np

NAME = "p07_sumset_ratio"
DESCRIPTION = (
    "Choose a set A of between 1 and 60 distinct integers, each in the range 0..299. Let A+A be the "
    "set of all sums a+b and A-A the set of all differences a-b with a, b in A (a = b allowed). "
    "Maximise |A+A|^2 / (|A-A| * |A|^1.5). This is not the usual 'more sums than differences' ratio "
    "|A+A| / |A-A|: the sumset size is squared and the score is divided by the set size to the "
    "power 1.5, so large dense sets with a ratio slightly above 1 score poorly."
)
TIME_LIMIT_S = 30
LIMIT = 300      # elements must satisfy 0 <= a < LIMIT
MAX_SIZE = 60
J_REF = 2.5117951154236646  # best of 88 annealing runs, see reference.py


def raw(values):
    a = np.asarray(values, dtype=np.int64)
    sums = np.count_nonzero(np.bincount((a[:, None] + a[None, :]).ravel(), minlength=2 * LIMIT))
    diffs = np.count_nonzero(np.bincount((a[:, None] - a[None, :]).ravel() + LIMIT, minlength=2 * LIMIT))
    return int(sums), int(diffs)


def score(output):
    try:
        if isinstance(output, (str, bytes, dict)) or not isinstance(output, (list, tuple, np.ndarray)):
            return 0.0, f"run() must return a list of integers; got {type(output).__name__}."
        if isinstance(output, np.ndarray) and output.ndim != 1:
            return 0.0, f"run() must return a flat list of integers; got an array of shape {output.shape}."
        items = list(output)
        if not 1 <= len(items) <= MAX_SIZE:
            return 0.0, f"The set must have between 1 and {MAX_SIZE} elements; got {len(items)}."
        values = []
        for i, v in enumerate(items):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                return 0.0, f"Element {i} ({v!r:.40}) is not an integer."
            if isinstance(v, (float, np.floating)) and not (np.isfinite(v) and float(v) == int(v)):
                return 0.0, f"Element {i} ({v!r}) is not a finite whole number."
            v = int(v)
            if not 0 <= v < LIMIT:
                return 0.0, f"Element {i} has value {v}, outside the allowed range 0..{LIMIT - 1}."
            values.append(v)
        if len(set(values)) != len(values):
            seen = set()
            dup = next(v for v in values if v in seen or seen.add(v))
            return 0.0, f"Elements must be distinct; the value {dup} appears more than once."
        sums, diffs = raw(values)
        return float(sums ** 2 / (diffs * len(values) ** 1.5) / J_REF), None
    except Exception as exc:
        return 0.0, f"The value returned by run() could not be scored: {exc}"
