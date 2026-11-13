"""Problem 5: closed tour with a turning penalty (hidden from the blind arm)."""
import numpy as np

NAME = "p05_tour_turns"
POINTS = np.array([
    [38, 12], [46, 57], [10, 34], [66, 67], [86, 96], [28, 22],
    [30, 72], [26, 79], [6, 79], [59, 69], [60, 54], [62, 94],
    [93, 16], [76, 65], [71, 60], [91, 40], [59, 86], [17, 78],
    [27, 48], [23, 15], [67, 27], [64, 37], [45, 48], [6, 72],
    [45, 77], [33, 7], [4, 61], [42, 29], [26, 90], [86, 62],
    [65, 83], [49, 23], [38, 77], [81, 34], [96, 71], [7, 5],
    [82, 70], [34, 43], [13, 23], [34, 90], [77, 12], [97, 8],
    [80, 21], [71, 36], [85, 78], [60, 7],
], dtype=float)
N = len(POINTS)
TURN_WEIGHT = 10.0
DESCRIPTION = (
    "Plan a closed tour that visits each of 46 fixed points in the plane exactly once and returns "
    "to the start. The points have integer coordinates in [3, 97] x [3, 97], were drawn at random "
    "with a fixed seed and are at least 6 apart (this is not a TSPLIB instance). The answer is a "
    "permutation of the indices 0..45. The cost to minimise is NOT the tour length alone: it is "
    "tour length + 10 * (sum over all 46 tour vertices of turning_angle^2), where the turning angle "
    "at a vertex is the angle in radians, between 0 (straight on) and pi (full reversal), between "
    "the direction of the leg arriving at the vertex and the direction of the leg leaving it. "
    "The wrap-around vertices count too. The score is reference_cost / cost with reference_cost "
    "about 1116.27, so smooth tours with few sharp turns beat the shortest tour."
)
TIME_LIMIT_S = 30
COST_REF = 1116.2663276720655  # best of 24 annealing runs of 1e6 steps each, see reference.py


def tour_cost(order):
    p = POINTS[order]
    leave = np.roll(p, -1, axis=0) - p
    arrive = np.roll(leave, 1, axis=0)
    cross = arrive[:, 0] * leave[:, 1] - arrive[:, 1] * leave[:, 0]
    dot = np.sum(arrive * leave, axis=1)
    turn = np.arctan2(np.abs(cross), dot)
    return float(np.hypot(leave[:, 0], leave[:, 1]).sum() + TURN_WEIGHT * np.sum(turn ** 2))


def score(output):
    try:
        if isinstance(output, (str, bytes, dict)) or output is None:
            return 0.0, f"run() must return a sequence of {N} point indices; got {type(output).__name__}."
        a = np.asarray(output)
        if a.dtype == object or a.dtype.kind not in "iuf":
            return 0.0, f"run() must return a flat sequence of {N} integer point indices; got element type {a.dtype}."
        if a.shape != (N,):
            return 0.0, f"Expected a flat sequence of {N} point indices; got shape {a.shape}."
        if not np.all(np.isfinite(a)):
            return 0.0, "The tour contains NaN or infinite values."
        if np.any(a != np.round(a)):
            k = int(np.argmax(a != np.round(a)))
            return 0.0, f"Tour position {k} holds {a[k]!r}, which is not an integer point index."
        a = np.round(a).astype(np.int64)
        bad = (a < 0) | (a >= N)
        if np.any(bad):
            k = int(np.argmax(bad))
            return 0.0, f"Tour position {k} holds index {int(a[k])}, outside the valid range 0..{N - 1}."
        counts = np.bincount(a, minlength=N)
        if np.any(counts != 1):
            twice = int(np.argmax(counts > 1))
            missing = int(np.argmax(counts == 0))
            return 0.0, (f"The tour must visit every point exactly once: point {twice} appears "
                         f"{int(counts[twice])} times and point {missing} is missing.")
        cost = tour_cost(a)
        if not np.isfinite(cost) or cost <= 0:
            return 0.0, "The tour cost could not be computed."
        return COST_REF / cost, None
    except Exception as exc:
        return 0.0, f"The value returned by run() could not be scored: {exc}"
