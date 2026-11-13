"""Problem 9: single-machine weighted tardiness with hidden family setup times
(the setup matrix is hidden from the blind arm)."""
import numpy as np

NAME = "p09_scheduling"
TIME_LIMIT_S = 30
N = 36
# Instance drawn once with numpy default_rng(9); see reference.py for the recipe.
PROC = [6, 11, 12, 5, 3, 8, 9, 10, 9, 9, 12, 12, 12, 11, 9, 12, 2, 2, 10, 6, 10, 7, 11, 2, 9, 2, 3, 11, 5,
        12, 5, 10, 12, 5, 12, 9]
DUE = [22, 170, 23, 40, 66, 110, 138, 13, 164, 135, 56, 33, 105, 105, 47, 29, 135, 119, 141, 38, 128, 13,
       1, 31, 138, 95, 109, 104, 112, 57, 127, 2, 66, 9, 22, 160]
WEIGHT = [9, 3, 5, 7, 9, 3, 5, 8, 8, 9, 2, 9, 2, 8, 8, 9, 8, 7, 6, 5, 1, 9, 9, 1, 6, 4, 7, 6, 6, 5, 2, 8,
          8, 2, 5, 1]
NUM_FAMILIES = 4
SETUP_TIME = 3
FAMILY = [j % NUM_FAMILIES for j in range(N)]
SETUP = [[0 if FAMILY[a] == FAMILY[b] else SETUP_TIME for b in range(N)] for a in range(N)]
REF_COST = 10608  # best of 12 annealing restarts (all 12 agree), see reference.py

DESCRIPTION = (
    f"Schedule {N} jobs (indexed 0..{N - 1}) on a single machine that processes one job at a time "
    "without preemption or idle time, starting at time 0. Return the processing order as a "
    f"permutation of the job indices. Processing times: {PROC}. Due dates: {DUE}. Weights: {WEIGHT}. "
    f"Each job belongs to a family given by (job index mod {NUM_FAMILIES}). Whenever two consecutive "
    f"jobs in the order belong to different families, a setup time of {SETUP_TIME} time units is "
    "inserted before the second of them (no setup between jobs of the same family, and none before "
    "the first job). The completion time of a job is the sum of all processing times and setup "
    "times up to and including that job. Minimise the total weighted tardiness, the sum over jobs "
    "of weight_j * max(0, completion_j - due_j), with the setup times included in the completion "
    f"times. The score is {REF_COST} divided by that cost."
)


def true_cost(order):
    t = 0
    total = 0
    prev = -1
    for j in order:
        if prev >= 0:
            t += SETUP[prev][j]
        t += PROC[j]
        if t > DUE[j]:
            total += WEIGHT[j] * (t - DUE[j])
        prev = j
    return total


def score(output):
    try:
        arr = np.asarray(output)
    except Exception:
        return 0.0, f"run() must return a sequence of {N} job indices (a permutation of 0..{N - 1})."
    if arr.dtype == object or arr.dtype.kind not in "iuf":
        return 0.0, (f"run() must return a flat sequence of {N} integer job indices; got an object of "
                     f"type {type(output).__name__} with element type {arr.dtype}.")
    if arr.shape != (N,):
        return 0.0, f"Expected a flat sequence of exactly {N} job indices; got shape {arr.shape}."
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            k = int(np.argmin(np.isfinite(arr)))
            return 0.0, f"Position {k} of the order holds a NaN or infinite value."
        if np.any(arr != np.round(arr)):
            k = int(np.argmax(arr != np.round(arr)))
            return 0.0, f"Position {k} of the order holds {arr[k]!r}, which is not an integer job index."
    order = [int(v) for v in arr]
    for k, j in enumerate(order):
        if j < 0 or j >= N:
            return 0.0, f"Position {k} of the order holds {j}, which is not a job index in 0..{N - 1}."
    seen = {}
    for k, j in enumerate(order):
        if j in seen:
            missing = sorted(set(range(N)) - set(order))
            return 0.0, (f"Job {j} appears twice in the order (positions {seen[j]} and {k}); each job "
                         f"must appear exactly once. Missing jobs: {missing}.")
        seen[j] = k
    cost = true_cost(order)
    return float(REF_COST / max(cost, 1e-9)), None
