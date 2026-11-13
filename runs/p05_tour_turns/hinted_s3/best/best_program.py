# EVOLVE-BLOCK-START
# Goal: plan a closed tour over the 46 fixed points in POINTS (defined below the
# evolved block). The tour visits every point exactly once and returns to its
# start. plan_tour returns the visiting order as a permutation of 0..45.
#
# Score: the evaluator computes
#     cost = tour_length + 10 * sum over all 46 tour vertices of turning_angle**2
# and the score is reference_cost / cost, with reference_cost about 1116.27, so
# 1.0 means matching the reference (higher is possible). tour_length is the
# Euclidean length of the closed tour, including the leg from the last point
# back to the first. The turning angle at a vertex is the angle in radians
# between the direction of the leg arriving there and the direction of the leg
# leaving it: 0 for straight on, pi/2 for a right-angle turn, pi for a full
# reversal. The first and last points of the order are vertices like any other.
# One right-angle turn costs as much as 24.7 units of length and one reversal
# as much as 98.7, on a board about 100 units wide. The cost is NOT the tour
# length alone, and the shortest tour is not the best one.
#
# Validity: a flat sequence of exactly 46 integers, each of 0..45 appearing
# exactly once. Any violation, an exception, or a run longer than 30 seconds
# scores 0.
import numpy as np
import math
import random
import time


def plan_tour(point_coordinates):
    P = point_coordinates
    n = len(P)
    rng = random.Random(12345)
    # Precompute pairwise distances and leg headings.
    D = np.hypot(P[None, :, 0] - P[:, None, 0], P[None, :, 1] - P[:, None, 1]).tolist()
    A = np.arctan2(P[None, :, 1] - P[:, None, 1], P[None, :, 0] - P[:, None, 0]).tolist()
    TP = 2 * math.pi

    def cost(o):
        s = 0.0
        for i in range(n):
            a, b, c = o[i - 2], o[i - 1], o[i]
            t = abs(A[b][c] - A[a][b])
            if t > math.pi:
                t = TP - t
            s += D[b][c] + 10.0 * t * t
        return s

    # Starting tour: angular sort around the centroid.
    c0 = P.mean(axis=0)
    cur = [int(x) for x in np.argsort(np.arctan2(P[:, 1] - c0[1], P[:, 0] - c0[0]))]
    best = cur[:]
    bc = cost(cur)

    rounds, total_time = 3, 10.0
    for _ in range(rounds):
        cur, cc = best[:], bc
        T0, T1 = 40.0, 0.05
        ts = time.time()
        Lr = total_time / rounds
        while True:
            el = (time.time() - ts) / Lr
            if el >= 1.0:
                break
            T = T0 * (T1 / T0) ** el
            for _ in range(300):
                i, j = sorted(rng.sample(range(n), 2))
                m = rng.random()
                if m < 0.5:  # 2-opt segment reversal
                    new = cur[:i] + cur[i:j + 1][::-1] + cur[j + 1:]
                elif m < 0.85:  # or-opt: move a short chain (maybe reversed)
                    seg = cur[i:i + rng.randint(1, 4)]
                    rest = [x for x in cur if x not in seg]
                    k = rng.randint(0, len(rest))
                    if rng.random() < 0.5:
                        seg = seg[::-1]
                    new = rest[:k] + seg + rest[k:]
                else:  # swap two points
                    new = cur[:]
                    new[i], new[j] = new[j], new[i]
                nc = cost(new)
                if nc < cc or rng.random() < math.exp((cc - nc) / T):
                    cur, cc = new, nc
                    if cc < bc:
                        best, bc = cur[:], cc
    return best


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
# The instance: (x, y) coordinates of the 46 points; row i is point index i.
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


def run():
    visiting_order = plan_tour(POINTS)
    return [int(point_index) for point_index in visiting_order]
