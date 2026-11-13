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


def plan_tour(Pc, budget=18.0):
    # SA on the exact objective, using a precomputed triple table
    # W[a][b][c] = len(b->c) + 10*turn(a,b,c)^2 so a full cost is 46 lookups.
    P = [tuple(p) for p in Pc.tolist()]
    n = len(P); R = range(n)
    D = [[math.dist(P[a], P[b]) for b in R] for a in R]
    H = [[math.atan2(P[b][1]-P[a][1], P[b][0]-P[a][0]) for b in R] for a in R]
    TP = 2 * math.pi

    def tc(x):
        x = abs(x) % TP; x = min(x, TP - x); return 10 * x * x
    W = [[[D[b][c] + tc(H[b][c] - H[a][b]) if a != b and b != c else 0.0
           for c in R] for b in R] for a in R]

    def cost(o):
        return sum(W[o[i-2]][o[i-1]][o[i]] for i in R)

    rng = random.Random(1)
    c = Pc.mean(axis=0)
    best = [int(x) for x in np.argsort(np.arctan2(Pc[:, 1]-c[1], Pc[:, 0]-c[0]))]
    bc = cost(best)
    runs = 8; dur = budget / runs
    for r in range(runs):
        o = best[:] if r % 2 else rng.sample(range(n), n)
        oc = cost(o); ts = time.time(); it = 0; T = 40.0
        while True:
            if it % 256 == 0:
                f = (time.time() - ts) / dur
                if f >= 1:
                    break
                T = 40.0 * (0.01 / 40.0) ** f
            it += 1
            i, j = sorted(rng.sample(range(n), 2))
            m = rng.random()
            if m < 0.5:
                no = o[:i] + o[i:j+1][::-1] + o[j+1:]
            else:
                seg = o[i:j+1] if j - i < 4 else o[i:i+rng.randint(1, 3)]
                rest = o[:i] + o[i+len(seg):]
                k = rng.randrange(len(rest) + 1)
                if m < 0.75:
                    seg = seg[::-1]
                no = rest[:k] + seg + rest[k:]
            nc = cost(no)
            if nc < oc or rng.random() < math.exp((oc - nc) / T):
                o, oc = no, nc
                if oc < bc:
                    best, bc = o[:], oc
    return [int(x) for x in best]


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
