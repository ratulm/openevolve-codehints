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


def plan_tour(P):
    n = len(P)
    pts = [(float(p[0]), float(p[1])) for p in P]
    D = [[math.hypot(b[0] - a[0], b[1] - a[1]) for b in pts] for a in pts]
    H = [[math.atan2(b[1] - a[1], b[0] - a[0]) for b in pts] for a in pts]
    TWO_PI = 2 * math.pi

    def cost(t):
        c = 0.0
        for i in range(n):
            a, b, d = t[i - 2], t[i - 1], t[i]
            h = (H[b][d] - H[a][b] + math.pi) % TWO_PI - math.pi
            c += D[b][d] + 10.0 * h * h
        return c

    c0 = P.mean(axis=0)
    best = [int(i) for i in np.argsort(np.arctan2(P[:, 1] - c0[1], P[:, 0] - c0[0]))]
    bc = cost(best)
    rng = random.Random(12345)
    t0 = time.time()
    deadline = 20.0
    done = False
    for _ in range(100):
        t, c = best[:], bc
        iters = 40000
        for k in range(iters):
            if k % 2000 == 0 and time.time() - t0 > deadline:
                done = True
                break
            T = 20.0 * (0.005 / 20.0) ** (k / iters)
            i, j = sorted(rng.sample(range(n), 2))
            m = rng.random()
            if m < 0.45:  # 2-opt segment reversal
                s = t[:i] + t[i:j + 1][::-1] + t[j + 1:]
            elif m < 0.9:  # or-opt: move a segment, maybe reversed
                L = rng.randint(1, 8)
                seg = t[i:i + L]
                rest = t[:i] + t[i + L:]
                p = rng.randrange(len(rest) + 1)
                if rng.random() < 0.5:
                    seg = seg[::-1]
                s = rest[:p] + seg + rest[p:]
            else:  # swap two points
                s = t[:]
                s[i], s[j] = s[j], s[i]
            sc = cost(s)
            if sc < c or rng.random() < math.exp((c - sc) / T):
                t, c = s, sc
                if c < bc:
                    best, bc = t[:], c
        if done:
            break
    # final deterministic polish: exhaustive or-opt (segments 1-4, both orientations)
    improved = True
    while improved and time.time() - t0 < 26:
        improved = False
        for L in range(1, 5):
            for i in range(n - L + 1):
                seg = best[i:i + L]
                rest = best[:i] + best[i + L:]
                for p in range(len(rest) + 1):
                    for sg in (seg, seg[::-1]):
                        s = rest[:p] + sg + rest[p:]
                        sc = cost(s)
                        if sc < bc - 1e-9:
                            best, bc = s, sc
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
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
