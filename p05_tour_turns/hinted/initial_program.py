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


def plan_tour(point_coordinates):
    # Naive starting tour: visit the points in order of their compass direction
    # as seen from the centroid of all points, ignoring how far out each one is.
    centroid = point_coordinates.mean(axis=0)
    direction_from_centroid = np.arctan2(
        point_coordinates[:, 1] - centroid[1], point_coordinates[:, 0] - centroid[0]
    )
    visiting_order = np.argsort(direction_from_centroid)
    return visiting_order


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
