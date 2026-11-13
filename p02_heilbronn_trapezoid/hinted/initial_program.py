# EVOLVE-BLOCK-START
# Goal: place 13 points inside a trapezoid with corners (0, 0), (1, 0),
# (0.9, 0.5) and (0.3, 0.5), so that the triangles formed by triples of points
# are large. run() returns the points as an array of shape (13, 2).
#
# Score: the evaluator computes the areas of all 286 triangles formed by
# triples of the 13 points, divides them by the trapezoid's area (0.4), and
# takes
#     (smallest normalised area)**0.25 * (median normalised area)**0.75
# where the median over 286 areas is the mean of the 143rd and 144th smallest.
# This is divided by a reference value of about 0.107, so 1.0 means matching
# the reference (higher is possible). It is NOT the plain smallest triangle
# area (the usual Heilbronn objective): the median triangle carries three
# times the weight of the smallest one. But the smallest still matters: if any
# three points are collinear or two points coincide, the score is 0.
#
# Validity: exactly 13 points with finite coordinates, each inside or on the
# boundary of the trapezoid: 0 <= y <= 0.5, x >= 0.6*y and x <= 1 - 0.2*y
# (tolerance 1e-9). Any violation, an exception, or a run longer than
# 30 seconds scores 0.
import numpy as np


def place_points():
    # Naive starting layout: all points evenly spaced on one small ellipse in
    # the middle of the trapezoid, leaving most of the region unused.
    num_points = 13
    points = np.zeros((num_points, 2))

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        points[i] = [0.52 + 0.2 * np.cos(angle), 0.25 + 0.12 * np.sin(angle)]

    return clamp_into_trapezoid(points)


def clamp_into_trapezoid(points):
    # Move any point that lies outside the trapezoid onto its boundary.
    clamped_points = np.array(points, dtype=float)

    for i in range(clamped_points.shape[0]):
        # Bottom side y = 0 and top side y = 0.5.
        y = min(max(clamped_points[i, 1], 0.0), 0.5)
        # Left slanted side x = 0.6*y and right slanted side x = 1 - 0.2*y.
        x = min(max(clamped_points[i, 0], 0.6 * y), 1.0 - 0.2 * y)
        clamped_points[i] = [x, y]

    return clamped_points


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    points = place_points()
    return points
