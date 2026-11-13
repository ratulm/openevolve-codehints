# EVOLVE-BLOCK-START
# Goal: place 18 points in an annulus (the disc of radius 1.0 centred at the
# origin with a concentric hole of radius 0.4) so that the two closest points
# are as far apart as possible, where "far apart" is measured with a special
# distance, NOT the Euclidean distance.
#
# Distance between points (x1, y1) and (x2, y2):
#     sqrt((3 * (x1 - x2))**2 + (y1 - y2)**2) * (1 + 0.4 * (y1 + y2) / 2)
# Horizontal separation counts 3 times as much as vertical separation, and the
# whole distance is scaled by a factor that grows with the pair's mean height:
# 0.6 at the bottom of the disc (y = -1), 1.0 at y = 0, 1.4 at the top (y = 1).
#
# Score: the smallest of the 153 pairwise distances, divided by a reference
# value of about 0.9314, so 1.0 means matching the reference (higher is
# possible). It is NOT the minimum Euclidean distance, and NOT the ratio of
# minimum to maximum distance. Two coincident points give a score of 0.
#
# Validity: run() returns an array of shape (18, 2) with finite coordinates,
# and every point satisfies 0.4 <= sqrt(x**2 + y**2) <= 1.0 (tolerance 1e-9).
# Any violation, an exception, or a run longer than 30 seconds scores 0.
import numpy as np


def build_points():
    # Naive starting layout: all points equally spaced by angle on one circle
    # of radius 0.55 around the origin.
    num_points = 18
    points = np.zeros((num_points, 2))

    for i in range(num_points):
        angle = 2.0 * np.pi * i / num_points
        points[i] = [0.55 * np.cos(angle), 0.55 * np.sin(angle)]

    return move_points_into_annulus(points)


def move_points_into_annulus(points):
    # Radially move any point that is in the hole or outside the disc onto the
    # nearest boundary circle of the annulus.
    moved_points = points.copy()

    for i in range(moved_points.shape[0]):
        distance_from_origin = np.hypot(moved_points[i, 0], moved_points[i, 1])
        if distance_from_origin < 0.4:
            moved_points[i] *= 0.4 / distance_from_origin
        elif distance_from_origin > 1.0:
            moved_points[i] *= 1.0 / distance_from_origin

    return moved_points


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    points = build_points()
    return points
