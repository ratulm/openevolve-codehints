# EVOLVE-BLOCK-START
# Goal: choose one point in the box [-4, 4]^3 for the function `landscape`
# defined in the fixed part below (a constant 13 minus five wells of different
# depths and widths, plus a small ripple, a weak bowl and a steep wall outside
# the box).
#
# Score: the point is NOT judged by landscape(point), and the lowest function
# value is NOT the target. The evaluator computes a robust value
#     R(point) = mean of landscape(point + offset) over 27 fixed offsets,
# where the offsets are the 3x3x3 grid {-0.5, 0, 0.5}^3 (the point itself is
# one of the 27). The score is R_ref / R(point) with R_ref = 6.2517, so 1.0
# means matching the reference and lower R is better (above 1.0 is possible).
# A wide, moderately low basin therefore scores better than a narrow, very
# deep well: the global minimum of `landscape` has R of about 8.6 and scores
# only about 0.73.
#
# Validity: run() must return exactly 3 finite numbers, each inside [-4, 4]
# (tolerance 1e-9). Anything else, an exception, or a run longer than
# 30 seconds scores 0.
import numpy as np


def choose_point():
    # Naive starting method: 12 uniformly random points in the box with a fixed
    # seed, keeping the one with the lowest plain function value.
    random_generator = np.random.default_rng(0)
    best_point = None
    best_value = np.inf

    for sample_index in range(12):
        candidate_point = random_generator.uniform(-4.0, 4.0, 3)
        candidate_value = landscape(candidate_point)

        # Keep the candidate if its single function value is the lowest so far.
        if candidate_value < best_value:
            best_point = candidate_point
            best_value = candidate_value

    return best_point


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
import numpy as np

# Centers, depths and widths of the five wells of the landscape.
well_centers = np.array([
    [1.8, -1.6, 1.2],
    [-2.0, 1.5, -1.0],
    [-1.85, 1.4, -0.9],
    [2.5, 2.4, -2.6],
    [-2.6, -2.4, 2.2],
])
well_depths = np.array([9.0, 6.5, 4.5, 6.0, 4.0])
well_widths = np.array([1.0, 0.8, 0.15, 0.3, 0.8])


def landscape(point):
    # The function of 3 variables whose neighbourhood average is scored.
    point = np.asarray(point, dtype=float)
    squared_distances = np.sum((point - well_centers) ** 2, axis=1)
    wells = np.sum(well_depths * np.exp(-squared_distances / (2.0 * well_widths ** 2)))
    ripple = (0.5 * np.sin(2.3 * point[0] + 1.1 * point[1]) * np.cos(1.7 * point[2] - 0.6 * point[0])
              + 0.3 * np.cos(2.9 * point[1] - 1.3 * point[2]))
    bowl = 0.02 * np.sum(point ** 2)
    # Steep penalty for leaving the box [-4, 4]^3.
    wall = 50.0 * np.sum(np.maximum(np.abs(point) - 4.0, 0.0))
    return 13.0 - wells + ripple + bowl + wall


def run():
    # Return the chosen point; the evaluator does the scoring.
    return choose_point()
