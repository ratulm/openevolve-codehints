# EVOLVE-BLOCK-START
# Goal: choose the 50 non-negative heights a_0..a_49 of a step function (50
# equal-width steps on an interval) so that a position-weighted peak of its
# autoconvolution is as small as possible relative to its squared total.
#
# Score: the evaluator computes the discrete autoconvolution
#   c_k = sum_i a_i * a_(k-i)   for k = 0..98   (same as np.convolve(a, a)),
# the weights w_k = 1 + 3*k/98 (rising linearly from 1 at k = 0 to 4 at
# k = 98), and the value
#   V = max_k (w_k * c_k) / (sum_i a_i)**2.
# The score is 0.0287461148884 / V, so lower V is better, 1.0 means matching
# the reference and higher is possible. The weight is asymmetric: the same
# autoconvolution value costs four times more at the largest shift than at
# shift 0. It is NOT the plain unweighted maximum of the autoconvolution, so
# reversing the vector changes the score. V does not change when all heights
# are multiplied by the same positive number.
#
# Validity: run() must return a one-dimensional vector of exactly 50 finite
# numbers, every one >= 0 (no tolerance: any negative value is invalid), and
# not all zero. Any violation, an exception, or a run longer than 30 seconds
# scores 0.
import numpy as np


def f():
    # Naive starting heights: a straight ramp rising from 1 on the first step
    # to 50 on the last step.
    num_steps = 50
    step_heights = np.zeros(num_steps)

    for step_index in range(num_steps):
        step_heights[step_index] = 1.0 + step_index

    step_heights = make_valid_and_normalise(step_heights)

    return step_heights


def make_valid_and_normalise(step_heights):
    # Clip negative heights to zero so the vector is valid.
    step_heights = np.maximum(step_heights, 0.0)
    total_height = np.sum(step_heights)

    # An all-zero vector is invalid; fall back to constant heights.
    if total_height <= 0.0:
        step_heights = np.ones(step_heights.shape[0])
        total_height = np.sum(step_heights)

    # Rescale so the heights sum to 1 (the score is unaffected by scale).
    return step_heights / total_height


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
