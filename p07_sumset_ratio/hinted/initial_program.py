# EVOLVE-BLOCK-START
# Goal: choose a set A of distinct integers. With A+A the set of all sums a+b
# and A-A the set of all differences a-b (a, b in A, a = b allowed), make the
# sumset large relative to the difference set while keeping A small.
#
# Score: the evaluator computes
#     |A+A|**2 / (|A-A| * |A|**1.5)
# and divides it by a reference value of about 2.5, so 1.0 means matching the
# reference (higher is possible). It is NOT the plain ratio |A+A| / |A-A| and
# NOT its logarithmic form: the sumset size is squared and the score is divided
# by the number of elements to the power 1.5, so a set with a good plain ratio
# but many elements scores poorly.
#
# Validity: run() must return a list of between 1 and 60 integers, all
# distinct, each in the range 0..299 (there is no tolerance; values are exact
# integers). Any violation, an exception, or a run longer than 30 seconds
# scores 0.
import numpy as np


def f():
    upper_limit = 300  # every element must be below this
    max_set_size = 60  # at most this many elements
    num_elements = 40
    common_difference = 7

    # Naive starting set: an arithmetic progression 0, 7, 14, ..., 273.
    candidate_set = np.arange(num_elements) * common_difference

    # Enforce the validity rules: distinct values, inside the range, not too many.
    candidate_set = np.unique(candidate_set)
    candidate_set = candidate_set[(candidate_set >= 0) & (candidate_set < upper_limit)]
    candidate_set = candidate_set[:max_set_size]

    return candidate_set


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return [int(v) for v in a]
