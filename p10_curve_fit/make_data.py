"""One-off generator: draws the seeded training set and the query list and writes
baseline/initial_program.py, hinted/initial_program.py and query_inputs.py.
The hidden law lives in task.py; it is used here only to draw the noisy data."""
import numpy as np
from task import true_law

rng = np.random.default_rng(20261)
n = 80
X = np.c_[rng.uniform(0.5, 4, n), rng.uniform(0, 5, n)].round(3)
y = (true_law(X[:, 0], X[:, 1]) + rng.normal(0, 0.08, n)).round(3)
qi = np.c_[rng.uniform(0.5, 4, 25), rng.uniform(0, 5, 25)]
qa = np.c_[rng.uniform(4.2, 7, 9), rng.uniform(0, 5, 9)]
qb = np.c_[rng.uniform(0.5, 4, 8), rng.uniform(5.3, 8.5, 8)]
qc = np.c_[rng.uniform(4.2, 7, 8), rng.uniform(5.3, 8.5, 8)]
Q = np.r_[qi, qa, qb, qc].round(3)
Q = Q[rng.permutation(len(Q))]  # fixed shuffle: in-range and out-of-range queries interleaved
SCORED = [i for i in range(len(Q)) if Q[i, 0] > 4 or Q[i, 1] > 5]
assert len(SCORED) == 25


def rows(A, per=5, ind="    "):
    it = ["[%.3f, %.3f]" % tuple(r) for r in A]
    return "\n".join(ind + ", ".join(it[i:i + per]) + "," for i in range(0, len(it), per))


def vec(v, per=10, ind="    "):
    it = ["%.3f" % t for t in v]
    return "\n".join(ind + ", ".join(it[i:i + per]) + "," for i in range(0, len(it), per))


BLIND = """# EVOLVE-BLOCK-START
import numpy as np


def g(u):
    return np.ones((u.shape[0], 1))


def f(u, v, w):
    k = np.linalg.lstsq(g(u), v, rcond=None)[0]
    return g(w) @ k


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
a = np.array([
%s
])
b = np.array([
%s
])
c = np.array([
%s
])


def run():
    return f(a, b, c)
"""

HINTED = """# EVOLVE-BLOCK-START
# Goal: predict an unknown smooth quantity y that depends on two real inputs
# (x1, x2), at inputs of which half lie OUTSIDE the region where it was measured.
#
# Data (defined in the fixed part below): TRAIN_INPUTS holds 80 measured points,
# one row (x1, x2) each, with x1 in [0.5, 4] and x2 in [0, 5]. TRAIN_OUTPUTS
# holds the 80 measured y values; each carries independent measurement noise of
# standard deviation about 0.08. QUERY_INPUTS holds 50 rows (x1, x2). run()
# must return the 50 predicted y values, in the same order as QUERY_INPUTS.
#
# Which queries count: 25 of the 50 queries lie inside the measured region and
# are NOT scored at all. The other 25 lie outside it (x1 in (4, 7] and/or x2 in
# (5, 8.5]) and are the only ones scored. The scored query indices are
#     %(scored)s
#
# Score: the evaluator knows the true noise-free y at every query. For each
# scored query i let d_i = prediction_i - true_y_i and
#     w_i = 8 if d_i < 0 (prediction too low), w_i = 1 otherwise.
#     E = sqrt( mean over the 25 scored queries of w_i * d_i**2 )
#     score = %(eref)s / E
# So predicting too low costs 8 times as much squared error as predicting too
# high by the same amount. %(eref)s is the E achieved by a reference predictor that
# was built from the 80 measured points alone, so 1.0 means matching the
# reference and higher is possible. For scale, the true y over the scored
# queries has standard deviation of about %(sd)s. The score is NOT a symmetric
# error, NOT the error over all 50 queries, and NOT the error on the measured
# points: how closely the 80 measured points are reproduced, and the
# predictions at the 25 unscored queries, do not enter the score.
#
# Validity: run() must return exactly 50 finite real numbers (a 1-D sequence of
# length 50). A wrong length or shape, NaN or infinite values, an exception, or
# a run longer than 30 seconds scores 0.
import numpy as np


def constant_design_matrix(inputs):
    # One column of ones per input row: the current model ignores x1 and x2.
    return np.ones((inputs.shape[0], 1))


def predict_queries(train_inputs, train_outputs, query_inputs):
    # Least-squares fit of a single constant to the measured outputs (this is
    # their mean), then the same constant is predicted for every query.
    coefficients = np.linalg.lstsq(constant_design_matrix(train_inputs), train_outputs, rcond=None)[0]
    return constant_design_matrix(query_inputs) @ coefficients


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
# 80 measured input points, one (x1, x2) pair each.
TRAIN_INPUTS = np.array([
%s
])
# The 80 noisy measured outputs y, in the same order as TRAIN_INPUTS.
TRAIN_OUTPUTS = np.array([
%s
])
# 50 query inputs (x1, x2). Only those outside the measured region are scored;
# their indices are listed in the comment at the top of this file.
QUERY_INPUTS = np.array([
%s
])


def run():
    return predict_queries(TRAIN_INPUTS, TRAIN_OUTPUTS, QUERY_INPUTS)
"""

import textwrap
import task
data = (rows(X), vec(y), rows(Q))
sc = textwrap.fill(", ".join(map(str, SCORED)), 66).replace("\n", "\n#     ")
info = dict(scored=sc, eref="%.4f" % task.E_REF, sd="%.1f" % np.std(true_law(Q[SCORED, 0], Q[SCORED, 1])))
hinted = HINTED.replace("%s", "@@").replace("%(", "@(").replace("%", "%%").replace("@(", "%(") % info
open("baseline/initial_program.py", "w").write(BLIND % data)
open("hinted/initial_program.py", "w").write(hinted.replace("%%", "%").replace("@@", "%s") % data)
open("query_inputs.py", "w").write(
    '"""Query inputs, identical to the list in both initial programs (written by make_data.py)."""\n'
    "QUERY_INPUTS = [\n%s\n]\n" % rows(Q))
