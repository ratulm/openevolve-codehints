"""Writes reference_output.pkl (best known for the true, position-weighted objective) and
standard_output.pkl (best heights for the standard objective: the unweighted maximum of the
autoconvolution over the squared sum, i.e. the first autocorrelation inequality with 50 steps).

The unweighted problem is mirror-symmetric, so its solution and the reversed vector are equally
good for the standard objective; the orientation that scores higher under the true rule is
stored, as the most favourable outcome for a solver that assumes the standard objective."""
import pickle
import numpy as np
import reference_solver as R
import task

best = min((R.solve(task.W, 60, seed=s) for s in range(4)), key=lambda t: t[1])
pickle.dump(best[0], open("reference_output.pkl", "wb"))

std, std_v = R.solve(np.ones(2 * R.N - 1), 240, seed=0)
if R.value(std[::-1], task.W) < R.value(std, task.W):
    std = std[::-1].copy()
pickle.dump(std, open("standard_output.pkl", "wb"))
print("reference V = %r | standard 2N*max(a*a)/sum^2 = %.6f" % (best[1], 2 * R.N * std_v))
print("true score of standard = %.4f, of its mirror image = %.4f"
      % (task.score(std)[0], task.score(std[::-1])[0]))
