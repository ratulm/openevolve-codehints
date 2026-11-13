"""Writes reference_output.pkl (best point found for the true objective, the
27-point neighbourhood mean) and standard_output.pkl (the plain global minimum
of the landscape inside the box, which is what the blind program suggests).

Both use the same recipe: 4000 uniform random starts, bounded L-BFGS-B from the
60 best, then a Nelder-Mead polish of the winner."""
import pickle
import numpy as np
from scipy.optimize import minimize
import task

BOUNDS = [(task.LOW, task.HIGH)] * task.DIM


def solve(objective, seed):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(task.LOW, task.HIGH, (4000, task.DIM))
    values = np.array([objective(p) for p in starts])
    best_x, best_v = None, np.inf
    for p in starts[np.argsort(values)[:60]]:
        res = minimize(objective, p, method="L-BFGS-B", bounds=BOUNDS, options={"ftol": 1e-14, "gtol": 1e-10})
        if res.fun < best_v:
            best_x, best_v = res.x, res.fun
    res = minimize(objective, best_x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-13})
    x = np.clip(res.x, task.LOW, task.HIGH)
    if objective(x) < best_v:
        best_x, best_v = x, objective(x)
    return best_x, best_v


plain = lambda p: float(task.landscape(p)[0])
ref = min((solve(task.robust_value, s) for s in range(3)), key=lambda t: t[1])
std = min((solve(plain, s) for s in range(3)), key=lambda t: t[1])
pickle.dump(np.asarray(ref[0]), open("reference_output.pkl", "wb"))
pickle.dump(np.asarray(std[0]), open("standard_output.pkl", "wb"))
print("reference point", ref[0], "robust value R = %r" % ref[1], "plain g =", plain(ref[0]))
print("standard point ", std[0], "plain g =", std[1], "robust value R =", task.robust_value(std[0]))
