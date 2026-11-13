"""Writes reference_output.pkl (best known for the true objective) and
standard_output.pkl (best packing for the standard objective, plain sum of radii)."""
import pickle
import numpy as np
import reference_solver as R

N = R.N
pack = lambda z: (np.c_[z[:N], z[N:2 * N]], z[2 * N:])
true_obj = lambda x, y, r: np.sum(r * (0.2 + x) ** 2)
best = max((R.solve(true_obj, 40, seed=s) for s in range(4)), key=lambda t: t[1])
pickle.dump(pack(best[0]), open("reference_output.pkl", "wb"))
std = R.solve(R.OBJECTIVES["sum_r"], 40, seed=0)
pickle.dump(pack(std[0]), open("standard_output.pkl", "wb"))
print("reference J =", best[1], "| standard sum_r =", std[1])
