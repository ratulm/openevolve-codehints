"""Writes reference_output.pkl (best known for the true anisotropic metric) and
standard_output.pkl (best max-min Euclidean dispersion on the same annulus)."""
import pickle
import numpy as np
import reference_solver as R

runs = [R.solve(R.true_dist, 250, seed=s) for s in range(4)]
print("true-metric best per seed:", [round(v, 6) for _, v in runs])
best = max(runs, key=lambda t: t[1])
pickle.dump(best[0], open("reference_output.pkl", "wb"))
std = max((R.solve(R.euclid_dist, 150, seed=s) for s in range(2)), key=lambda t: t[1])
pickle.dump(std[0], open("standard_output.pkl", "wb"))
print("reference D_REF =", repr(float(R.true_dist(best[0]).min())), "| standard euclid =", std[1],
      "| standard under true metric =", R.true_dist(std[0]).min())
