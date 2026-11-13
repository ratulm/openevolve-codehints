"""Writes reference_output.pkl (best known order for the true objective: total
weighted tardiness with family setup times) and standard_output.pkl (best order
for the standard objective: total weighted tardiness with no setup times).

The instance in task.py was drawn once as follows and then frozen as literals:
    rng = np.random.default_rng(9); n = 36
    proc = rng.integers(2, 13, n); weight = rng.integers(1, 10, n)
    due = rng.uniform(0.0, 0.6 * proc.sum(), n).astype(int)
"""
import pickle

import solver
import task

N = task.N
no_setup = [[0] * N for _ in range(N)]

ref_order, ref_cost, ref_all = solver.solve(task.PROC, task.DUE, task.WEIGHT, task.SETUP, 12, 400000, 5)
pickle.dump([int(j) for j in ref_order], open("reference_output.pkl", "wb"))

std_order, std_cost, std_all = solver.solve(task.PROC, task.DUE, task.WEIGHT, no_setup, 6, 200000, 5)
pickle.dump([int(j) for j in std_order], open("standard_output.pkl", "wb"))

print("reference true cost =", ref_cost, "per restart:", ref_all)
print("standard no-setup cost =", std_cost, "per restart:", std_all,
      "| its true cost =", task.true_cost(std_order))
