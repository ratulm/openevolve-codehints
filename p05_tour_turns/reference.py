"""Writes reference_output.pkl (best known tour for the true cost, length plus
10 * sum of squared turning angles) and standard_output.pkl (best tour found for
the standard objective, plain tour length). Takes about four minutes on 8 cores.
Prints the reference cost; task.COST_REF must equal it."""
import pickle
from multiprocessing import Pool

import reference_solver as R
import task


def true_job(seed):
    return R.anneal(task.POINTS, task.TURN_WEIGHT, 2, seed, steps=1000000, t0=15.0)


def length_job(seed):
    return R.anneal(task.POINTS, 0.0, 2, seed, steps=300000, t0=10.0)


if __name__ == "__main__":
    with Pool(8) as pool:
        true_runs = pool.map(true_job, range(24))
        length_runs = pool.map(length_job, range(16))
    best = min(true_runs, key=lambda r: r[1])
    std = min(length_runs, key=lambda r: r[1])
    pickle.dump([int(i) for i in best[0]], open("reference_output.pkl", "wb"))
    pickle.dump([int(i) for i in std[0]], open("standard_output.pkl", "wb"))
    print("true costs of all runs:", sorted(round(r[1], 2) for r in true_runs))
    print("lengths of all standard runs:", sorted(round(r[1], 2) for r in length_runs))
    print("reference cost = %r (check)= %r" % (best[1], task.tour_cost(best[0])))
    print("standard length = %r, its true cost = %r, score = %.4f"
          % (std[1], task.tour_cost(std[0]), task.tour_cost(best[0]) / task.tour_cost(std[0])))
