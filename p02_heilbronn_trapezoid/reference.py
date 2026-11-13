"""Writes reference_output.pkl (best known for the true objective
min^0.25 * median^0.75) and standard_output.pkl (best configuration found for the
standard objective, the plain smallest triangle area). Takes several minutes on 8 cores."""
import pickle
import sys
from multiprocessing import Pool

import reference_solver as R

A_TRUE = 0.25
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 16

if __name__ == "__main__":
    jobs = [(a, s, 6, 60000, 40) for a in (A_TRUE, 1.0) for s in range(SEEDS)]
    with Pool(8) as pool:
        res = pool.map(R.job, jobs, chunksize=1)
    true_runs, std_runs = res[:SEEDS], res[SEEDS:]
    print("true objective per seed:", sorted(round(v, 6) for _, v in true_runs))
    print("min area per seed:      ", sorted(round(v, 6) for _, v in std_runs))
    ref = max(true_runs, key=lambda t: t[1])
    std = max(std_runs, key=lambda t: t[1])
    # Second stage: longer basin hopping from the best of each, 8 independent seeds.
    jobs = [(q[0], a, s, 150) for q, a in ((ref, A_TRUE), (std, 1.0)) for s in range(8)]
    with Pool(8) as pool:
        res = pool.map(R.job2, jobs, chunksize=1)
    print("stage 2 true:", sorted(round(v, 6) for _, v in res[:8]))
    print("stage 2 min: ", sorted(round(v, 6) for _, v in res[8:]))
    ref = max(res[:8], key=lambda t: t[1])
    std = max(res[8:], key=lambda t: t[1])
    pickle.dump(ref[0], open("reference_output.pkl", "wb"))
    pickle.dump(std[0], open("standard_output.pkl", "wb"))
    print("reference J = %.16g" % (ref[1] / R.AREA), "stats", R.stats(ref[0]))
    print("standard min area =", std[1], "stats", R.stats(std[0]),
          "| true-objective ratio = %.4f" % (R.value(std[0], A_TRUE) / ref[1]))
