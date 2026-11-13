"""Writes reference_output.pkl (best known set for the true objective
|A+A|^2 / (|A-A| * |A|^1.5)) and standard_output.pkl (best set found for the
standard objective, the plain ratio |A+A| / |A-A|, on the same bounds).

Takes a few minutes; uses all cores. Deterministic (seeded)."""
import pickle
from multiprocessing import Pool

import solver as R

true_obj = lambda S, D, n: S ** 2 / (D * n ** 1.5)
std_obj = lambda S, D, n: S / D


def true_job(job):
    # The best sets found are symmetric (A = c - A, which forces |A+A| = |A-A|) and
    # nearly Sidon, so search symmetric sets, then polish without the restriction.
    c, seed = job
    if c is None:
        return R.anneal(true_obj, seed, steps=300000)
    a, _ = R.anneal(true_obj, seed, steps=300000, mirror=c)
    return R.anneal(true_obj, seed, steps=30000, t0=0.005, start=a)


def std_job(seed):
    return R.anneal(std_obj, seed, steps=400000, t0=0.02, t1=0.0002)


if __name__ == "__main__":
    jobs = [(c, s) for c in range(290, 300) for s in range(8)] + [(None, s) for s in range(8)]
    with Pool() as pool:
        ref = max(pool.map(true_job, jobs), key=lambda t: t[1])
        std = max(pool.map(std_job, range(16)), key=lambda t: t[1])
    pickle.dump([int(v) for v in ref[0]], open("reference_output.pkl", "wb"))
    pickle.dump([int(v) for v in std[0]], open("standard_output.pkl", "wb"))
    print("reference J =", repr(ref[1]), R.counts(ref[0]), len(ref[0]), ref[0])
    print("standard ratio =", std[1], R.counts(std[0]), len(std[0]), "true J =", true_obj(*R.counts(std[0]), len(std[0])))
