# EVOLVE-BLOCK-START
# Goal: choose the order in which a single machine processes 36 jobs (indexed
# 0..35). The machine starts at time 0, handles one job at a time, and never
# idles or interrupts a job. Each job has a processing time, a due date and a
# weight (the three arrays defined below the evolved block).
#
# Hidden setup times: every job belongs to a family, family = job_index % 4.
# Whenever two consecutive jobs in the order belong to different families, a
# setup of 3 time units is inserted before the second one. There is no setup
# between two jobs of the same family and none before the first job. The setup
# rule is not present anywhere in this program's data; it exists only in the
# evaluator.
#
# Score: the evaluator computes each job's completion time as the sum of all
# processing times and setup times up to and including that job, then
#   cost = sum_j weight_j * max(0, completion_j - due_j)
# and returns 10608 / cost, where 10608 is the cost of a reference schedule.
# So 1.0 means matching the reference, and higher is possible. It is NOT the
# plain total weighted tardiness without setups: an order that is best when
# setups are ignored scores only about 0.70.
#
# Validity: the function must return a flat sequence of exactly 36 integers
# that is a permutation of 0..35 (each job exactly once). Anything else, an
# exception, or a run longer than 30 seconds scores 0.
import numpy as np


import random
import math
import time


def order_jobs(processing_times, due_dates, weights):
    # Setup-aware simulated annealing + insertion local search.
    p = [int(x) for x in processing_times]
    d = [int(x) for x in due_dates]
    w = [int(x) for x in weights]
    n = len(p)
    fam = [j % 4 for j in range(n)]

    def cost(o):
        t = 0
        c = 0
        prev = -1
        for j in o:
            if prev >= 0 and fam[j] != fam[prev]:
                t += 3
            t += p[j]
            prev = j
            if t > d[j]:
                c += w[j] * (t - d[j])
        return c

    rng = random.Random(12345)
    best = sorted(range(n), key=lambda j: d[j])
    bc = cost(best)
    t0 = time.time()
    for r in range(10):
        if r % 2 == 0:
            cur = best[:]
        else:
            cur = sorted(range(n), key=lambda j: d[j] - p[j] * rng.random() * 3)
        cc = cost(cur)
        T = 200.0
        for it in range(30000):
            if (it & 1023) == 0 and time.time() - t0 > 12:
                break
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b:
                continue
            new = cur[:]
            if rng.random() < 0.5:
                x = new.pop(a)
                new.insert(b, x)
            else:
                new[a], new[b] = new[b], new[a]
            nc = cost(new)
            if nc <= cc or rng.random() < math.exp((cc - nc) / T):
                cur, cc = new, nc
                if cc < bc:
                    best, bc = cur[:], cc
            T = max(0.5, T * 0.9996)
        if time.time() - t0 > 12:
            break

    # Final first-improvement insertion descent.
    improved = True
    while improved:
        improved = False
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                new = best[:]
                x = new.pop(a)
                new.insert(b, x)
                nc = cost(new)
                if nc < bc:
                    best, bc, improved = new, nc, True

    return np.array(best)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
# Job data, indexed by job number 0..35.
PROCESSING_TIMES = np.array([6, 11, 12, 5, 3, 8, 9, 10, 9, 9, 12, 12, 12, 11, 9, 12, 2, 2, 10, 6, 10, 7, 11, 2, 9, 2, 3, 11, 5,
                             12, 5, 10, 12, 5, 12, 9])
DUE_DATES = np.array([22, 170, 23, 40, 66, 110, 138, 13, 164, 135, 56, 33, 105, 105, 47, 29, 135, 119, 141, 38, 128, 13,
                      1, 31, 138, 95, 109, 104, 112, 57, 127, 2, 66, 9, 22, 160])
WEIGHTS = np.array([9, 3, 5, 7, 9, 3, 5, 8, 8, 9, 2, 9, 2, 8, 8, 9, 8, 7, 6, 5, 1, 9, 9, 1, 6, 4, 7, 6, 6, 5, 2, 8,
                    8, 2, 5, 1])


def run():
    # Returns the processing order as an array of job indices.
    return order_jobs(PROCESSING_TIMES, DUE_DATES, WEIGHTS)
