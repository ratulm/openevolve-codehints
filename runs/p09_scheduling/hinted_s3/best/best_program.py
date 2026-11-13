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
import time


def order_jobs(processing_times, due_dates, weights):
    P = [int(x) for x in processing_times]
    D = [int(x) for x in due_dates]
    W = [int(x) for x in weights]
    n = len(P)

    def cost(o):
        t = s = 0
        prev = -1
        for j in o:
            if prev >= 0 and prev % 4 != j % 4:
                t += 3
            t += P[j]
            if t > D[j]:
                s += W[j] * (t - D[j])
            prev = j
        return s

    def local_search(o):
        c = cost(o)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for k in range(n):
                    if i == k:
                        continue
                    m = o[:]
                    m.insert(k, m.pop(i))
                    v = cost(m)
                    if v < c:
                        o, c, improved = m, v, True
            for i in range(n):
                for k in range(i + 1, n):
                    m = o[:]
                    m[i], m[k] = m[k], m[i]
                    v = cost(m)
                    if v < c:
                        o, c, improved = m, v, True
        return o, c

    rng = random.Random(12345)
    start = time.time()
    best, best_c = local_search(sorted(range(n), key=lambda j: (D[j], j)))
    cur, cur_c = best, best_c
    # Iterated local search with random perturbations, time-bounded.
    while time.time() - start < 12:
        o = cur[:]
        for _ in range(rng.randint(2, 6)):
            i, k = rng.sample(range(n), 2)
            o.insert(k, o.pop(i))
        o, c = local_search(o)
        if c <= cur_c:
            cur, cur_c = o, c
            if c < best_c:
                best, best_c = o, c
        elif rng.random() < 0.05:
            cur, cur_c = best, best_c
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
