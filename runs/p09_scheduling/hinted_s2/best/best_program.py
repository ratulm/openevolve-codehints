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


def order_jobs(processing_times, due_dates, weights):
    # Setup-aware simulated annealing on the true evaluator objective.
    n = len(processing_times)
    p = [int(x) for x in processing_times]
    d = [int(x) for x in due_dates]
    w = [int(x) for x in weights]

    def cost(o):
        t = 0
        c = 0
        prev = -1
        for j in o:
            if prev >= 0 and (prev & 3) != (j & 3):
                t += 3
            t += p[j]
            if t > d[j]:
                c += w[j] * (t - d[j])
            prev = j
        return c

    rng = random.Random(12345)
    best = sorted(range(n), key=lambda j: (d[j], j))
    bc = cost(best)

    for _ in range(4):
        cur = best[:]
        cc = bc
        T = 40.0
        for _ in range(40000):
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b:
                continue
            nw = cur[:]
            m = rng.random()
            if m < 0.4:
                nw[a], nw[b] = nw[b], nw[a]
            elif m < 0.85:
                nw.insert(b, nw.pop(a))
            else:
                if a > b:
                    a, b = b, a
                nw[a:b + 1] = nw[a:b + 1][::-1]
            nc = cost(nw)
            if nc <= cc or rng.random() < math.exp((cc - nc) / T):
                cur, cc = nw, nc
                if cc < bc:
                    best, bc = cur[:], cc
            T = max(0.3, T * 0.99985)

    # Final deterministic polish: best-insertion and adjacent swaps.
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                nw = best[:]
                nw.insert(k, nw.pop(i))
                nc = cost(nw)
                if nc < bc:
                    best, bc, improved = nw, nc, True
        for i in range(n - 1):
            nw = best[:]
            nw[i], nw[i + 1] = nw[i + 1], nw[i]
            nc = cost(nw)
            if nc < bc:
                best, bc, improved = nw, nc, True

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
