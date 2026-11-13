# EVOLVE-BLOCK-START
# Goal: place 13 points inside a trapezoid with corners (0, 0), (1, 0),
# (0.9, 0.5) and (0.3, 0.5), so that the triangles formed by triples of points
# are large. run() returns the points as an array of shape (13, 2).
#
# Score: the evaluator computes the areas of all 286 triangles formed by
# triples of the 13 points, divides them by the trapezoid's area (0.4), and
# takes
#     (smallest normalised area)**0.25 * (median normalised area)**0.75
# where the median over 286 areas is the mean of the 143rd and 144th smallest.
# This is divided by a reference value of about 0.107, so 1.0 means matching
# the reference (higher is possible). It is NOT the plain smallest triangle
# area (the usual Heilbronn objective): the median triangle carries three
# times the weight of the smallest one. But the smallest still matters: if any
# three points are collinear or two points coincide, the score is 0.
#
# Validity: exactly 13 points with finite coordinates, each inside or on the
# boundary of the trapezoid: 0 <= y <= 0.5, x >= 0.6*y and x <= 1 - 0.2*y
# (tolerance 1e-9). Any violation, an exception, or a run longer than
# 30 seconds scores 0.
import numpy as np


import itertools
import time

_TRI = np.array(list(itertools.combinations(range(13), 3)))


def _score(p):
    a, b, c = p[_TRI[:, 0]], p[_TRI[:, 1]], p[_TRI[:, 2]]
    A = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
               - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 0.8
    s = np.partition(A, (0, 142, 143))
    return s[0] ** 0.25 * (0.5 * (s[142] + s[143])) ** 0.75


def _random_layout(rng):
    y = rng.uniform(0, 0.5, 13)
    x = rng.uniform(0.6 * y, 1 - 0.2 * y)
    return np.c_[x, y]


def _anneal(p, rng, iters, deadline):
    cur = clamp_into_trapezoid(p)
    cs = _score(cur)
    best, bs = cur.copy(), cs
    T0, step0 = 0.02, 0.08
    for k in range(iters):
        if (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / iters
        T = T0 * (1 - frac) ** 2 + 1e-6
        step = step0 * (1 - frac) + 0.0005
        cand = cur.copy()
        i = rng.integers(13)
        cand[i] += rng.normal(0, step, 2)
        cand[i:i + 1] = clamp_into_trapezoid(cand[i:i + 1])
        s = _score(cand)
        if s >= cs or rng.random() < np.exp((s - cs) / (T * max(cs, 1e-9))):
            cur, cs = cand, s
            if s > bs:
                best, bs = cand.copy(), s
    return best, bs


def place_points():
    # Multi-start simulated annealing directly on the evaluation objective
    # (min**0.25 * median**0.75), with a hard wall-clock budget.
    rng = np.random.default_rng(12345)
    start = time.time()
    deadline = start + 21.0
    # Start from the ellipse layout so we always have a valid fallback.
    num_points = 13
    ang = 2 * np.pi * np.arange(num_points) / num_points
    ell = np.c_[0.52 + 0.2 * np.cos(ang), 0.25 + 0.12 * np.sin(ang)]
    best = clamp_into_trapezoid(ell)
    bs = _score(best)
    # Exploration phase: several short anneals from random starts.
    cands = []
    while time.time() < start + 10.0:
        p, s = _anneal(_random_layout(rng), rng, 6000, start + 10.0)
        cands.append((s, p))
        if s > bs:
            best, bs = p, s
    # Exploitation phase: long refinement of the best few.
    cands.sort(key=lambda t: -t[0])
    for s0, p0 in cands[:4]:
        if time.time() > deadline:
            break
        p, s = _anneal(p0, rng, 40000, deadline)
        if s > bs:
            best, bs = p, s
    # Polish phase: greedy hill-climb with adaptive tiny steps.
    step = 0.003
    fails = 0
    while time.time() < start + 25.0 and step > 1e-7:
        cand = best.copy()
        i = rng.integers(13)
        cand[i] += rng.normal(0, step, 2)
        cand = clamp_into_trapezoid(cand)
        s = _score(cand)
        if s > bs:
            best, bs = cand, s
            fails = 0
            step *= 1.2
        else:
            fails += 1
            if fails > 300:
                step *= 0.5
                fails = 0
    return clamp_into_trapezoid(best)


def clamp_into_trapezoid(points):
    # Move any point that lies outside the trapezoid onto its boundary.
    p = np.array(points, dtype=float)
    # Bottom/top sides, then left (x = 0.6y) and right (x = 1 - 0.2y) sides.
    p[:, 1] = np.clip(p[:, 1], 0.0, 0.5)
    p[:, 0] = np.clip(p[:, 0], 0.6 * p[:, 1], 1.0 - 0.2 * p[:, 1])
    return p


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    points = place_points()
    return points
