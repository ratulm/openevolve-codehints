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

_TRIPLES = np.array(list(itertools.combinations(range(13), 3)))


def _score(p):
    a, b, c = p[_TRIPLES[:, 0]], p[_TRIPLES[:, 1]], p[_TRIPLES[:, 2]]
    ar = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 0.8
    s = np.partition(ar, (0, 142, 143))
    return s[0] ** 0.25 * ((s[142] + s[143]) / 2) ** 0.75


def _random_layout(rng, n=13):
    y = rng.uniform(0, 0.5, n)
    x = 0.6 * y + rng.uniform(0, 1, n) * (1 - 0.8 * y)
    return np.c_[x, y]


def place_points():
    num_points = 13
    fallback = np.zeros((num_points, 2))
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        fallback[i] = [0.52 + 0.2 * np.cos(angle), 0.25 + 0.12 * np.sin(angle)]
    fallback = clamp_into_trapezoid(fallback)

    rng = np.random.default_rng(12345)
    t0 = time.time()
    budget = 18.0
    best, best_s = fallback, _score(fallback)
    restarts = 6
    for r in range(restarts):
        if time.time() - t0 > budget:
            break
        cur = clamp_into_trapezoid(_random_layout(rng))
        cur_s = _score(cur)
        loc_best, loc_s = cur.copy(), cur_s
        end = t0 + budget * (r + 1) / restarts
        start = time.time()
        span = max(end - start, 1e-3)
        while True:
            now = time.time()
            if now > end:
                break
            frac = (now - start) / span
            step = 0.08 * (1 - frac) + 0.001
            temp = 0.002 * (1 - frac) ** 2 + 1e-7
            for _ in range(200):
                cand = cur.copy()
                k = rng.integers(num_points)
                cand[k] += rng.normal(0, step, 2)
                if rng.random() < 0.1:
                    k2 = rng.integers(num_points)
                    cand[k2] += rng.normal(0, step, 2)
                cand = clamp_into_trapezoid(cand)
                s = _score(cand)
                if s > cur_s or rng.random() < np.exp((s - cur_s) / temp):
                    cur, cur_s = cand, s
                    if s > loc_s:
                        loc_best, loc_s = cand.copy(), s
        if loc_s > best_s:
            best, best_s = loc_best, loc_s

    if not np.all(np.isfinite(best)) or best_s <= 0:
        return fallback
    return clamp_into_trapezoid(best)


def clamp_into_trapezoid(points):
    # Vectorised projection onto the trapezoid (bounds on y, then slanted x).
    p = np.array(points, dtype=float)
    y = np.clip(p[:, 1], 0.0, 0.5)
    x = np.clip(p[:, 0], 0.6 * y, 1.0 - 0.2 * y)
    return np.c_[x, y]


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    points = place_points()
    return points
