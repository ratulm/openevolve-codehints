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
    a = p[_TRIPLES[:, 0]]
    b = p[_TRIPLES[:, 1]]
    c = p[_TRIPLES[:, 2]]
    ar = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 0.8
    s = np.partition(ar, (0, 142, 143))
    if s[0] <= 0:
        return 0.0
    return s[0] ** 0.25 * ((s[142] + s[143]) / 2) ** 0.75


def _clamp(p):
    p = p.copy()
    eps = 1e-12
    p[:, 1] = np.clip(p[:, 1], eps, 0.5 - eps)
    p[:, 0] = np.clip(p[:, 0], 0.6 * p[:, 1] + eps, 1 - 0.2 * p[:, 1] - eps)
    return p


def _ellipse():
    ang = 2 * np.pi * np.arange(13) / 13
    return _clamp(np.c_[0.52 + 0.2 * np.cos(ang), 0.25 + 0.12 * np.sin(ang)])


def _structured():
    pts = [[0, 0], [1, 0], [0.9, 0.5], [0.3, 0.5],      # corners
           [0.5, 0], [0.6, 0.5], [0.15, 0.25], [0.95, 0.25],  # edge mids
           [0.28, 0.12], [0.75, 0.12], [0.45, 0.36], [0.78, 0.38],
           [0.55, 0.22]]
    return _clamp(np.array(pts, dtype=float))


def _random_start(rng):
    y = rng.uniform(0, 0.5, 13)
    x = 0.6 * y + rng.uniform(0, 1, 13) * (1 - 0.8 * y)
    return _clamp(np.c_[x, y])


def _boundary_start(rng):
    # Points spread around the trapezoid perimeter (plus jitter), a good
    # prior for Heilbronn-type layouts.
    corners = np.array([[0, 0], [1, 0], [0.9, 0.5], [0.3, 0.5]], dtype=float)
    seg = np.r_[1.0, np.hypot(0.1, 0.5), 0.6, np.hypot(0.3, 0.5)]
    cum = np.r_[0, np.cumsum(seg)]
    t = (np.sort(rng.uniform(0, cum[-1], 13)))
    pts = []
    for tt in t:
        k = min(np.searchsorted(cum, tt, side="right") - 1, 3)
        f = (tt - cum[k]) / seg[k]
        pts.append(corners[k] + f * (corners[(k + 1) % 4] - corners[k]))
    pts = np.array(pts)
    # pull a few points inward
    c = np.array([0.55, 0.25])
    idx = rng.choice(13, 3, replace=False)
    pts[idx] = c + rng.uniform(0.2, 0.6, (3, 1)) * (pts[idx] - c)
    return _clamp(pts + rng.normal(0, 0.01, pts.shape))


def _polish(p, s, rng, t_end):
    # Greedy fine-scale hill climbing with an adaptive step size.
    step = 0.01
    while time.time() < t_end and step > 2e-5:
        improved = 0
        for _ in range(300):
            cand = p.copy()
            i = rng.integers(13)
            d = rng.normal(0, step, 2)
            if rng.random() < 0.3:
                d[rng.integers(2)] = 0.0
            cand[i] += d
            cand = _clamp(cand)
            cs = _score(cand)
            if cs > s:
                p, s = cand, cs
                improved += 1
        if improved == 0:
            step *= 0.6
        elif improved > 15:
            step *= 1.3
    return p, s


def _anneal(p, rng, t_end):
    cur = p.copy()
    cs = _score(cur)
    best, bs = cur.copy(), cs
    t0 = time.time()
    total = max(t_end - t0, 1e-3)
    while True:
        now = time.time()
        if now >= t_end:
            break
        frac = (now - t0) / total
        step = 0.08 * (1 - frac) + 0.002
        temp = 0.01 * (1 - frac) ** 2 + 1e-6
        for _ in range(200):
            cand = cur.copy()
            i = rng.integers(13)
            cand[i] += rng.normal(0, step, 2)
            cand = _clamp(cand)
            s = _score(cand)
            if s >= cs or rng.random() < np.exp((s - cs) / (temp * max(cs, 1e-9))):
                cur, cs = cand, s
                if s > bs:
                    best, bs = cand.copy(), s
    return best, bs


def place_points():
    rng = np.random.default_rng(12345)
    t_start = time.time()
    deadline = t_start + 21.0
    starts = ([_structured(), _ellipse(), _boundary_start(rng)]
              + [_random_start(rng) for _ in range(2)])
    best, bs = _ellipse(), _score(_ellipse())
    per = (deadline - time.time()) / (len(starts) + 2)
    for st in starts:
        p, s = _anneal(st, rng, min(time.time() + per, deadline))
        if s > bs:
            best, bs = p, s
    # Re-anneal from the best layout found so far.
    while time.time() < deadline - 0.1:
        p, s = _anneal(best, rng, min(time.time() + per, deadline))
        if s > bs:
            best, bs = p, s
    best, bs = _polish(best, bs, rng, t_start + 24.0)
    if not np.all(np.isfinite(best)) or bs <= 0:
        best = _ellipse()
    return clamp_into_trapezoid(best)


def clamp_into_trapezoid(points):
    # Move any point that lies outside the trapezoid onto its boundary.
    clamped_points = np.array(points, dtype=float)

    for i in range(clamped_points.shape[0]):
        # Bottom side y = 0 and top side y = 0.5.
        y = min(max(clamped_points[i, 1], 0.0), 0.5)
        # Left slanted side x = 0.6*y and right slanted side x = 1 - 0.2*y.
        x = min(max(clamped_points[i, 0], 0.6 * y), 1.0 - 0.2 * y)
        clamped_points[i] = [x, y]

    return clamped_points


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    points = place_points()
    return points
