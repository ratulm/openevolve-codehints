# EVOLVE-BLOCK-START
# Goal: place 18 points in an annulus (the disc of radius 1.0 centred at the
# origin with a concentric hole of radius 0.4) so that the two closest points
# are as far apart as possible, where "far apart" is measured with a special
# distance, NOT the Euclidean distance.
#
# Distance between points (x1, y1) and (x2, y2):
#     sqrt((3 * (x1 - x2))**2 + (y1 - y2)**2) * (1 + 0.4 * (y1 + y2) / 2)
# Horizontal separation counts 3 times as much as vertical separation, and the
# whole distance is scaled by a factor that grows with the pair's mean height:
# 0.6 at the bottom of the disc (y = -1), 1.0 at y = 0, 1.4 at the top (y = 1).
#
# Score: the smallest of the 153 pairwise distances, divided by a reference
# value of about 0.9314, so 1.0 means matching the reference (higher is
# possible). It is NOT the minimum Euclidean distance, and NOT the ratio of
# minimum to maximum distance. Two coincident points give a score of 0.
#
# Validity: run() returns an array of shape (18, 2) with finite coordinates,
# and every point satisfies 0.4 <= sqrt(x**2 + y**2) <= 1.0 (tolerance 1e-9).
# Any violation, an exception, or a run longer than 30 seconds scores 0.
import numpy as np


import time

N = 18
_I, _J = np.triu_indices(N, 1)


def pair_dists(P):
    x, y = P[:, 0], P[:, 1]
    return np.sqrt((3 * (x[_I] - x[_J])) ** 2 + (y[_I] - y[_J]) ** 2) * (
        1 + 0.2 * (y[_I] + y[_J]))


def naive_points():
    a = 2.0 * np.pi * np.arange(N) / N
    return np.c_[0.55 * np.cos(a), 0.55 * np.sin(a)]


def optimise(P0, minimize):
    z0 = np.concatenate([P0.ravel(), [pair_dists(P0).min()]])
    cons = [
        dict(type='ineq', fun=lambda z: pair_dists(z[:-1].reshape(N, 2)) - z[-1]),
        dict(type='ineq', fun=lambda z: 1.0 - np.sum(z[:-1].reshape(N, 2) ** 2, 1)),
        dict(type='ineq', fun=lambda z: np.sum(z[:-1].reshape(N, 2) ** 2, 1) - 0.16),
    ]
    r = minimize(lambda z: -z[-1], z0, constraints=cons, method='SLSQP',
                 options=dict(maxiter=400, ftol=1e-12))
    return move_points_into_annulus(r.x[:-1].reshape(N, 2))


def build_points():
    best = move_points_into_annulus(naive_points())
    best_s = pair_dists(best).min()
    try:
        from scipy.optimize import minimize
    except Exception:
        return best
    rng = np.random.default_rng(12345)
    start = time.time()
    # deterministic two-ring start first, then random restarts
    a1 = 2 * np.pi * np.arange(8) / 8
    a2 = 2 * np.pi * (np.arange(10) + 0.5) / 10
    starts = [np.r_[np.c_[0.45 * np.cos(a1), 0.45 * np.sin(a1)],
                    np.c_[0.95 * np.cos(a2), 0.95 * np.sin(a2)]]]
    while time.time() - start < 12.0:
        if starts:
            P0 = starts.pop()
        else:
            a = rng.uniform(0, 2 * np.pi, N)
            r = np.sqrt(rng.uniform(0.16, 1.0, N))
            P0 = np.c_[r * np.cos(a), r * np.sin(a)]
        try:
            P = optimise(P0, minimize)
        except Exception:
            continue
        if not np.all(np.isfinite(P)):
            continue
        s = pair_dists(P).min()
        if s > best_s:
            best_s, best = s, P
    return best


def move_points_into_annulus(points):
    # Radially move any point that is in the hole or outside the disc onto the
    # nearest boundary circle of the annulus.
    moved_points = points.copy()

    for i in range(moved_points.shape[0]):
        distance_from_origin = np.hypot(moved_points[i, 0], moved_points[i, 1])
        if distance_from_origin < 1e-12:
            moved_points[i] = [0.4, 0.0]
        elif distance_from_origin < 0.4:
            moved_points[i] *= 0.4 / distance_from_origin
        elif distance_from_origin > 1.0:
            moved_points[i] *= 1.0 / distance_from_origin

    return moved_points


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    points = build_points()
    return points
