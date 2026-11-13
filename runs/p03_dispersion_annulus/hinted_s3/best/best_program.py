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


N = 18
_I, _J = np.triu_indices(N, 1)


def _dists(P):
    dx = P[_I, 0] - P[_J, 0]
    dy = P[_I, 1] - P[_J, 1]
    return np.sqrt(9 * dx * dx + dy * dy + 1e-18) * (1 + 0.2 * (P[_I, 1] + P[_J, 1]))


def _fallback():
    # Two rings: inner ring 6 points, outer ring 12 points.
    pts = []
    for i in range(6):
        a = 2 * np.pi * i / 6 + 0.3
        pts.append([0.45 * np.cos(a), 0.45 * np.sin(a)])
    for i in range(12):
        a = 2 * np.pi * i / 12
        pts.append([0.97 * np.cos(a), 0.97 * np.sin(a)])
    return move_points_into_annulus(np.array(pts))


def _solve(P0, minimize):
    z0 = np.concatenate([P0.ravel(), [_dists(P0).min()]])

    def rr(z):
        return (z[:-1].reshape(N, 2) ** 2).sum(1)

    cons = [
        {'type': 'ineq', 'fun': lambda z: _dists(z[:-1].reshape(N, 2)) - z[-1]},
        {'type': 'ineq', 'fun': lambda z: 1.0 - rr(z)},
        {'type': 'ineq', 'fun': lambda z: rr(z) - 0.16},
    ]
    r = minimize(lambda z: -z[-1], z0, constraints=cons, method='SLSQP',
                 options={'maxiter': 300, 'ftol': 1e-12})
    P = move_points_into_annulus(r.x[:-1].reshape(N, 2))
    return P, _dists(P).min()


def build_points():
    import time
    best = _fallback()
    best_v = _dists(best).min()
    try:
        from scipy.optimize import minimize
    except Exception:
        return best
    rng = np.random.default_rng(12345)
    t0 = time.time()
    while time.time() - t0 < 18.0:
        r = np.sqrt(rng.uniform(0.16, 1.0, N))
        a = rng.uniform(0, 2 * np.pi, N)
        try:
            P, v = _solve(np.c_[r * np.cos(a), r * np.sin(a)], minimize)
        except Exception:
            continue
        if np.all(np.isfinite(P)) and v > best_v:
            best, best_v = P, v
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
