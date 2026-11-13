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
_IU = np.triu_indices(N, 1)


def pair_dists(p):
    x, y = p[:, 0], p[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(9.0 * dx ** 2 + dy ** 2) * (1.0 + 0.2 * (y[:, None] + y[None, :]))
    return d[_IU]


def ring_start(rng=None, jitter=0.0):
    # Two rings: 11 points on the outer circle, 7 on an inner circle.
    pts = []
    for k in range(11):
        a = 2 * np.pi * k / 11 + 0.3
        pts.append([0.97 * np.cos(a), 0.97 * np.sin(a)])
    for k in range(7):
        a = 2 * np.pi * k / 7
        pts.append([0.5 * np.cos(a), 0.5 * np.sin(a)])
    p = np.array(pts)
    if rng is not None and jitter > 0:
        p += rng.normal(0, jitter, p.shape)
    return move_points_into_annulus(p)


_I, _J = _IU
_M = len(_I)
_R = np.arange(_M)
_K = np.arange(N)
_GOBJ = np.zeros(2 * N + 1)
_GOBJ[-1] = -1.0


def _dist_jac(z):
    p = z[:-1].reshape(N, 2)
    dx = p[_I, 0] - p[_J, 0]
    dy = p[_I, 1] - p[_J, 1]
    s = np.sqrt(9.0 * dx * dx + dy * dy + 1e-18)
    f = 1.0 + 0.2 * (p[_I, 1] + p[_J, 1])
    gx = 9.0 * dx * f / s
    gy = dy * f / s
    G = np.zeros((_M, 2 * N + 1))
    G[_R, 2 * _I] = gx
    G[_R, 2 * _J] = -gx
    G[_R, 2 * _I + 1] = gy + 0.2 * s
    G[_R, 2 * _J + 1] = -gy + 0.2 * s
    G[:, -1] = -1.0
    return G


def _ring_fun(z):
    q = (z[:-1].reshape(N, 2) ** 2).sum(1)
    return np.r_[1.0 - q, q - 0.16]


def _ring_jac(z):
    p = z[:-1].reshape(N, 2)
    G = np.zeros((2 * N, 2 * N + 1))
    G[_K, 2 * _K] = -2 * p[:, 0]
    G[_K, 2 * _K + 1] = -2 * p[:, 1]
    G[N + _K, 2 * _K] = 2 * p[:, 0]
    G[N + _K, 2 * _K + 1] = 2 * p[:, 1]
    return G


def slsqp_solve(p0, maxiter=300):
    from scipy.optimize import minimize
    cons = [
        dict(type='ineq', fun=lambda z: pair_dists(z[:-1].reshape(N, 2)) - z[-1],
             jac=_dist_jac),
        dict(type='ineq', fun=_ring_fun, jac=_ring_jac),
    ]
    z0 = np.append(p0.ravel(), pair_dists(p0).min())
    r = minimize(lambda z: -z[-1], z0, jac=lambda z: _GOBJ, method='SLSQP',
                 constraints=cons, options=dict(maxiter=maxiter, ftol=1e-13))
    return move_points_into_annulus(r.x[:-1].reshape(N, 2).copy())


def build_points():
    best = ring_start()
    best_s = pair_dists(best).min()
    try:
        rng = np.random.default_rng(12345)
        t0 = time.time()
        it = 0
        while time.time() - t0 < 12.0:
            if it % 5 == 0:
                p0 = ring_start(rng, rng.uniform(0.02, 0.08))
            elif it % 5 == 1 and it > 10:
                p0 = move_points_into_annulus(best + rng.normal(0, 0.03, best.shape))
            else:
                a = rng.uniform(0, 2 * np.pi, N)
                r = np.sqrt(rng.uniform(0.16, 1.0, N))
                p0 = np.c_[r * np.cos(a), r * np.sin(a)]
            it += 1
            try:
                p = slsqp_solve(p0)
            except Exception:
                continue
            if not np.all(np.isfinite(p)):
                continue
            s = pair_dists(p).min()
            if s > best_s:
                best, best_s = p, s
        # final polish of the best configuration
        try:
            p = slsqp_solve(best, maxiter=1000)
            if np.all(np.isfinite(p)) and pair_dists(p).min() > best_s:
                best = p
        except Exception:
            pass
    except Exception:
        pass
    return move_points_into_annulus(best)


def move_points_into_annulus(points):
    # Radially move any point that is in the hole or outside the disc onto the
    # nearest boundary circle of the annulus.
    moved_points = points.copy()

    for i in range(moved_points.shape[0]):
        distance_from_origin = np.hypot(moved_points[i, 0], moved_points[i, 1])
        if distance_from_origin < 1e-12:
            moved_points[i] = [0.0, 0.4 + 1e-12]
        elif distance_from_origin < 0.4:
            moved_points[i] *= (0.4 + 1e-12) / distance_from_origin
        elif distance_from_origin > 1.0:
            moved_points[i] *= (1.0 - 1e-12) / distance_from_origin

    return moved_points


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    points = build_points()
    return points
