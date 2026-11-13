# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_T = np.array(list(combinations(range(13), 3)))


def _sa(p):
    a, b, c = p[_T[:, 0]], p[_T[:, 1]], p[_T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _m(p):
    return np.abs(_sa(p)).min()


def _pol(p):
    # SLSQP max-min polish: maximize t s.t. sign_k * area_k >= t, points in region
    try:
        from scipy.optimize import minimize
    except Exception:
        return p
    for _ in range(4):
        sg = np.sign(_sa(p))
        sg[sg == 0] = 1
        cons = [{'type': 'ineq',
                 'fun': lambda z: sg * _sa(z[:-1].reshape(-1, 2)) - z[-1]},
                {'type': 'ineq',
                 'fun': lambda z: np.r_[z[1:-1:2], 0.5 - z[1:-1:2],
                                        z[0:-1:2] - 0.6 * z[1:-1:2],
                                        1 - 0.2 * z[1:-1:2] - z[0:-1:2]]}]
        try:
            r = minimize(lambda z: -z[-1], np.r_[p.ravel(), _m(p)],
                         jac=lambda z: np.r_[np.zeros(z.size - 1), -1.0],
                         method='SLSQP', constraints=cons,
                         options={'maxiter': 500, 'ftol': 1e-15})
            if not np.all(np.isfinite(r.x)):
                break
            q = g(r.x[:-1].reshape(-1, 2))
        except Exception:
            break
        if _m(q) > _m(p):
            p = q
        else:
            break
    return p


def f():
    n = 13
    t = 2 * np.pi * np.arange(n) / n
    p0 = g(np.stack([0.52 + 0.2 * np.cos(t), 0.25 + 0.12 * np.sin(t)], 1))
    rng = np.random.default_rng(7)
    best, bs = p0, _m(p0)
    K = 25000
    for r in range(6):
        x = p0.copy() if r == 0 else g(rng.random((n, 2)) * [1.0, 0.5])
        s = _m(x)
        for k in range(K):
            st = 0.05 * (1 - k / K) ** 2 + 3e-4
            mk = rng.random((n, 1)) < 0.3
            y = g(x + rng.normal(0, st, x.shape) * mk)
            v = _m(y)
            if v >= s:
                x, s = y, v
        x = _pol(x)
        s = _m(x)
        if s > bs:
            best, bs = x, s
    # many cheap diversified starts: short hill-climb + SLSQP polish
    K2 = 3000
    for r in range(40):
        x = g(rng.random((n, 2)) * [1.0, 0.5])
        s = _m(x)
        for k in range(K2):
            st = 0.06 * (1 - k / K2) ** 2 + 5e-4
            mk = rng.random((n, 1)) < 0.3
            y = g(x + rng.normal(0, st, x.shape) * mk)
            v = _m(y)
            if v >= s:
                x, s = y, v
        x = _pol(x)
        s = _m(x)
        if s > bs:
            best, bs = x, s
    for k in range(20000):
        y = best.copy()
        y[rng.integers(n)] += rng.normal(0, 2e-4, 2)
        y = g(y)
        v = _m(y)
        if v >= bs:
            best, bs = y, v
    return _pol(best)


def g(p):
    q = np.array(p, dtype=float)
    v = np.clip(q[:, 1], 0.0, 0.5)
    u = np.minimum(np.maximum(q[:, 0], 0.6 * v), 1.0 - 0.2 * v)
    return np.stack([u, v], 1)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
