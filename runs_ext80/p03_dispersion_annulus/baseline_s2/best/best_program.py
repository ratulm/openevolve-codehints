# EVOLVE-BLOCK-START
import numpy as np


def f():
    # Two staggered rings: 12 on the outer boundary (axis-aligned),
    # 6 inside, offset by 15 deg. r_in = 2*sin(15deg) makes the outer,
    # inner and cross-ring neighbour distances all equal.
    a = 2.0 * np.pi * np.arange(12) / 12
    b = 2.0 * np.pi * np.arange(6) / 6 + np.pi / 12
    r_in = 2.0 * np.sin(np.pi / 12)
    p = g(np.vstack([np.c_[np.cos(a), np.sin(a)],
                     r_in * np.c_[np.cos(b), np.sin(b)]]))
    try:
        return _refine(p)
    except Exception:
        return p


def _md(p):
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    np.fill_diagonal(d, 9.0)
    return d.min()


def _refine(p0):
    # Max-min-distance SLSQP: maximise t s.t. |pi-pj|^2 >= t^2, 0.16<=|pi|^2<=1
    from scipy.optimize import minimize
    n = len(p0)
    iu = np.triu_indices(n, 1)

    def cons(x):
        q = x[:-1].reshape(n, 2)
        d = np.sum((q[:, None] - q[None]) ** 2, -1)[iu] - x[-1] ** 2
        r = np.sum(q * q, 1)
        return np.r_[d, 1 - r, r - 0.16]

    rng = np.random.default_rng(0)
    starts = [p0] + [p0 + rng.normal(0, 0.03, p0.shape) for _ in range(11)]
    for _ in range(30):
        r = np.sqrt(rng.uniform(0.16, 1.0, n))
        t = rng.uniform(0, 2 * np.pi, n)
        starts.append(np.c_[r * np.cos(t), r * np.sin(t)])

    best, bm = p0, _md(p0)
    for s in starts:
        x0 = np.r_[s.ravel(), _md(s)]
        res = minimize(lambda x: -x[-1], x0, method='SLSQP',
                       constraints=dict(type='ineq', fun=cons),
                       options=dict(maxiter=500, ftol=1e-12))
        q = g(res.x[:-1].reshape(n, 2))
        m = _md(q)
        if m > bm:
            best, bm = q, m
    return best


def g(p):
    q = p.copy()

    for i in range(q.shape[0]):
        d = np.hypot(q[i, 0], q[i, 1])
        if d < 0.4:
            q[i] *= 0.4 / d
        elif d > 1.0:
            q[i] *= 1.0 / d

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
