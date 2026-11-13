# EVOLVE-BLOCK-START
import numpy as np


def f():
    import itertools
    import time
    n = 13
    D = [1e-4]
    p0 = np.zeros((n, 2))
    for i in range(n):
        t = 2 * np.pi * i / n + 0.3
        p0[i] = [0.55 + 0.33 * np.cos(t), 0.25 + 0.24 * np.sin(t)]
    p0 = g(p0)
    try:
        from scipy.optimize import minimize
    except Exception:
        return p0

    T = np.array(list(itertools.combinations(range(n), 3)))

    def areas(p):
        a, b, c = p[T[:, 0]], p[T[:, 1]], p[T[:, 2]]
        return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

    def cons(z):
        d = D[0]
        p = z[:-1].reshape(n, 2)
        x, y = p[:, 0], p[:, 1]
        return np.concatenate([areas(p) - z[-1], y - d, 0.5 - d - y,
                               x - 0.6 * y - d, 1.0 - 0.2 * y - x - d])

    m = len(T)
    R = np.arange(m)
    LB = np.zeros((4 * n, 2 * n + 1))
    for i in range(n):
        LB[i, 2 * i + 1] = 1
        LB[n + i, 2 * i + 1] = -1
        LB[2 * n + i, 2 * i] = 1
        LB[2 * n + i, 2 * i + 1] = -0.6
        LB[3 * n + i, 2 * i] = -1
        LB[3 * n + i, 2 * i + 1] = -0.2

    def jac(z):
        p = z[:-1].reshape(n, 2)
        a, b, c = p[T[:, 0]], p[T[:, 1]], p[T[:, 2]]
        sa = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
              - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        sg = 0.5 * np.where(sa >= 0, 1.0, -1.0)
        J = np.zeros((m, 2 * n + 1))
        i0, i1, i2 = 2 * T[:, 0], 2 * T[:, 1], 2 * T[:, 2]
        J[R, i0] += sg * (b[:, 1] - c[:, 1])
        J[R, i0 + 1] += sg * (c[:, 0] - b[:, 0])
        J[R, i1] += sg * (c[:, 1] - a[:, 1])
        J[R, i1 + 1] += sg * (a[:, 0] - c[:, 0])
        J[R, i2] += sg * (a[:, 1] - b[:, 1])
        J[R, i2 + 1] += sg * (b[:, 0] - a[:, 0])
        J[:, -1] = -1
        return np.vstack([J, LB])

    obj = lambda z: -z[-1]
    ojac = lambda z: np.r_[np.zeros(2 * n), -1.0]

    rng = np.random.default_rng(12345)
    best, bp = areas(p0).min(), p0
    pool = []
    t0 = time.time()
    k = 0
    while time.time() - t0 < 25 and k < 400:
        if k == 0:
            p = p0.copy()
        elif k % 4 != 0:
            p = g(bp + rng.normal(0, [0.05, 0.02, 0.006][k % 3], (n, 2)))
        else:
            y = rng.random(n) * 0.5
            x = 0.6 * y + rng.random(n) * (1.0 - 0.8 * y)
            p = g(np.column_stack([x, y]))
        k += 1
        z = np.append(p.ravel(), areas(p).min())
        try:
            r = minimize(obj, z, jac=ojac, method='SLSQP',
                         constraints={'type': 'ineq', 'fun': cons, 'jac': jac},
                         options={'maxiter': 300, 'ftol': 1e-12})
        except Exception:
            continue
        if not np.all(np.isfinite(r.x)):
            continue
        q = g(r.x[:-1].reshape(n, 2))
        v = areas(q).min()
        if not any(abs(v - c[0]) < 1e-9 for c in pool):
            pool.append((v, q))
            pool.sort(key=lambda c: -c[0])
            del pool[6:]
        if v > best:
            best, bp = v, q
    gbest, gbp = best, bp
    for _, cp in [(best, bp)] + pool:
        cb = areas(cp).min()
        for dd, it, tol in [(1e-4, 1000, 1e-15), (1e-6, 1000, 1e-16),
                            (1e-8, 1000, 1e-16), (0.0, 1000, 1e-16)]:
            D[0] = dd
            try:
                z = np.append(cp.ravel(), cb)
                r = minimize(obj, z, jac=ojac, method='SLSQP',
                             constraints={'type': 'ineq', 'fun': cons, 'jac': jac},
                             options={'maxiter': it, 'ftol': tol})
                if np.all(np.isfinite(r.x)):
                    q = g(r.x[:-1].reshape(n, 2))
                    v = areas(q).min()
                    if v > cb:
                        cb, cp = v, q
            except Exception:
                pass
        if cb > gbest:
            gbest, gbp = cb, cp
    return gbp


def g(p):
    q = np.array(p, dtype=float)

    for i in range(q.shape[0]):
        v = min(max(q[i, 1], 0.0), 0.5)
        u = min(max(q[i, 0], 0.6 * v), 1.0 - 0.2 * v)
        q[i] = [u, v]

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p = f()
    return p
