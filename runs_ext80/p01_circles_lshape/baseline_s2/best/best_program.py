# EVOLVE-BLOCK-START
import numpy as np


N = 21


def rows_layout(rows):
    pts = []
    h = len(rows)
    for j, m in enumerate(rows):
        y = (j + 0.5) / h
        off = 0.5 / m if j % 2 == 0 else 0.5 / m
        for i in range(m):
            pts.append([off + i / m, y])
    return np.array(pts[:N])


def solve(p0, r0):
    from scipy.optimize import minimize
    n = N
    iu, ju = np.triu_indices(n, 1)
    npair = len(iu)

    def cons(z):
        p = z[:2 * n].reshape(n, 2)
        q = z[2 * n:]
        d = np.sqrt(((p[iu] - p[ju]) ** 2).sum(1))
        a = np.maximum(0.6 - p[:, 0], 0)
        b = np.maximum(0.6 - p[:, 1], 0)
        return np.concatenate([d - q[iu] - q[ju],
                               p[:, 0] - q, p[:, 1] - q,
                               1 - p[:, 0] - q, 1 - p[:, 1] - q,
                               np.hypot(a, b) - q])

    def jac(z):
        p = z[:2 * n].reshape(n, 2)
        diff = p[iu] - p[ju]
        d = np.sqrt((diff ** 2).sum(1)) + 1e-12
        u = diff / d[:, None]
        J = np.zeros((npair + 5 * n, 3 * n))
        aa = np.maximum(0.6 - p[:, 0], 0)
        bb = np.maximum(0.6 - p[:, 1], 0)
        hh = np.hypot(aa, bb) + 1e-12
        A = np.arange(n)
        J[npair + 4 * n + A, 2 * A] = -aa / hh
        J[npair + 4 * n + A, 2 * A + 1] = -bb / hh
        J[npair + 4 * n + A, 2 * n + A] = -1
        k = np.arange(npair)
        J[k, 2 * iu] = u[:, 0]
        J[k, 2 * iu + 1] = u[:, 1]
        J[k, 2 * ju] = -u[:, 0]
        J[k, 2 * ju + 1] = -u[:, 1]
        J[k, 2 * n + iu] = -1
        J[k, 2 * n + ju] = -1
        a = np.arange(n)
        b = npair
        J[b + a, 2 * a] = 1
        J[b + n + a, 2 * a + 1] = 1
        J[b + 2 * n + a, 2 * a] = -1
        J[b + 3 * n + a, 2 * a + 1] = -1
        for t in range(4):
            J[b + t * n + a, 2 * n + a] = -1
        return J

    z0 = np.concatenate([p0.ravel(), r0])
    obj = lambda z: -z[2 * n:].sum()
    grad = lambda z: np.concatenate([np.zeros(2 * n), -np.ones(n)])
    res = minimize(obj, z0, jac=grad, method='SLSQP',
                   constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}],
                   bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                   options={'maxiter': 500, 'ftol': 1e-12})
    z = res.x
    return z[:2 * n].reshape(n, 2), z[2 * n:]


def f():
    n = N
    p = np.zeros((n, 2))
    k = 0
    for j in range(5):
        m = 5 if j < 3 else 3
        for i in range(m):
            p[k] = [0.06 + 0.12 * i, 0.06 + 0.12 * j]
            k += 1

    starts = [p, rows_layout([5, 4, 5, 4]), rows_layout([4, 5, 4, 4, 4])]
    rng = np.random.default_rng(12345)
    for _ in range(16):
        starts.append(rng.uniform(0.08, 0.92, (n, 2)))

    import time
    t0 = time.time()
    best_p, best_q = p, g(p)
    for s in starts:
        if time.time() - t0 > 30:
            break
        try:
            s = np.clip(s + rng.normal(0, 1e-3, s.shape), 0.01, 0.99)
            sp, sq = solve(s, np.full(n, 0.02))
            if not (np.all(np.isfinite(sp)) and np.all(np.isfinite(sq))):
                continue
            sp = np.clip(sp, 0.0, 1.0)
            vq = g(sp, sq)
            if vq.sum() > best_q.sum():
                best_p, best_q = sp, vq
        except Exception:
            continue

    return best_p, best_q


def g(p, r=None):
    n = p.shape[0]
    q = np.zeros(n)

    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        q[i] = min(u, v, 1 - u, 1 - v, w)
        if r is not None:
            q[i] = min(q[i], max(r[i], 0.0))
        q[i] = max(q[i] - 1e-9, 0.0)

    for _ in range(50):
        ch = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((p[i] - p[j]) ** 2))
                if q[i] + q[j] > d - 1e-9:
                    s = max(d - 2e-9, 0.0) / (q[i] + q[j])
                    q[i] *= s
                    q[j] *= s
                    ch = True
        if not ch:
            break

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
