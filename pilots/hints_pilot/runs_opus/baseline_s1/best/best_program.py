# EVOLVE-BLOCK-START
import numpy as np


def construct_packing():
    from scipy.optimize import minimize

    n = 26
    rng = np.random.default_rng(1)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    ar = np.arange(n)

    def obj(x):
        return -x[2 * n:].sum()

    def gobj(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1
        return g

    def cons(x):
        p = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        d = np.sqrt(((p[iu] - p[ju]) ** 2).sum(-1))
        return np.concatenate([d - r[iu] - r[ju],
                               p[:, 0] - r, p[:, 1] - r,
                               1 - p[:, 0] - r, 1 - p[:, 1] - r])

    def jcons(x):
        p = x[:2 * n].reshape(n, 2)
        diff = p[iu] - p[ju]
        d = np.maximum(np.sqrt((diff ** 2).sum(-1)), 1e-12)
        u = diff / d[:, None]
        J = np.zeros((m + 4 * n, 3 * n))
        k = np.arange(m)
        J[k, 2 * iu] = u[:, 0]
        J[k, 2 * iu + 1] = u[:, 1]
        J[k, 2 * ju] = -u[:, 0]
        J[k, 2 * ju + 1] = -u[:, 1]
        J[k, 2 * n + iu] = -1
        J[k, 2 * n + ju] = -1
        for b, (c, s) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            rows = m + b * n + ar
            J[rows, 2 * ar + c] = s
            J[rows, 2 * n + ar] = -1
        return J

    g = [(i + 0.5) / 5 for i in range(5)]
    grid = np.array([(a, b) for a in g for b in g] + [(0.5, 0.5)])
    hexp = []
    for row, cnt in enumerate([5, 6, 5, 6, 4]):
        y = 0.1 + row * 0.2
        for c in range(cnt):
            hexp.append(((c + 0.5) / cnt, y))
    hexp = np.array(hexp)

    best, bx = -1.0, None
    for t in range(40):
        k = t % 4
        if k == 0:
            p0 = grid + rng.normal(0, 0.04, (n, 2))
        elif k == 1:
            p0 = hexp + rng.normal(0, 0.04, (n, 2))
        elif k == 2 or bx is None:
            p0 = rng.uniform(0.05, 0.95, (n, 2))
        else:
            p0 = bx[:2 * n].reshape(n, 2) + rng.normal(0, 0.05, (n, 2))
        p0 = np.clip(p0, 0.03, 0.97)
        r0 = helper(p0)
        x0 = np.concatenate([p0.ravel(), r0])
        try:
            res = minimize(obj, x0, jac=gobj, method='SLSQP',
                           constraints={'type': 'ineq', 'fun': cons,
                                        'jac': jcons},
                           bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                           options={'maxiter': 600, 'ftol': 1e-12})
        except Exception:
            continue
        if cons(res.x).min() > -1e-8 and -res.fun > best:
            best, bx = -res.fun, res.x

    if bx is None:
        p = np.clip(grid, 0.01, 0.99)
        q = helper(p)
        return p, q, np.sum(q)

    p = bx[:2 * n].reshape(n, 2)
    q = helper(p, np.maximum(bx[2 * n:] - 1e-9, 0))
    return p, q, np.sum(q)


def helper(p, r0=None):
    n = p.shape[0]
    q = np.ones(n) if r0 is None else np.array(r0, dtype=float)

    for i in range(n):
        u, v = p[i]
        q[i] = max(0.0, min(q[i], u, v, 1 - u, 1 - v))

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

            if q[i] + q[j] > d:
                k = d / (q[i] + q[j])
                q[i] *= k
                q[j] *= k

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
