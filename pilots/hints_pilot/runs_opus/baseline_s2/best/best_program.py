# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    n = 26
    iu = np.triu_indices(n, 1)

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.sqrt((x[iu[0]] - x[iu[1]]) ** 2 + (y[iu[0]] - y[iu[1]]) ** 2)
        return np.concatenate([d - r[iu[0]] - r[iu[1]],
                               x - r, 1 - x - r, y - r, 1 - y - r])

    def cjac(z):
        x, y = z[:n], z[n:2 * n]
        i, j = iu
        m = len(i)
        dx, dy = x[i] - x[j], y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy) + 1e-12
        J = np.zeros((m + 4 * n, 3 * n))
        a = np.arange(m)
        J[a, i], J[a, j] = dx / d, -dx / d
        J[a, n + i], J[a, n + j] = dy / d, -dy / d
        J[a, 2 * n + i] = -1
        J[a, 2 * n + j] = -1
        k = np.arange(n)
        for b, (c, s) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1)]):
            J[m + b * n + k, c * n + k] = s
            J[m + b * n + k, 2 * n + k] = -1
        return J

    gr = np.r_[np.zeros(2 * n), -np.ones(n)]

    def opt(z0):
        return minimize(lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: gr,
                        constraints={"type": "ineq", "fun": cons, "jac": cjac},
                        bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                        method="SLSQP",
                        options={"maxiter": 1000, "ftol": 1e-12}).x

    import time
    t_start = time.time()
    rng = np.random.default_rng(0)
    g = [(i + 0.5) / 5 for i in range(5)]
    grid = np.array([(a, b) for a in g for b in g] + [(0.5, 0.5)])
    best, bs = None, 0.0
    for t in range(30):
        if t % 2 == 0:
            p0 = grid + rng.normal(0, 0.03, (n, 2))
        else:
            p0 = rng.uniform(0.1, 0.9, (n, 2))
        p0 = np.clip(p0, 0.05, 0.95)
        z0 = np.concatenate([p0[:, 0], p0[:, 1], np.full(n, 0.05)])
        try:
            z = opt(z0)
        except Exception:
            continue
        if cons(z).min() > -1e-9 and np.sum(z[2 * n:]) > bs:
            bs, best = np.sum(z[2 * n:]), z.copy()
    # Basin-hopping style refinement around the incumbent
    if best is not None:
        for k in range(300):
            if time.time() - t_start > 40:
                break
            z0 = best.copy()
            sig = [0.02, 0.05, 0.1][k % 3]
            z0[:2 * n] = np.clip(z0[:2 * n] + rng.normal(0, sig, 2 * n),
                                 0.02, 0.98)
            z0[2 * n:] *= 0.8
            try:
                z = opt(z0)
            except Exception:
                continue
            if cons(z).min() > -1e-9 and np.sum(z[2 * n:]) > bs + 1e-8:
                bs, best = np.sum(z[2 * n:]), z.copy()
    if best is not None:
        p = np.c_[best[:n], best[n:2 * n]]
        q = np.maximum(best[2 * n:] - 1e-9, 0)
        return p, q, float(np.sum(q))

    p = np.zeros((n, 2))

    p[0] = [0.5, 0.5]

    for i in range(8):
        t = 2 * np.pi * i / 8
        p[i + 1] = [0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)]

    for i in range(16):
        t = 2 * np.pi * i / 16
        p[i + 9] = [0.5 + 0.7 * np.cos(t), 0.5 + 0.7 * np.sin(t)]

    p = np.clip(p, 0.01, 0.99)

    q = helper(p)

    s = np.sum(q)

    return p, q, s


def helper(p):
    n = p.shape[0]
    q = np.ones(n)

    for i in range(n):
        u, v = p[i]
        q[i] = min(u, v, 1 - u, 1 - v)

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
