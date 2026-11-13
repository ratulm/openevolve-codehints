# EVOLVE-BLOCK-START
import numpy as np


def construct_packing():
    import time

    n = 26
    t0 = time.time()
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    rows = np.arange(m)
    I = np.eye(n)

    # z = [x (n), y (n), r (n)]; all constraints must be >= 0
    def con(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.hypot(x[iu] - x[ju], y[iu] - y[ju])
        return np.concatenate(
            [d - r[iu] - r[ju], x - r, 1 - x - r, y - r, 1 - y - r])

    def jac(z):
        x, y = z[:n], z[n:2 * n]
        dx, dy = x[iu] - x[ju], y[iu] - y[ju]
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        J = np.zeros((m + 4 * n, 3 * n))
        J[rows, iu] = dx / d
        J[rows, ju] = -dx / d
        J[rows, n + iu] = dy / d
        J[rows, n + ju] = -dy / d
        J[rows, 2 * n + iu] = -1
        J[rows, 2 * n + ju] = -1
        J[m:, 2 * n:] = -np.tile(I, (4, 1))
        J[m:m + n, :n] = I
        J[m + n:m + 2 * n, :n] = -I
        J[m + 2 * n:m + 3 * n, n:2 * n] = I
        J[m + 3 * n:, n:2 * n] = -I
        return J

    g = np.zeros(3 * n)
    g[2 * n:] = -1

    def start(layout, rng, noise):
        pts = [((i + 0.5) / k, (j + 0.5) / len(layout))
               for j, k in enumerate(layout) for i in range(k)]
        return np.clip(np.array(pts) + rng.normal(0, noise, (n, 2)), 0.05, 0.95)

    layouts = ([5, 5, 6, 5, 5], [6, 5, 5, 5, 5], [5, 6, 5, 6, 4],
               [6, 7, 6, 7], [5, 5, 5, 5, 6], [4, 5, 4, 5, 4, 4])
    rng = np.random.default_rng(0)
    p = start(layouts[0], rng, 0.0)
    q = helper(p)

    try:
        from scipy.optimize import minimize
        # Phase 1: Initial exploration with multiple layouts
        for trial in range(12):
            if time.time() - t0 > 12:
                break
            p0 = start(layouts[trial % len(layouts)], rng, 0.02)
            z0 = np.concatenate([p0[:, 0], p0[:, 1], helper(p0)])
            res = minimize(lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: g,
                           method="SLSQP",
                           bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                           constraints=dict(type="ineq", fun=con, jac=jac),
                           options=dict(maxiter=300, ftol=1e-9))
            pc = np.clip(np.column_stack([res.x[:n], res.x[n:2 * n]]), 0, 1)
            qc = fix(pc, res.x[2 * n:])
            if np.sum(qc) > np.sum(q):
                p, q = pc, qc
        
        # Phase 2: Basin hopping around best solution
        h = 0
        while time.time() - t0 < 32:
            h += 1
            p0 = p.copy()
            if h % 3 == 0:
                p0 = p0 + rng.normal(0, 0.02, p.shape)
            elif h % 3 == 1:
                i = np.argsort(q)[rng.integers(0, min(4, n))]
                p0[i] = rng.uniform(0.05, 0.95, 2)
            else:
                idx = rng.choice(n, int(rng.integers(2, 6)), replace=False)
                p0[idx] += rng.normal(0, 0.06, (len(idx), 2))
            p0 = np.clip(p0, 0.03, 0.97)
            z0 = np.concatenate([p0[:, 0], p0[:, 1], helper(p0) * 0.9])
            try:
                res = minimize(lambda z: -np.sum(z[2 * n:]), z0, jac=lambda z: g,
                               method="SLSQP",
                               bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                               constraints=dict(type="ineq", fun=con, jac=jac),
                               options=dict(maxiter=300, ftol=1e-9))
                pc = np.clip(np.column_stack([res.x[:n], res.x[n:2 * n]]), 0, 1)
                qc = fix(pc, res.x[2 * n:])
                if np.sum(qc) > np.sum(q):
                    p, q = pc, qc
            except Exception:
                pass
    except Exception:
        pass

    s = float(np.sum(q))

    return p, q, s


def helper(p, r=None):
    # Valid radii: cap by walls (and by suggested r), then shrink overlaps.
    # Radii only ever decrease, so one pass leaves every pair non-overlapping.
    n = p.shape[0]
    q = np.minimum(p, 1 - p).min(axis=1)
    if r is not None:
        q = np.minimum(q, np.maximum(r, 0))

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

            if q[i] + q[j] > d:
                k = d / (q[i] + q[j]) * (1 - 1e-12)
                q[i] *= k
                q[j] *= k

    return q


def fix(p, r):
    """Shrink radii so the packing is strictly valid."""
    n = len(r)
    q = np.minimum(np.maximum(r, 0), np.min(np.hstack([p, 1 - p]), axis=1))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))
            if q[i] + q[j] > d:
                f = d / (q[i] + q[j])
                q[i] *= f
                q[j] *= f
    return q * (1 - 1e-9)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
