# EVOLVE-BLOCK-START
import numpy as np


def construct_packing():
    import time

    n = 26
    t0 = time.time()
    rng = np.random.default_rng(3)
    # Staggered row layouts as starting points for joint optimisation.
    starts = []
    for rows in ([5, 5, 6, 5, 5], [5, 6, 5, 5, 5], [6, 5, 4, 5, 6], [5, 4, 5, 4, 5, 3],
                 [5, 6, 5, 6, 4], [5, 5, 5, 5, 6]):
        m = len(rows)
        starts.append(np.array([[(i + 0.5) / k, (j + 0.5) / m]
                                for j, k in enumerate(rows) for i in range(k)]))

    best, bs = None, -1.0
    for t in range(400):
        if t < len(starts):
            p0 = starts[t] + rng.normal(0, 0.01, (n, 2))
        elif t % 3 == 0:
            # fresh, more strongly perturbed structured start (diversification)
            p0 = starts[rng.integers(len(starts))] + rng.normal(0, 0.03, (n, 2))
        else:
            # basin hopping: perturb the incumbent (intensification)
            p0 = best[0] + rng.normal(0, rng.choice([0.01, 0.02, 0.04]), (n, 2))
            if t % 3 == 1:
                # relocate the smallest circle to a random spot
                p0[np.argmin(best[1])] = rng.uniform(0.05, 0.95, 2)
        p = np.clip(p0, 0.05, 0.95)
        try:
            p, q = refine(p)
        except Exception:
            q = helper(p)
        if q.sum() > bs:
            best, bs = (p, q), q.sum()
        if time.time() - t0 > 20:
            break

    p, q = best
    return p, q, np.sum(q)


def refine(p):
    """Jointly optimise centers and radii (maximise sum of radii) with SLSQP."""
    from scipy.optimize import minimize

    n = len(p)
    I, J = np.triu_indices(n, 1)
    m = len(I)
    K = np.arange(m)

    def con(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        d = np.hypot(x[I] - x[J], y[I] - y[J])
        return np.concatenate([d - r[I] - r[J], x - r, y - r, 1 - x - r, 1 - y - r])

    def jac(z):
        x, y = z[:n], z[n:2 * n]
        dx, dy = x[I] - x[J], y[I] - y[J]
        d = np.hypot(dx, dy) + 1e-12
        A = np.zeros((m + 4 * n, 3 * n))
        A[K, I] = dx / d
        A[K, J] = -dx / d
        A[K, n + I] = dy / d
        A[K, n + J] = -dy / d
        A[K, 2 * n + I] = -1
        A[K, 2 * n + J] = -1
        e = np.eye(n)
        for k, (cx, cy) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
            a = m + k * n
            A[a:a + n, :n] = cx * e
            A[a:a + n, n:2 * n] = cy * e
            A[a:a + n, 2 * n:] = -e
        return A

    g = np.zeros(3 * n)
    g[2 * n:] = -1
    z = np.concatenate([p[:, 0], p[:, 1], helper(p) * 0.9])
    res = minimize(lambda z: -z[2 * n:].sum(), z, jac=lambda z: g, method="SLSQP",
                   constraints=dict(type="ineq", fun=con, jac=jac),
                   bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                   options=dict(maxiter=300, ftol=1e-10))
    z = res.x
    p = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0, 1)
    return p, helper(p, z[2 * n:])


def helper(p, q=None):
    """Valid radii for centers p; if q is given, shrink it until valid."""
    n = p.shape[0]
    b = np.minimum(np.minimum(p[:, 0], p[:, 1]),
                   np.minimum(1 - p[:, 0], 1 - p[:, 1]))
    q = b.copy() if q is None else np.minimum(np.maximum(q, 0), b)
    q = q * (1 - 1e-9)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

            if q[i] + q[j] > d:
                k = d / (q[i] + q[j]) * (1 - 1e-12)
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
