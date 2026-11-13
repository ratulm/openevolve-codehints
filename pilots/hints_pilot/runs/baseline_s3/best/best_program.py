# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.optimize import minimize


def construct_packing():
    n = 26
    # variables z = [x (n), y (n), r (n)]; maximize sum(r) with SLSQP
    I, J = np.triu_indices(n, 1)
    K = np.arange(len(I))
    A = np.zeros((len(I), 3 * n))
    A[K, 2 * n + I] = -1
    A[K, 2 * n + J] = -1
    # walls: x - r >= 0, y - r >= 0, 1 - x - r >= 0, 1 - y - r >= 0
    E = np.eye(n)
    Z = np.zeros((n, n))
    B = np.vstack([np.hstack([E, Z, -E]), np.hstack([Z, E, -E]),
                   np.hstack([-E, Z, -E]), np.hstack([Z, -E, -E])])
    c = np.concatenate([np.zeros(2 * n), np.ones(2 * n)])

    def dist(z):
        dx = z[I] - z[J]
        dy = z[n + I] - z[n + J]
        return dx, dy, np.sqrt(dx * dx + dy * dy + 1e-18)

    def f(z):
        d = dist(z)[2]
        return np.concatenate([d - z[2 * n + I] - z[2 * n + J], B @ z + c])

    def g(z):
        dx, dy, d = dist(z)
        G = A.copy()
        G[K, I] = dx / d
        G[K, J] = -dx / d
        G[K, n + I] = dy / d
        G[K, n + J] = -dy / d
        return np.vstack([G, B])

    e = np.zeros(3 * n)
    e[2 * n:] = -1
    layouts = [[5, 5, 6, 5, 5], [6, 5, 5, 5, 5], [5, 6, 5, 6, 4], [4, 6, 6, 6, 4]]
    rng = np.random.default_rng(1)
    best = None
    t0 = time.time()
    k = 0
    # phase 1 (k < 24): proven multi-start from row layouts
    # phase 2: basin hopping - perturb the best packing found and re-optimize
    rng2 = np.random.default_rng(7)
    while (k < 24 and (k < 1 or time.time() - t0 < 12)) or (
            k >= 24 and time.time() - t0 < 22):
        if k < 24:
            rows = layouts[k % 4]
            p = np.array([[(i + 0.5) / w, (j + 0.5) / 5]
                          for j, w in enumerate(rows) for i in range(w)])
            s = 0.02 * (k > 3) + 0.01 * (k // 8)
            p = np.clip(p + rng.normal(0, 1, p.shape) * s, 0.05, 0.95)
        else:
            p = best[0].copy()
            if k % 2:
                # relocate a few circles to random spots
                m = int(rng2.integers(1, 4))
                idx = rng2.choice(n, m, replace=False)
                p[idx] = rng2.uniform(0.05, 0.95, (m, 2))
            else:
                # jitter everything
                p += rng2.normal(0, 0.02 + 0.03 * ((k // 2) % 2), p.shape)
            p = np.clip(p, 0.03, 0.97)
        z = np.concatenate([p[:, 0], p[:, 1], helper(p) * 0.9])
        k += 1
        try:
            z = minimize(lambda z: -z[2 * n:].sum(), z, jac=lambda z: e,
                         method="SLSQP",
                         constraints=dict(type="ineq", fun=f, jac=g),
                         bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                         options=dict(maxiter=300, ftol=1e-10)).x
        except Exception:
            pass
        if not np.all(np.isfinite(z)):
            continue
        p = np.clip(np.column_stack([z[:n], z[n:2 * n]]), 0, 1)
        q = helper(p, z[2 * n:])
        if best is None or q.sum() > best[1].sum():
            best = (p, q)

    p, q = best
    return p, q, np.sum(q)


def helper(p, q=None):
    # returns valid radii: start from wall distance (or given radii, clipped),
    # then shrink any overlapping pair
    n = p.shape[0]
    b = np.minimum(p, 1 - p).min(axis=1)
    q = b.copy() if q is None else np.maximum(np.minimum(q, b) - 1e-9, 0)

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
