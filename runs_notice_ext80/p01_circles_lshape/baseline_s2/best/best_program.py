# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 21
    rng = np.random.default_rng(0)
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    iu = np.triu_indices(n, 1)
    I, J = iu
    m = len(I)

    def parts(x):
        P = x[:2 * n].reshape(n, 2)
        du = np.maximum(0.6 - P[:, 0], 0)
        dv = np.maximum(0.6 - P[:, 1], 0)
        w = np.sqrt(du ** 2 + dv ** 2 + 1e-16)
        return P, x[2 * n:], du, dv, w

    def cons(x):
        P, r, du, dv, w = parts(x)
        D = np.sqrt(((P[I] - P[J]) ** 2).sum(-1) + 1e-16)
        return np.concatenate([D - r[I] - r[J], P[:, 0] - r, P[:, 1] - r,
                               1 - P[:, 0] - r, 1 - P[:, 1] - r, w - r])

    def jac(x):
        P, r, du, dv, w = parts(x)
        diff = P[I] - P[J]
        D = np.sqrt((diff ** 2).sum(-1) + 1e-16)
        u = diff / D[:, None]
        A = np.zeros((m + 5 * n, 3 * n))
        k = np.arange(m)
        A[k, 2 * I] = u[:, 0]
        A[k, 2 * I + 1] = u[:, 1]
        A[k, 2 * J] = -u[:, 0]
        A[k, 2 * J + 1] = -u[:, 1]
        A[k, 2 * n + I] = -1
        A[k, 2 * n + J] = -1
        a = np.arange(n)
        A[m + a, 2 * a] = 1
        A[m + n + a, 2 * a + 1] = 1
        A[m + 2 * n + a, 2 * a] = -1
        A[m + 3 * n + a, 2 * a + 1] = -1
        A[m + 4 * n + a, 2 * a] = -du / w
        A[m + 4 * n + a, 2 * a + 1] = -dv / w
        for b in range(5):
            A[m + b * n + a, 2 * n + a] = -1
        return A

    best_p, best_q, best_s = None, None, -1.0
    gobj = np.concatenate([np.zeros(2 * n), -np.ones(n)])

    def solve(x0):
        res = minimize(lambda x: -x[2 * n:].sum(), x0,
                       jac=lambda x: gobj,
                       constraints=[dict(type="ineq", fun=cons, jac=jac)],
                       bounds=[(0, 1)] * (2 * n) + [(0, 0.5)] * n,
                       method="SLSQP",
                       options=dict(maxiter=600, ftol=1e-12))
        P = np.clip(res.x[:2 * n].reshape(n, 2), 0, 1)
        return P, g(P, np.maximum(res.x[2 * n:], 0))
    starts = []
    for t in range(30):
        c = rng.uniform(0.05, 0.95, (300, 2))
        starts.append(c[(c[:, 0] < 0.55) | (c[:, 1] < 0.55)][:n])

    for p0 in starts:
        q0 = g(p0)
        if minimize is None:
            P, Q = p0, q0
        else:
            try:
                P, Q = solve(np.concatenate([p0.ravel(), q0]))
            except Exception:
                P, Q = p0, q0
        s = Q.sum()
        if s > best_s:
            best_p, best_q, best_s = P.copy(), Q.copy(), s

    if minimize is not None:
        for t in range(12):
            p0 = best_p + rng.normal(0, 0.004 * (1 + t % 3), best_p.shape)
            if t % 2:
                k = rng.integers(n)
                c = rng.uniform(0.05, 0.95, 2)
                while c[0] > 0.55 and c[1] > 0.55:
                    c = rng.uniform(0.05, 0.95, 2)
                p0[k] = c
            p0 = np.clip(p0, 0.01, 0.99)
            try:
                P, Q = solve(np.concatenate([p0.ravel(), best_q * 0.97]))
            except Exception:
                continue
            if Q.sum() > best_s + 1e-9:
                best_p, best_q, best_s = P.copy(), Q.copy(), Q.sum()

    return best_p, best_q


def g(p, r=None):
    n = p.shape[0]
    q = np.zeros(n)
    for i in range(n):
        u, v = p[i]
        w = np.hypot(max(0.6 - u, 0.0), max(0.6 - v, 0.0))
        b = max(0.0, min(u, v, 1 - u, 1 - v, w))
        q[i] = b if r is None else max(0.0, min(r[i], b))
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((p[i] - p[j]) ** 2))
                if q[i] + q[j] > d:
                    s = d / (q[i] + q[j]) * (1 - 1e-12)
                    q[i] *= s
                    q[j] *= s
    return q








# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    p, q = f()
    return p, q
