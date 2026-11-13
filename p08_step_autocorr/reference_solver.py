"""Offline solver for the weighted step-function autoconvolution problem (not shown to the LLM).

minimise  max_k w[k] * (a*a)[k] / (sum a)^2   over a >= 0 of length N.
Epigraph form solved by SLSQP from many random starts, each polished by
successive linear programmes (linearise a*a around the current point).
"""
import numpy as np
from scipy.optimize import minimize, linprog

N = 50


def value(a, w):
    a = np.asarray(a, dtype=float)
    return float(np.max(w * np.convolve(a, a)) / np.sum(a) ** 2)


def slsqp(a0, w, maxiter=400):
    a0 = a0 / a0.sum()
    z0 = np.append(a0, np.max(w * np.convolve(a0, a0)))

    def con(z):
        return z[-1] - w * np.convolve(z[:-1], z[:-1])

    def jac(z):
        a = z[:-1]
        J = np.zeros((2 * N - 1, N + 1))
        for i in range(N):
            J[i:i + N, i] = -2 * w[i:i + N] * a
        J[:, -1] = 1
        return J

    g = np.zeros(N + 1)
    g[-1] = 1
    res = minimize(lambda z: z[-1], z0, jac=lambda z: g, method="SLSQP",
                   bounds=[(0, None)] * (N + 1),
                   constraints=[{"type": "ineq", "fun": con, "jac": jac},
                                {"type": "eq", "fun": lambda z: z[:-1].sum() - 1,
                                 "jac": lambda z: np.append(np.ones(N), 0.0)}],
                   options={"maxiter": maxiter, "ftol": 1e-13})
    a = np.maximum(res.x[:-1], 0)
    return a / a.sum()


def lp_polish(a, w, rounds=60):
    """Successive LP: find b (sum 1, b>=0) minimising max_k w_k (a*b)_k, then step towards it."""
    a = a / a.sum()
    v = value(a, w)
    for _ in range(rounds):
        A = np.zeros((2 * N - 1, N + 1))
        for i in range(N):
            A[i:i + N, i] = w[i:i + N] * a
        A[:, -1] = -1
        c = np.zeros(N + 1)
        c[-1] = 1
        res = linprog(c, A_ub=A, b_ub=np.zeros(2 * N - 1),
                      A_eq=[np.append(np.ones(N), 0.0)], b_eq=[1.0],
                      bounds=[(0, None)] * (N + 1), method="highs")
        if res.status != 0:
            break
        b = res.x[:-1]
        improved = False
        for s in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 0.003):
            c2 = (1 - s) * a + s * b
            v2 = value(c2, w)
            if v2 < v - 1e-14:
                a, v, improved = c2 / c2.sum(), v2, True
                break
        if not improved:
            break
    return a, v


def solve(w, starts, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    best_a, best_v = None, np.inf
    for s in range(starts):
        kind = s % 3
        if kind == 0:
            a0 = rng.uniform(0.2, 1.0, N)
        elif kind == 1:
            a0 = rng.exponential(1.0, N) + 0.05
        else:
            a0 = rng.uniform(0.5, 1.0, N) * np.exp(-rng.uniform(0, 4) * np.arange(N) / N)
        a = slsqp(a0, w)
        a, v = lp_polish(a, w)
        if verbose:
            print(s, v)
        if v < best_v:
            best_a, best_v = a, v
    return best_a, best_v
