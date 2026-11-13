"""Offline reference solver for the L-shaped packing instance (not shown to the LLM).

Used to (1) pick a hidden objective that penalises the standard guesses and
(2) compute the normalisation constant for the evaluator.
"""
import sys
import numpy as np
from scipy.optimize import minimize

N = 21
A = 0.6  # the notch [A,1] x [A,1] is removed from the unit square

OBJECTIVES = {
    "sum_r": lambda x, y, r: np.sum(r),
    "sum_r2": lambda x, y, r: np.sum(r**2),
    "sum_r1.5": lambda x, y, r: np.sum(np.abs(r) ** 1.5),
    "pos_weighted": lambda x, y, r: np.sum(r * (0.5 + x)),
    "sum_sqrt_r": lambda x, y, r: np.sum(np.sqrt(np.abs(r) + 1e-12)),
}


def split(z):
    return z[:N], z[N : 2 * N], z[2 * N :]


def constraints():
    iu = np.triu_indices(N, 1)

    def walls(z):
        x, y, r = split(z)
        notch = np.hypot(np.maximum(A - x, 0), np.maximum(A - y, 0)) - r
        return np.concatenate([x - r, y - r, 1 - x - r, 1 - y - r, notch, r])

    def pairs(z):
        x, y, r = split(z)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        return dx**2 + dy**2 - (r[iu[0]] + r[iu[1]]) ** 2

    return [{"type": "ineq", "fun": walls}, {"type": "ineq", "fun": pairs}]


def is_valid(z, tol=1e-9):
    return all(np.all(c["fun"](z) >= -tol) for c in constraints())


def random_start(rng):
    pts = []
    while len(pts) < N:
        p = rng.uniform(0.05, 0.95, 2)
        if not (p[0] > A - 0.05 and p[1] > A - 0.05):
            pts.append(p)
    pts = np.array(pts)
    return np.concatenate([pts[:, 0], pts[:, 1], np.full(N, 0.03)])


def solve(obj, starts, seed=0):
    rng = np.random.default_rng(seed)
    cons = constraints()
    best_z, best_v = None, -np.inf
    for _ in range(starts):
        res = minimize(
            lambda z: -obj(*split(z)), random_start(rng), method="SLSQP",
            constraints=cons, options={"maxiter": 300, "ftol": 1e-10},
        )
        z = res.x.copy()
        z[2 * N :] = np.maximum(z[2 * N :] - 1e-8, 0)  # shrink for strict validity
        if is_valid(z):
            v = obj(*split(z))
            if v > best_v:
                best_z, best_v = z, v
    return best_z, best_v


if __name__ == "__main__":
    starts = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sols = {k: solve(f, starts) for k, f in OBJECTIVES.items()}
    names = list(OBJECTIVES)
    print("rows: objective used to score; cols: objective the packing was optimised for")
    print("%-14s" % "" + "".join("%-14s" % n for n in names))
    for k in names:
        ref = sols[k][1]
        row = [OBJECTIVES[k](*split(sols[j][0])) / ref for j in names]
        print("%-14s" % k + "".join("%-14.3f" % v for v in row) + "  (best=%.5f)" % ref)
