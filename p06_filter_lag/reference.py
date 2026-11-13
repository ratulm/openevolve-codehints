"""Writes reference_output.pkl (exact minimiser of the true cost) and
standard_output.pkl (taps minimising plain mean squared error to the clean signal).

Both costs are convex quadratics in the 24 taps, so each optimum is a linear solve;
the solution is then polished and checked against task.cost directly."""
import pickle
import numpy as np
from scipy.optimize import minimize
import task as T


def quadratic(w_track, w_lag, w_rough):
    """Normal equations of w_track*E_track + w_lag*E_lag + w_rough*E_rough."""
    X, c = T.X, T.CLEAN
    Q = w_track * X.T @ X / len(c)
    b = w_track * X.T @ c / len(c)
    Xl = X[T.LAG_IDX] * T.LAG_SCALE[:, None]
    cl = c[T.LAG_IDX] * T.LAG_SCALE
    Q += w_lag * Xl.T @ Xl / len(cl)
    b += w_lag * Xl.T @ cl / len(cl)
    D = X[T.DIFF_IDX + 1] - X[T.DIFF_IDX]
    Q += w_rough * D.T @ D / len(D)
    return Q, b


def solve(w_track, w_lag, w_rough, objective):
    Q, b = quadratic(w_track, w_lag, w_rough)
    h = np.linalg.solve(Q, b)
    bounds = [(-T.TAP_BOUND, T.TAP_BOUND)] * T.N_TAPS
    h = np.clip(h, -T.TAP_BOUND, T.TAP_BOUND)
    res = minimize(objective, h, method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-15, "gtol": 1e-12})
    return res.x if objective(res.x) < objective(h) else h


if __name__ == "__main__":
    ref = solve(1.0, T.W_LAG, T.W_ROUGH, T.cost)
    std = solve(1.0, 0.0, 0.0, lambda h: T.components(h)[0])
    pickle.dump(ref, open("reference_output.pkl", "wb"))
    pickle.dump(std, open("standard_output.pkl", "wb"))
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    print("reference cost =", repr(T.cost(ref)), "components", T.components(ref))
    print("reference taps", ref)
    print("standard  cost =", T.cost(std), "components", T.components(std))
    print("standard taps", std)
    ma = np.full(T.N_TAPS, 1.0 / T.N_TAPS)
    print("moving average cost =", T.cost(ma), "components", T.components(ma))
    for name, h in (("reference", ref), ("standard", std), ("moving average", ma)):
        print(name, "score", T.score(h))
