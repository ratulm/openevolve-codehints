"""Writes reference_output.pkl and standard_output.pkl. Neither uses the hidden law:
both are computed from the 80 training points stored in the initial program.

standard_output.pkl  -- the "standard objective": minimise prediction error on the
    training range. A thin-plate smoothing spline whose smoothing level is chosen by
    8-fold cross-validation on the training points.

reference_output.pkl -- best extrapolating model found from the data alone. A library
    of 120 small parametric structures  c0 + c1*g(x1)*h(x2) + k(x1, x2)  is fitted by
    multi-start least squares. Structures are ranked by how well a fit to the INNER part
    of the training region (x1 <= 3.1 and x2 <= 3.9) predicts the held-out OUTER band of
    the training region, i.e. by measured extrapolation skill. The top 5 are refitted on
    all 80 points. The prediction is the mean of the top m plus c times the spread (std)
    of the top 5, with m and c chosen on the outer band under the true asymmetric loss.
"""
import importlib.util
import itertools
import pickle
import warnings

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("prog", "baseline/initial_program.py")
prog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prog)
X, y, Q = prog.a, prog.b, prog.c
n = len(y)

# ---------------- standard: in-range cross-validated smoothing spline ----------------
folds = np.array_split(np.random.default_rng(0).permutation(n), 8)


def cv_rmse(s):
    e = 0.0
    for f in folds:
        m = np.ones(n, bool)
        m[f] = False
        e += np.sum((RBFInterpolator(X[m], y[m], kernel="thin_plate_spline", smoothing=s)(X[f]) - y[f]) ** 2)
    return np.sqrt(e / n)


grid = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
s_best = min(grid, key=cv_rmse)
standard = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=s_best)(Q)
pickle.dump(standard, open("standard_output.pkl", "wb"))
print(f"standard: thin-plate spline, smoothing={s_best}, in-range CV RMSE={cv_rmse(s_best):.4f}")

# ---------------- reference: structure search ranked by held-out outer band ----------------
G = {  # shape in x1: (function, starting parameters)
    "hyp": (lambda x, b: x / (np.abs(b) + x), [1.0]),
    "exp": (lambda x, b: 1 - np.exp(-np.abs(b) * x), [0.5]),
    "tanh": (lambda x, b: np.tanh(np.abs(b) * x), [0.5]),
    "pow": (lambda x, b: x ** b, [0.5]),
    "lin": (lambda x: x, []),
    "one": (lambda x: 1 + 0 * x, []),
}
H = {  # shape in x2
    "cos": (lambda x, w, p: np.cos(w * x + p), [0.5, 0.0]),
    "quad": (lambda x, a, b: 1 + a * x + b * x * x, [0.0, 0.0]),
    "gauss": (lambda x, m, s: np.exp(-((x - m) / s) ** 2), [0.0, 3.0]),
    "lin": (lambda x, a: 1 + a * x, [0.0]),
}
K = {  # additive trend
    "pow1": (lambda a, b, c, p: c * a ** p, [0.3, 1.0]),
    "lin12": (lambda a, b, c, d: c * a + d * b, [0.1, 0.1]),
    "pow1lin2": (lambda a, b, c, p, d: c * a ** p + d * b, [0.3, 1.0, 0.1]),
    "pow1pow2": (lambda a, b, c, p, d, q: c * a ** p + d * b ** q, [0.3, 1.0, 0.1, 1.0]),
    "none": (lambda a, b: 0 * a, []),
}


def make(g, h, k):
    ng, nh = len(G[g][1]), len(H[h][1])

    def f(t, Z):
        return (t[0] + t[1] * G[g][0](Z[:, 0], *t[2:2 + ng]) * H[h][0](Z[:, 1], *t[2 + ng:2 + ng + nh])
                + K[k][0](Z[:, 0], Z[:, 1], *t[2 + ng + nh:]))

    return f, np.r_[0.0, 1.0, G[g][1], H[h][1], K[k][1]]


def fit(f, t0, A, b, starts=6):
    r = np.random.default_rng(0)
    best = None
    for s in range(starts):
        t = t0 * (1 + 0.6 * r.normal(size=len(t0))) + 0.3 * r.normal(size=len(t0)) if s else t0
        try:
            o = least_squares(lambda t: np.nan_to_num(f(t, A) - b, nan=1e3, posinf=1e3, neginf=-1e3),
                              t, method="lm", max_nfev=600)
        except Exception:
            continue
        if best is None or o.cost < best.cost:
            best = o
    return best


inner = (X[:, 0] <= 3.1) & (X[:, 1] <= 3.9)
cands = []
for g, h, k in itertools.product(G, H, K):
    f, t0 = make(g, h, k)
    o = fit(f, t0, X[inner], y[inner])
    if o is None:
        continue
    vp = f(o.x, X[~inner])
    v = np.sqrt(np.mean((vp - y[~inner]) ** 2))
    if np.isfinite(v):
        cands.append((v, (g, h, k), vp))
cands.sort(key=lambda c: c[0])
top = cands[:5]
W = 8.0  # the true rule: squared error counts 8x where the prediction is too low


def asym(p, t):
    d = p - t
    return np.sqrt(np.mean(np.where(d < 0, W, 1.0) * d ** 2))


# Asymmetry handling: predict (mean of top-m structures) + c * (disagreement among the top 5
# structures at that point). m and c are chosen on the held-out outer band under the asymmetric
# loss, so the upward shift grows where the candidate structures diverge.
band_spread = np.std([c[2] for c in top], 0)
best = min(((asym(np.mean([c[2] for c in top[:m]], 0) + c * band_spread, y[~inner]), m, c)
            for m in range(1, 6) for c in np.arange(0, 4.01, 0.1)), key=lambda t: t[0])
_, m_best, c_best = best
preds = []
for v, name, _ in top:
    f, t0 = make(*name)
    preds.append(f(fit(f, t0, X, y).x, Q))
    print(f"library member {name}: outer-band validation RMSE={v:.4f}")
reference = np.mean(preds[:m_best], 0) + c_best * np.std(preds, 0)
pickle.dump(reference, open("reference_output.pkl", "wb"))
print(f"reference: mean of top {m_best} + {c_best:.1f} * spread of top 5; outer-band asymmetric loss={best[0]:.4f}")

import task
print("E_REF (put in task.py) =", repr(task.extrapolation_error(reference)))
print("scores: reference", task.score(reference)[0], "| standard", task.score(standard)[0],
      "| initial", task.score(prog.run())[0], "| unshifted mean of top m", task.score(np.mean(preds[:m_best], 0))[0])
