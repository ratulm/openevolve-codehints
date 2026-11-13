# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Sampled 1/sqrt(x) profile: its autoconvolution is nearly flat, so its peak
    # is lower than the triangular peak of a flat step function.
    for k in range(m):
        h[k] = 1.0 / np.sqrt(k + 0.5)

    h = h / np.sum(h)

    h = lp(opt(h))

    # A few perturbed restarts, each polished by the exact LP step.
    rng = np.random.default_rng(0)
    bv = peak(h)
    for r in range(6):
        x0 = h * np.exp(0.25 * (0.8 ** r) * rng.standard_normal(n))
        y = lp(opt(x0 / x0.sum()))
        v = peak(y)
        if v < bv:
            bv, h = v, y

    h = g(h)

    return h


def lp(x, iters=300):
    # Sequential linear programming on the exact peak:
    # min t  s.t.  c + 2*conv(x, d) <= t,  sum(d) = 0,  x + d >= 0,  |d| <= tr
    try:
        from scipy.optimize import linprog
    except Exception:
        return x
    n = len(x)
    x = x / x.sum()
    tr = 0.05 / n
    for _ in range(iters):
        c = np.convolve(x, x)
        M = np.zeros((2 * n - 1, n))
        for i in range(n):
            M[i:i + n, i] += 2.0 * x
        A = np.hstack([M, -np.ones((2 * n - 1, 1))])
        try:
            res = linprog(np.r_[np.zeros(n), 1.0], A_ub=A, b_ub=-c,
                          A_eq=np.r_[np.ones(n), 0.0][None], b_eq=[0.0],
                          bounds=[(max(-x[i], -tr), tr) for i in range(n)]
                          + [(None, None)], method='highs')
        except Exception:
            break
        if res.x is None:
            break
        y = np.maximum(x + res.x[:n], 0.0)
        y /= y.sum()
        if peak(y) < peak(x) - 1e-15:
            x = y
        else:
            tr *= 0.5
            if tr < 1e-12:
                break
    return x


def peak(x):
    s = np.sum(x)
    return np.max(np.convolve(x, x)) / (s * s) if s > 0 else np.inf


def opt(h0):
    # Smoothed-max (softmax) projected gradient descent on autoconvolution peak.
    x = h0.copy()
    best, bv = x.copy(), peak(x)
    for beta in [20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]:
        x = best.copy()
        lr = 0.05
        for _ in range(1500):
            c = np.convolve(x, x)
            w = np.exp(beta * (c / c.max() - 1.0))
            w /= w.sum()
            gr = 2.0 * np.correlate(w, x, 'valid')
            gr -= gr.mean()
            gm = np.abs(gr).max()
            if gm <= 0:
                break
            x = np.maximum(x - lr * gr / gm * x.mean(), 0.0)
            x /= x.sum()
            v = peak(x)
            if v < bv:
                bv, best = v, x.copy()
            lr *= 0.998
    return best


def g(h):
    n = h.shape[0]
    q = np.zeros(n)

    for k in range(n):
        q[k] = min(max(h[k], -2.0), 2.0)

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
