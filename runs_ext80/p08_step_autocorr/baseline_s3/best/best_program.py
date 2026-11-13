# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    a = np.zeros(n)

    # Arcsine-type profile 1/sqrt(1 - t^2) sampled at cell midpoints on (-1, 1).
    # Its peak autoconvolution is much lower than the linear ramp's
    # (about pi/2 vs about 2.21 in normalised units), and it stays smooth
    # and spread out. Midpoint sampling keeps the endpoint values finite.
    x = (np.arange(n) + 0.5) / n
    a = 1.0 / np.sqrt(x * (1.0 - x))
    a = 0.85 * a / a.mean() + 0.15

    def peak(v):
        v = v / v.sum()
        return np.max(np.convolve(v, v))

    # Annealed multi-coordinate hill-climb on the peak autoconvolution.
    rng = np.random.default_rng(0)
    best = peak(a)
    T = 60000
    for k in range(T):
        s = 0.1 * (1 - k / T) + 0.005
        idx = rng.integers(n, size=rng.integers(1, 4))
        c = a.copy()
        c[idx] = np.maximum(0.0, c[idx] * (1 + s * rng.normal(size=idx.size))
                            + 0.1 * s * rng.normal(size=idx.size))
        if c.sum() <= 0:
            continue
        v = peak(c)
        if v <= best:
            best, a = v, c

    # Smoothed-max multiplicative polish; keep only true improvements.
    a = a / a.sum()
    best_a, lr = a.copy(), 0.02
    for it in range(5000):
        cv = np.convolve(a, a)
        m = cv.max()
        w = np.exp((cv - m) / (m * 0.002))
        w /= w.sum()
        gr = np.correlate(w, a, 'valid')
        na = np.maximum(a * np.exp(-np.clip(lr * (gr / gr.mean() - 1), -1, 1)), 1e-12)
        na /= na.sum()
        v = peak(na)
        if v < best:
            best, best_a, a = v, na.copy(), na
            lr = min(lr * 1.1, 0.3)
        else:
            lr = max(lr * 0.5, 1e-5)
            a = best_a.copy()

    # Sequential LP with a trust region: linearise conv(a+d, a+d) around a and
    # minimise the worst entry directly; accept only true improvements.
    try:
        from scipy.optimize import linprog
    except Exception:
        return g(best_a)
    m = 2 * n - 1
    cost = np.r_[np.zeros(n), 1.0]
    Aeq = np.r_[np.ones(n), 0.0][None]
    ii = np.arange(n)
    rows = ii[:, None] + ii[None, :]

    def slp(a, iters):
        a = a / a.sum()
        cur = peak(a)
        r = 0.5
        for it in range(iters):
            c = np.convolve(a, a)
            M = np.zeros((m, n))
            M[rows, ii[None, :]] = 2 * a[:, None]
            A = np.hstack([M, -np.ones((m, 1))])
            sc = r * np.maximum(a, 0.01 / n)
            bnds = [(max(-a[i], -sc[i]), sc[i]) for i in range(n)] + [(None, None)]
            try:
                res = linprog(cost, A_ub=A, b_ub=-c, A_eq=Aeq, b_eq=[0.0],
                              bounds=bnds, method='highs')
            except Exception:
                break
            if not res.success:
                r *= 0.5
            else:
                na = np.maximum(a + res.x[:n], 0.0)
                na /= na.sum()
                v = peak(na)
                if v < cur:
                    cur, a = v, na
                    r = min(r * 1.5, 1.0)
                else:
                    r *= 0.5
            if r < 1e-9:
                break
        return a, cur

    a, v = slp(best_a, 800)
    if v < best:
        best, best_a = v, a.copy()

    # Basin hopping: perturb the incumbent and re-run SLP; keep improvements.
    for rep in range(12):
        s = 0.02 + 0.08 * rng.random()
        cand = best_a * np.exp(s * rng.normal(size=n))
        cand, v = slp(cand, 300)
        if v < best:
            best, best_a = v, cand.copy()

    return g(best_a)


def g(a):
    a = np.maximum(a, 0.0)
    s = np.sum(a)

    if s <= 0.0:
        a = np.ones(a.shape[0])
        s = np.sum(a)

    return a / s


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
