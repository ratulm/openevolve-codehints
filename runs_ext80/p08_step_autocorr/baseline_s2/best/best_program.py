# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    # Points at the centre of each of the n equal intervals of [0, 1]
    x = (np.arange(n) + 0.5) / n

    # Smooth hump (beta-like) with a small rise at the left edge.
    # These values set the shape and can be tuned.
    alpha, beta = 1.5, 2.3
    hump = x ** (alpha - 1.0) * (1.0 - x) ** (beta - 1.0)
    # Spikes at both ends tend to flatten the self-convolution
    edge = 0.35 * np.exp(-x / 0.08) + 0.2 * np.exp(-(1.0 - x) / 0.06)
    base = 0.15  # small constant floor: keeps every weight above zero

    a = g(base + hump / np.max(hump) + edge)

    best, v = h(a)
    # Perturbed restarts: kept only if they strictly lower the true peak
    rng = np.random.default_rng(1)
    for _ in range(4):
        b, vb = h(g(best * np.exp(0.25 * rng.standard_normal(n))), 4000)
        if vb < v:
            best, v = b, vb
    try:
        best = slp(best)
    except ImportError:
        pass
    return g(best)


def slp(a, iters=300):
    # Sequential LP on the exact peak: linearise every entry of a*a around
    # the current point, minimise the bound t inside a trust region, and
    # accept a step only when the true peak strictly decreases.
    from scipy.optimize import linprog
    n = len(a)
    a = g(a)
    v = np.max(np.convolve(a, a))
    r = 0.5 * a.max()
    for _ in range(iters):
        if r < 1e-9:
            break
        c0 = np.convolve(a, a)
        J = np.zeros((2 * n - 1, n))
        for i in range(n):
            J[i:i + n, i] = 2.0 * a
        # variables [d (n), t]; constraint c0 + J d <= t
        A = np.hstack([J, -np.ones((2 * n - 1, 1))])
        res = linprog(np.r_[np.zeros(n), 1.0], A_ub=A, b_ub=-c0,
                      A_eq=np.r_[np.ones(n), 0.0][None, :], b_eq=[0.0],
                      bounds=[(max(-r, -ai), r) for ai in a] + [(None, None)],
                      method='highs')
        if res.status != 0:
            r *= 0.5
            continue
        cand = np.maximum(a + res.x[:n], 0.0)
        cand /= cand.sum()
        vc = np.max(np.convolve(cand, cand))
        if vc < v - 1e-13:
            a, v = cand, vc
            r *= 1.2
        else:
            r *= 0.5
    return a


def h(a, iters=6000):
    # Minimise the peak of the self-convolution by gradient descent on a
    # smoothed peak (log-sum-exp); a = softmax(w) keeps a positive.
    n = len(a)
    w = np.log(np.maximum(a, 1e-12))
    best, best_v = a.copy(), np.max(np.convolve(a, a))
    for t in range(iters):
        p = np.exp(w - w.max())
        p /= p.sum()
        c = np.convolve(p, p) * (2 * n)
        v = c.max() / (2 * n)
        if v < best_v:
            best, best_v = p.copy(), v
        b = 10.0 * (1.0 + t / 60.0)
        s = np.exp(b * (c - c.max()))
        s /= s.sum()
        gp = 2.0 * np.correlate(s, p, 'valid')
        gw = p * (gp - np.dot(p, gp))
        w -= 0.4 / np.sqrt(1.0 + t / 30.0) * gw / (np.linalg.norm(gw) + 1e-15)
    return best, best_v


def g(a):
    a = np.maximum(np.nan_to_num(np.asarray(a, float), posinf=0.0, neginf=0.0), 0.0)
    s = a.sum()
    return a / s if s > 0 else np.ones(len(a)) / len(a)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
