# EVOLVE-BLOCK-START
import numpy as np


def _ratio_and_grad(x, beta):
    a = x * x
    c = np.convolve(a, a)
    s = a.sum()
    m = c.max()
    w = np.exp(beta * (c - m) / m)
    w /= w.sum()
    sm = float(np.dot(w, c))
    ga = 2.0 * np.correlate(w, a, mode="valid")
    gr = ga / (s * s) - 2.0 * sm / (s ** 3)
    return m / (s * s), gr * 2.0 * x


def _exact(a):
    s = a.sum()
    return np.convolve(a, a).max() / (s * s) if s > 0 else np.inf


def f():
    # Evaluator requires length 50. Gentle annealed Adam from the uniform
    # baseline + mild random starts, then a strictly-improving polish (0.6160).
    n = 50
    rng = np.random.default_rng(0)
    t = np.linspace(-1.0, 1.0, n)
    best_a = np.ones(n)
    best_v = _exact(best_a)
    for trial in range(8):
        if trial == 0:
            x = np.ones(n)
        else:
            x = np.sqrt(np.clip(1.0 + 0.3 * rng.standard_normal(n) + 0.5 * t * t, 0.05, None))
        mom = np.zeros(n)
        vel = np.zeros(n)
        iters = 2000
        for it in range(1, iters + 1):
            frac = it / iters
            beta = 20.0 + 980.0 * frac
            lr = 0.02 * (1.0 - 0.95 * frac)
            _, gx = _ratio_and_grad(x, beta)
            if not np.all(np.isfinite(gx)):
                break
            ve = _exact(x * x)
            if ve < best_v:
                best_v, best_a = ve, (x * x).copy()
            mom = 0.9 * mom + 0.1 * gx
            vel = 0.999 * vel + 0.001 * gx * gx
            x = x - lr * (mom / (1 - 0.9 ** it)) / (np.sqrt(vel / (1 - 0.999 ** it)) + 1e-12)
            x = x / np.sqrt((x * x).sum() / n)
    best_a = _polish(best_a / best_a.sum())
    return g(best_a)


def _polish(a):
    # Only ever accepts a strict improvement in the exact ratio.
    n = a.shape[0]
    bv = _exact(a)
    try:
        from scipy.optimize import linprog
        for _ in range(60):
            A = np.zeros((2 * n - 1, n))
            for i in range(n):
                A[i:i + n, i] = a
            Aub = np.hstack([2.0 * A, -np.ones((2 * n - 1, 1))])
            res = linprog(np.r_[np.zeros(n), 1.0], A_ub=Aub, b_ub=np.zeros(2 * n - 1),
                          A_eq=np.r_[np.ones(n), 0.0][None], b_eq=[1.0],
                          bounds=[(0, None)] * n + [(None, None)], method="highs")
            if not res.success:
                break
            b = res.x[:n]
            moved = False
            for lam in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
                c = (1 - lam) * a + lam * b
                vc = _exact(c)
                if vc < bv - 1e-15:
                    a, bv, moved = c / c.sum(), vc, True
                    break
            if not moved:
                break
    except Exception:
        pass
    rng = np.random.default_rng(1)
    step = 0.05
    for it in range(20000):
        c = a * np.exp(step * rng.standard_normal(n) * (rng.random(n) < 0.2))
        c /= c.sum()
        vc = _exact(c)
        if vc < bv:
            a, bv = c, vc
        elif it % 2000 == 1999:
            step *= 0.7
    return a


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
