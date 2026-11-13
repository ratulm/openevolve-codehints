# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    a = np.zeros(n)

    # Arcsine-like profile: heavier mass near both ends gives a flatter
    # autoconvolution (lower peak / higher L2-to-L1·Linf ratio) than a ramp.
    for i in range(n):
        x = (i + 0.5) / n
        a[i] = 1.0 / np.sqrt(x * (1.0 - x))

    # Mild blending with a constant to tame the extreme endpoints
    a = 0.85 * a / np.mean(a) + 0.15

    a = g(a)

    # Refine: minimise the peak of the autoconvolution (flatter a*a) via
    # softmax-smoothed minimax gradient descent on the simplex.
    best, best_val = a.copy(), np.max(np.convolve(a, a))
    for beta, lr, steps in ((200.0, 0.05, 1500), (1000.0, 0.02, 1500),
                            (5000.0, 0.005, 2000)):
        for _ in range(steps):
            c = np.convolve(a, a)
            z = beta * n * (c - c.max())
            w = np.exp(z)
            w /= w.sum()
            grad = 2.0 * np.correlate(w, a, 'valid')
            grad -= grad.mean()
            a = g(a - lr * grad / (np.abs(grad).max() + 1e-12) / n)
            v = np.max(np.convolve(a, a))
            if v < best_val:
                best, best_val = a.copy(), v
        a = best.copy()

    return best


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
