# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Taper toward the middle (arcsine-like profile): heavier at the
    # ends, lighter in the centre. Avoids the central overlap peak a
    # flat vector produces.
    x = (np.arange(m) + 0.5) / m
    w = (x * (1.0 - x)) ** -0.65  # steeper taper: more weight at the ends
    w = 0.8 * w / w.mean() + 0.2  # less flat weight, so the centre is lower
    h[:m] = w
    h = h / np.sum(h)  # keep the total at 1

    h = g(h)

    return h


def g(h):
    n = h.shape[0]
    return np.clip(np.asarray(h, dtype=float).reshape(n), -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
