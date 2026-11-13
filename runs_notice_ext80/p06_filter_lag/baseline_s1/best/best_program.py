# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 24
    m = 24
    h = np.zeros(n)

    # Pure edge-heavy profile; p2.2->0.4107, p3.0->0.4169, p4.0->0.4165.
    # Quadratic fit through these peaks near p~3.45.
    x = (np.arange(m) + 0.5) / m * 2.0 - 1.0  # bin centres in (-1, 1)
    w = (1.0 - x * x) ** -3.4
    h[:m] = w / w.sum()  # unit total mass

    h = g(h)

    return h


def g(h):
    return np.clip(h, -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
