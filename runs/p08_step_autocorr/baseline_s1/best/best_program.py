# EVOLVE-BLOCK-START
import numpy as np


def f():
    n = 50
    x = np.arange(n, dtype=float)

    # flat 0.4 -> 0.564, 0.6 -> 0.581, ~0.8 -> 0.577: optimum lies between,
    # closer to 0.6. Keep the ramp:sqrt ratio 5:3.
    r = (1.0 + x) / n
    a = 0.65 + 0.21875 * r + 0.13125 * np.sqrt(r)

    a = g(a)

    return a


def g(a, eps=1e-12):
    a = np.asarray(a, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    # Small floor so no weight is exactly zero (keeps log/ratio scoring safe)
    a = np.maximum(a, 0.0) + eps
    return a / np.sum(a)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    a = f()
    return a
