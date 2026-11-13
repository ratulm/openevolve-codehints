# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 24
    # Score rose monotonically with edge exponent (pure profile, no blend):
    # 0.5/0.75/1.0/1.4/1.8/2.5 -> .267/.294/.324/.355/.395/.415.
    # Continue the trend, loosening the cap in step so edges aren't flattened.
    x = (np.arange(m) + 0.5) / m * 2.0 - 1.0          # bin centres in (-1, 1)
    arc = (1.0 - x * x) ** -3.0
    arc = np.minimum(arc, 12.0 * arc.mean())           # keep end bins bounded
    return g(arc / arc.sum())                          # total mass = 1


def g(h):
    return np.clip(np.asarray(h, dtype=float), -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
