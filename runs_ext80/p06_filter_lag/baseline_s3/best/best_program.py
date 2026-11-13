# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 24
    # Arcsine/flat blend; scores keep rising with more edge weight:
    # 3.0->.3613, 3.6->.3917, 4.5->.4354, 5.0->.4569 (~+0.045/unit, linear).
    # Max normalized value ~0.45, min ~-0.05: far inside the [-2, 2] clip.
    edge = 7.0
    x = (np.arange(m) + 0.5) / m
    arc = 1.0 / np.sqrt(x * (1.0 - x))
    arc /= arc.mean()
    w = (1.0 - edge) + edge * arc
    return g(w / w.sum())


def g(h):
    # Same clipping to [-2, 2], done in one numpy call
    return np.clip(np.asarray(h, dtype=float), -2.0, 2.0)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run():
    h = f()
    return h
