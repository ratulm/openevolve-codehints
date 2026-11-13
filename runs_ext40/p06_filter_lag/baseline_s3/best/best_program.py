# EVOLVE-BLOCK-START
import numpy as np


def f():
    m = 24
    # Arcsine/flat blend; scores keep rising with more edge weight:
    # 0.35->0.2414, 0.55->0.2491, 0.75->0.2570, 1.0->0.2673, 1.2->0.2758.
    # Push further past pure arcsine (weights stay positive until edge ~3).
    edge = 1.6
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
