"""Offline solver for the turn-penalised tour instance (not shown to the LLM).

Simulated annealing over closed tours with segment-reversal and segment-move
steps, followed by a greedy descent. Used with the true cost for the reference
and with weight 0 (plain length) for the standard guess.
"""
import numpy as np


def make_points():
    rng = np.random.default_rng(20250511)
    pts = []
    while len(pts) < 46:
        c = rng.integers(3, 98, 2)
        if all(np.hypot(*(c - q)) >= 6 for q in pts):
            pts.append(c)
    return np.array(pts, dtype=float)


def parts(pts, order):
    """Return (length, turning angles in radians) of the closed tour."""
    p = pts[np.asarray(order)]
    d = np.roll(p, -1, axis=0) - p          # d[i] = leg leaving vertex i
    e = np.roll(d, 1, axis=0)               # leg arriving at vertex i
    ang = np.arctan2(np.abs(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]), np.sum(e * d, axis=1))
    return float(np.hypot(d[:, 0], d[:, 1]).sum()), ang


def cost(pts, order, w, pw):
    length, ang = parts(pts, order)
    return length + w * float(np.sum(ang ** pw))


def neighbour(t, rng):
    n = len(t)
    i, j = sorted(rng.integers(0, n, 2))
    if i == j:
        return None
    if rng.random() < 0.5:
        return np.concatenate([t[:i], t[i:j + 1][::-1], t[j + 1:]])
    m = min(j - i, int(rng.integers(1, 4)))
    seg = t[i:i + m]
    rest = np.concatenate([t[:i], t[i + m:]])
    k = int(rng.integers(0, len(rest) + 1))
    if rng.random() < 0.5:
        seg = seg[::-1]
    return np.concatenate([rest[:k], seg, rest[k:]])


def anneal(pts, w, pw, seed, steps=150000, t0=30.0, t1=0.02):
    rng = np.random.default_rng(seed)
    t = rng.permutation(len(pts))
    c = cost(pts, t, w, pw)
    best, bc = t, c
    for s in range(steps):
        temp = t0 * (t1 / t0) ** (s / steps)
        u = neighbour(t, rng)
        if u is None:
            continue
        cu = cost(pts, u, w, pw)
        if cu < c or rng.random() < np.exp((c - cu) / temp):
            t, c = u, cu
            if c < bc:
                best, bc = t, c
    return descend(pts, best, w, pw)


def descend(pts, t, w, pw):
    """Exhaustive first-improvement over all reversals and short segment moves."""
    n = len(t)
    c = cost(pts, t, w, pw)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                u = np.concatenate([t[:i], t[i:j + 1][::-1], t[j + 1:]])
                cu = cost(pts, u, w, pw)
                if cu < c - 1e-9:
                    t, c, improved = u, cu, True
        for m in (1, 2, 3):
            for i in range(n - m + 1):
                seg = t[i:i + m]
                rest = np.concatenate([t[:i], t[i + m:]])
                for k in range(len(rest) + 1):
                    for sg in (seg, seg[::-1]):
                        u = np.concatenate([rest[:k], sg, rest[k:]])
                        cu = cost(pts, u, w, pw)
                        if cu < c - 1e-9:
                            t, c, improved = u, cu, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return t, c


def solve(pts, w, pw, seeds, steps=150000):
    return min((anneal(pts, w, pw, s, steps) for s in seeds), key=lambda r: r[1])
