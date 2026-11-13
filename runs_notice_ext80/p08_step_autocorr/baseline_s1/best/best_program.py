# EVOLVE-BLOCK-START
import numpy as np


def _score(a):
    s = np.sum(a)
    if s <= 0:
        return np.inf
    c = np.convolve(a, a)
    return np.max(c) * 2.0 * a.shape[0] / (s * s)


def _fconv(p, q):
    L = p.shape[0] + q.shape[0] - 1
    N = 1 << (L - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(p, N) * np.fft.rfft(q, N), N)[:L]


def _optimize(x, iters=4000, lr=0.02):
    n = x.shape[0]
    m = np.zeros(n)
    v = np.zeros(n)
    b1, b2, eps = 0.9, 0.999, 1e-12
    best_x = x.copy()
    best = _score(x * x)
    for t in range(1, iters + 1):
        beta = 20.0 * (100.0 ** (t / iters))
        a = x * x
        s = np.sum(a)
        c = _fconv(a, a)
        r = c * 2.0 * n / (s * s)
        w = np.exp(beta * (r - np.max(r)))
        w /= np.sum(w)
        corr = _fconv(w, a[::-1])[n - 1:2 * n - 1]
        ga = (2.0 * n / (s * s)) * (2.0 * corr
                                    - 2.0 * np.sum(w * c) / s)
        gx = 2.0 * x * ga
        m = b1 * m + (1 - b1) * gx
        v = b2 * v + (1 - b2) * gx * gx
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
        x = x / np.sqrt(np.mean(x * x) + 1e-18)
        sc = _score(x * x)
        if sc < best:
            best = sc
            best_x = x.copy()
    return best_x, best


def f():
    n = 50
    a = np.zeros(n)

    for i in range(n):
        a[i] = 1.0 + i

    best_a = a.copy()
    best = _score(a)

    rng = np.random.default_rng(12345)
    starts = []
    for _ in range(12):
        starts.append(np.sqrt(rng.uniform(0.2, 1.0, n)))
    xs = (2.0 * (np.arange(n) + 0.5) / n) - 1.0
    starts.append(np.sqrt((1.0 - xs * xs) ** (-0.4)))
    for x0 in starts:
        x, sc = _optimize(x0)
        if sc < best:
            best = sc
            best_a = x * x

    a = g(best_a)

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
