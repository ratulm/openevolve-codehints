# EVOLVE-BLOCK-START
import numpy as np


def construct_packing():
    n = 26
    p = np.zeros((n, 2))

    p[0] = [0.5, 0.5]

    for i in range(8):
        t = 2 * np.pi * i / 8
        p[i + 1] = [0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)]

    for i in range(16):
        t = 2 * np.pi * i / 16
        p[i + 9] = [0.5 + 0.7 * np.cos(t), 0.5 + 0.7 * np.sin(t)]

    p = np.clip(p, 0.01, 0.99)

    q = helper(p)

    s = np.sum(q)

    return p, q, s


def helper(p):
    n = p.shape[0]
    q = np.ones(n)

    for i in range(n):
        u, v = p[i]
        q[i] = min(u, v, 1 - u, 1 - v)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((p[i] - p[j]) ** 2))

            if q[i] + q[j] > d:
                k = d / (q[i] + q[j])
                q[i] *= k
                q[j] *= k

    return q


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(sum_radii)
