"""Offline solver used by reference.py: simulated annealing plus exhaustive
swap/insert descent for single-machine total weighted tardiness with an
optional sequence-dependent setup matrix."""
import math
import random


def cost(order, proc, due, weight, setup):
    t = 0
    total = 0
    prev = -1
    for j in order:
        if prev >= 0:
            t += setup[prev][j]
        t += proc[j]
        late = t - due[j]
        if late > 0:
            total += weight[j] * late
        prev = j
    return total


def descend(order, proc, due, weight, setup):
    n = len(order)
    best = cost(order, proc, due, weight, setup)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                cand = order[:]
                cand.insert(k, cand.pop(i))
                c = cost(cand, proc, due, weight, setup)
                if c < best:
                    order, best, improved = cand, c, True
                if k > i:
                    cand = order[:]
                    cand[i], cand[k] = cand[k], cand[i]
                    c = cost(cand, proc, due, weight, setup)
                    if c < best:
                        order, best, improved = cand, c, True
    return order, best


def anneal(start, proc, due, weight, setup, rng, steps, t0, t1):
    n = len(start)
    cur = start[:]
    cur_c = cost(cur, proc, due, weight, setup)
    best, best_c = cur[:], cur_c
    for s in range(steps):
        temp = t0 * (t1 / t0) ** (s / steps)
        cand = cur[:]
        r = rng.random()
        i, k = rng.randrange(n), rng.randrange(n)
        if r < 0.4:
            cand.insert(k, cand.pop(i))
        elif r < 0.7:
            cand[i], cand[k] = cand[k], cand[i]
        else:
            length = rng.randrange(1, 6)
            i = rng.randrange(n - length + 1)
            block = cand[i:i + length]
            del cand[i:i + length]
            k = rng.randrange(len(cand) + 1)
            cand[k:k] = block
        c = cost(cand, proc, due, weight, setup)
        if c <= cur_c or rng.random() < math.exp((cur_c - c) / temp):
            cur, cur_c = cand, c
            if c < best_c:
                best, best_c = cand[:], c
    return best, best_c


def solve(proc, due, weight, setup, restarts, steps, seed):
    n = len(proc)
    rng = random.Random(seed)
    best, best_c = None, float("inf")
    results = []
    for r in range(restarts):
        if r % 2 == 0:
            start = sorted(range(n), key=lambda j: due[j])
        else:
            start = list(range(n))
            rng.shuffle(start)
        order, c = anneal(start, proc, due, weight, setup, rng, steps, 60.0, 0.2)
        order, c = descend(order, proc, due, weight, setup)
        results.append(c)
        if c < best_c:
            best, best_c = order, c
    return best, best_c, results
