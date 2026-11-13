"""Blind-arm extension to 80 iterations.

Joins iterations 1-20 (runs/), 21-40 (runs_ext40/) and 41-80 (runs_ext80/) of
each blind run and compares with the hinted arm at 20 iterations.
"""
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
from summarize import problem_constants, trajectory

SEGMENTS = [("runs", 1, 20), ("runs_ext40", 21, 40), ("runs_ext80", 41, 80)]


def blind_scores(problem, seed):
    sc, done = {}, {}
    for d, lo, hi in SEGMENTS:
        log = H / d / problem / f"baseline_s{seed}.log"
        part = {i: v for i, v in trajectory(log).items() if lo <= i <= hi} if log.exists() else {}
        sc.update(part)
        done[d] = len(part)
    return sc, done


def curve(init, sc, upto):
    b, last, out = init, 0, {}
    for it in range(1, upto + 1):
        v = sc.get(it)
        if v is not None and v > b + 1e-4:
            b, last = v, it
        out[it] = b
    return out, last


def main():
    print("%-24s %-4s %6s %6s %6s %6s | %-9s | %s" % ("problem", "seed", "@20", "@40", "@60", "@80", "last gain", "iters 41-80 done"))
    rows = []
    for d in sorted(H.glob("p[0-9][0-9]_*")):
        init, _ = problem_constants(d)
        hinted = [curve(init, trajectory(H / "runs" / d.name / f"hinted_s{k}.log"), 20)[0][20] for k in (1, 2, 3)]
        per = {20: [], 40: [], 80: []}
        for k in (1, 2, 3):
            sc, done = blind_scores(d.name, k)
            c, last = curve(init, sc, 80)
            for t in per:
                per[t].append(c[t])
            print("%-24s s%-3d %6.3f %6.3f %6.3f %6.3f | iter %-4s | %d" % (
                d.name, k, c[20], c[40], c[60], c[80], last or "-", done["runs_ext80"]))
        rows.append((d.name, init, {t: np.mean(v) for t, v in per.items()}, np.mean(hinted)))
    print("\n%-24s %8s %8s %8s %10s %16s" % ("problem", "blind@20", "blind@40", "blind@80", "hinted@20", "gap closed by 80"))
    for name, init, b, h in rows:
        closed = (b[80] - b[20]) / (h - b[20]) if h > b[20] else float("nan")
        print("%-24s %8.3f %8.3f %8.3f %10.3f %15.0f%%" % (name, b[20], b[40], b[80], h, 100 * closed))
    from scipy.stats import wilcoxon
    diffs = [h - b[80] for _, _, b, h in rows]
    print("\nhinted@20 minus blind@80: mean %+.3f, median %+.3f, hinted better in %d/%d, Wilcoxon p = %.4f" % (
        np.mean(diffs), np.median(diffs), sum(x > 0 for x in diffs), len(diffs), wilcoxon(diffs).pvalue))
    for t in (20, 40, 80):
        pct = [100 * (b[t] - init) / init for _, init, b, _ in rows]
        print("blind@%d %% improvement: mean %.0f%%, median %.0f%%" % (t, np.mean(pct), np.median(pct)))
    pct = [100 * (h - init) / init for _, init, _, h in rows]
    print("hinted@20 %% improvement: mean %.0f%%, median %.0f%%" % (np.mean(pct), np.median(pct)))


if __name__ == "__main__":
    main()
