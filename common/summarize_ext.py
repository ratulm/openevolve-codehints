"""Blind-arm extension to 40 iterations.

Joins iterations 1-20 from runs/<problem>/baseline_s<k>.log with iterations
21-40 from runs_ext40/<problem>/baseline_s<k>.log, and compares with the
hinted arm at 20 iterations (runs/, unchanged).
"""
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
from summarize import problem_constants, trajectory


def best_curve(init, scores, upto):
    b, last, curve = init, 0, {}
    for it in range(1, upto + 1):
        v = scores.get(it)
        if v is not None and v > b + 1e-4:
            b, last = v, it
        curve[it] = b
    return curve, last


def main():
    print("%-24s %-4s %6s %6s %6s %6s | %-9s | %s" % (
        "problem", "seed", "@20", "@30", "@40", "gain", "last gain", "iters 21-40 done"))
    rows = []
    for d in sorted(H.glob("p[0-9][0-9]_*")):
        if not (H / "runs_ext40" / d.name).exists():
            continue
        init, _ = problem_constants(d)
        hinted = []
        for log in sorted((H / "runs" / d.name).glob("hinted_s*.log")):
            hinted.append(best_curve(init, trajectory(log), 20)[0][20])
        b20, b40 = [], []
        for k in (1, 2, 3):
            first = H / "runs" / d.name / f"baseline_s{k}.log"
            ext = H / "runs_ext40" / d.name / f"baseline_s{k}.log"
            sc = trajectory(first)
            sc = {i: v for i, v in sc.items() if i <= 20}
            late = {i: v for i, v in trajectory(ext).items() if 21 <= i <= 40} if ext.exists() else {}
            sc.update(late)
            curve, last = best_curve(init, sc, 40)
            b20.append(curve[20]); b40.append(curve[40])
            print("%-24s s%-3d %6.3f %6.3f %6.3f %+6.3f | iter %-4s | %d" % (
                d.name, k, curve[20], curve[30], curve[40], curve[40] - curve[20], last or "-", len(late)))
        rows.append((d.name, np.mean(b20), np.mean(b40), np.mean(hinted)))
    print("\n%-24s %8s %8s %10s %12s" % ("problem", "blind@20", "blind@40", "hinted@20", "gap closed"))
    for name, a, b, h in rows:
        closed = (b - a) / (h - a) if h > a else float("nan")
        print("%-24s %8.3f %8.3f %10.3f %11.0f%%" % (name, a, b, h, 100 * closed))
    if len(rows) >= 5:
        from scipy.stats import wilcoxon
        diffs = [h - b for _, _, b, h in rows]
        print("\nhinted@20 minus blind@40: mean %+.3f, median %+.3f, hinted better in %d/%d, Wilcoxon p = %.4f" % (
            np.mean(diffs), np.median(diffs), sum(x > 0 for x in diffs), len(diffs), wilcoxon(diffs).pvalue))


if __name__ == "__main__":
    main()
