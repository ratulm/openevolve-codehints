"""Obfuscation-notice arm vs the original blind and hinted arms, at 10 iterations.

NoHints@10 and Hints@10 come from the first 10 iterations of runs/; the notice
arm is runs_notice/ (fresh 10-iteration runs, same seeds, same blind programs,
only the system message differs).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
from summarize import problem_constants, trajectory


def best(init, log, n=10):
    sc = trajectory(log) if log.exists() else {}
    return max([init] + [v for i, v in sc.items() if i <= n]), sum(1 for i, v in sc.items() if i <= n and v == 0.0), \
        sum(1 for i in sc if i <= n)


def main():
    arms = [("NoHints", "runs", "baseline"), ("NoHints+notice", "runs_notice", "baseline"), ("Hints", "runs", "hinted")]
    print("%-24s %6s | %-20s | %-20s | %-20s" % ("problem", "start", *[a[0] + " @10" for a in arms]))
    pct = {a[0]: [] for a in arms}
    inv = {a[0]: [0, 0] for a in arms}
    for d in sorted(H.glob("p[0-9][0-9]_*")):
        init, _ = problem_constants(d)
        cells = []
        for name, root, arm in arms:
            vals = []
            for k in (1, 2, 3):
                b, z, n = best(init, H / root / d.name / f"{arm}_s{k}.log")
                vals.append(b); inv[name][0] += z; inv[name][1] += n
            pct[name].append(100 * (np.mean(vals) - init) / init)
            cells.append("%.3f %.3f %.3f" % tuple(vals))
        print("%-24s %6.3f | %-20s | %-20s | %-20s" % (d.name, init, *cells))
    print()
    for name, *_ in arms:
        v = np.array(pct[name])
        print("%-15s %% improvement over start: median %4.0f%%  mean %4.0f%%  | invalid children %d/%d" % (
            name, np.median(v), v.mean(), *inv[name]))
    d = np.array(pct["NoHints+notice"]) - np.array(pct["NoHints"])
    print("\nnotice minus plain NoHints: better on %d, worse on %d, tied %d; Wilcoxon p = %s" % (
        (d > 0.5).sum(), (d < -0.5).sum(), (abs(d) <= 0.5).sum(),
        "%.3f" % wilcoxon(d).pvalue if np.any(d != 0) else "n/a"))
    d2 = np.array(pct["Hints"]) - np.array(pct["NoHints+notice"])
    print("Hints minus notice: Hints better on %d/10; Wilcoxon p = %.4f" % ((d2 > 0).sum(), wilcoxon(d2).pvalue))


if __name__ == "__main__":
    main()
