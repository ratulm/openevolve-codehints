"""All arms at every budget: NoHints and NoHints+notice at 10/20/40/80
iterations, Hints at 10/20. Per-problem mean over 3 seeds; % improvement over
the starting program; median/mean across problems; paired tests.

Segments joined per run:
  NoHints:        runs/ (1-20) + runs_ext40/ (21-40) + runs_ext80/ (41-80)
  NoHints+notice: runs_notice/ (1-10) + runs_notice_ext80/ (11-80)
  Hints:          runs/ (1-20)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
from summarize import problem_constants, trajectory

SEGMENTS = {
    "NoHints": [("runs", 1, 20), ("runs_ext40", 21, 40), ("runs_ext80", 41, 80)],
    "NoHints+notice": [("runs_notice", 1, 10), ("runs_notice_ext80", 11, 80)],
    "Hints": [("runs", 1, 20)],
}
ARM_FILE = {"NoHints": "baseline", "NoHints+notice": "baseline", "Hints": "hinted"}
COLUMNS = [("NoHints", 10), ("NoHints", 20), ("NoHints", 40), ("NoHints", 80),
           ("NoHints+notice", 10), ("NoHints+notice", 20), ("NoHints+notice", 40), ("NoHints+notice", 80),
           ("Hints", 10), ("Hints", 20)]


def joined(arm, problem, seed):
    sc, done = {}, 0
    for root, lo, hi in SEGMENTS[arm]:
        log = H / root / problem / f"{ARM_FILE[arm]}_s{seed}.log"
        if log.exists():
            part = {i: v for i, v in trajectory(log).items() if lo <= i <= hi}
            sc.update(part)
            done += len(part)
    return sc, done


def table():
    names, scores, pct, done = [], [], [], {}
    for d in sorted(H.glob("p[0-9][0-9]_*")):
        init, _ = problem_constants(d)
        runs = {arm: [joined(arm, d.name, k) for k in (1, 2, 3)] for arm in SEGMENTS}
        for arm, rs in runs.items():
            done.setdefault(arm, []).extend(n for _, n in rs)
        row = [np.mean([max([init] + [v for i, v in sc.items() if i <= n]) for sc, _ in runs[arm]]) for arm, n in COLUMNS]
        names.append(d.name); scores.append(row); pct.append([100 * (x - init) / init for x in row])
    return names, np.array(scores), np.array(pct), done


def main():
    names, scores, pct, done = table()
    head = ["%s@%d" % (a.replace("NoHints+notice", "Notice"), n) for a, n in COLUMNS]
    print("Mean best score over 3 seeds (1.0 = reference)\n%-24s" % "problem" + "".join("%11s" % h for h in head))
    for name, row in zip(names, scores):
        print("%-24s" % name + "".join("%11.3f" % x for x in row))
    print("\n%% improvement over start\n%-24s" % "" + "".join("%11s" % h for h in head))
    print("%-24s" % "median" + "".join("%10.0f%%" % x for x in np.median(pct, axis=0)))
    print("%-24s" % "mean" + "".join("%10.0f%%" % x for x in pct.mean(axis=0)))
    print("\nNotice vs NoHints at the same budget:")
    for n in (10, 20, 40, 80):
        a, b = pct[:, COLUMNS.index(("NoHints+notice", n))], pct[:, COLUMNS.index(("NoHints", n))]
        d = a - b
        p = wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-9) else float("nan")
        print("  @%-3d notice better %d, worse %d, tied %d; Wilcoxon p = %.3f" % (
            n, (d > 0.5).sum(), (d < -0.5).sum(), (abs(d) <= 0.5).sum(), p))
    print("Hints vs Notice:")
    for h in (10, 20):
        for n in (10, 20, 40, 80):
            d = pct[:, COLUMNS.index(("Hints", h))] - pct[:, COLUMNS.index(("NoHints+notice", n))]
            print("  Hints@%d vs Notice@%-3d hints better %d/10; Wilcoxon p = %.4f" % (h, n, (d > 0).sum(), wilcoxon(d).pvalue))
    print("\niterations logged per run (min-max): " + ", ".join("%s %d-%d" % (a, min(v), max(v)) for a, v in done.items()))


if __name__ == "__main__":
    main()
