"""Cross-problem summary of the hints study.

Usage: summarize.py [--iters 20]
For every problem with runs under runs/<problem>/, prints per-arm best scores
(best-so-far, floored at the initial program's score), the hint effect, the
"standard guess" ceiling, and the share of invalid (score 0) children. Then a
paired Wilcoxon test across problems on the per-problem mean difference.
"""
import importlib.util
import pickle
import re
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
import harness

ITER = re.compile(r"Iteration (\d+): Program")
SCORE = re.compile(r"combined_score=([0-9.]+)")


def trajectory(log):
    scores, pending = {}, None
    for line in log.read_text(errors="replace").splitlines():
        m = ITER.search(line)
        if m:
            pending = int(m.group(1))
        elif pending is not None and "Metrics:" in line:
            s = SCORE.search(line)
            scores[pending] = float(s.group(1)) if s else 0.0
            pending = None
    return scores


def problem_constants(d):
    spec = importlib.util.spec_from_file_location("task_" + d.name, d / "task.py")
    task = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(d))
    spec.loader.exec_module(task)
    sys.path.pop(0)
    sys.modules.pop("task", None)
    out, _ = harness.run_program(str(d / "baseline" / "initial_program.py"), task.TIME_LIMIT_S)
    init = task.score(out)[0]
    std = task.score(pickle.load(open(d / "standard_output.pkl", "rb")))[0]
    return init, std


def main():
    at = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 20
    rows, diffs = [], []
    print(f"best-so-far after iteration {at}; 1.0 = offline reference\n")
    print("%-24s %5s %5s | %-26s %-26s | %6s | %-11s | %s" % (
        "problem", "start", "std", "blind (seeds)", "hinted (seeds)", "effect", "invalid b/h", "blind>std") + " | iters done")
    for d in sorted(H.glob("p[0-9][0-9]_*")):
        runs = H / "runs" / d.name
        if not runs.exists():
            continue
        init, std = problem_constants(d)
        best, zeros, total, done = {}, {}, {}, []
        for arm in ("baseline", "hinted"):
            best[arm], zeros[arm], total[arm] = [], 0, 0
            for log in sorted(runs.glob(f"{arm}_s*.log")):
                sc = {k: v for k, v in trajectory(log).items() if k <= at}
                if not sc:
                    continue
                best[arm].append(max([init] + list(sc.values())))
                zeros[arm] += sum(1 for v in sc.values() if v == 0.0)
                total[arm] += len(sc)
                done.append(len(trajectory(log)))
        if not best["baseline"] or not best["hinted"]:
            continue
        fmt = lambda xs: " ".join("%.3f" % x for x in xs)
        diff = float(np.mean(best["hinted"]) - np.mean(best["baseline"]))
        if len(done) == 6 and min(done) >= 19:
            diffs.append(diff)
        inv = "%3.0f%%/%3.0f%%" % (100 * zeros["baseline"] / max(total["baseline"], 1),
                                   100 * zeros["hinted"] / max(total["hinted"], 1))
        beat = sum(1 for x in best["baseline"] if x > std + 0.02)
        print("%-24s %5.3f %5.3f | %-26s %-26s | %+6.3f | %-11s | %d/%d" % (
            d.name, init, std, fmt(best["baseline"]), fmt(best["hinted"]), diff, inv, beat, len(best["baseline"]))
              + " | %s%d-%d of 20, %d/6 runs" % ("" if len(done) == 6 and min(done) >= 19 else "PARTIAL ", min(done), max(done), len(done)))
    if len(diffs) >= 5:
        from scipy.stats import wilcoxon
        w = wilcoxon(diffs)
        print("\nproblems: %d | mean effect %+.3f | median %+.3f | hinted better in %d | Wilcoxon p = %.4f" % (
            len(diffs), np.mean(diffs), np.median(diffs), sum(d > 0 for d in diffs), w.pvalue))


if __name__ == "__main__":
    main()
