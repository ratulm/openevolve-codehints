"""Summarise hints-pilot runs from their logs.

Usage: analyze.py <runs_dir> [threshold ...]
Reads <arm>_s<seed>.log files and prints, per run, the best-so-far
combined_score after selected iterations, the first iteration reaching each
threshold, and what the final best program looks like.
"""
import re
import sys
from pathlib import Path

ITER = re.compile(r"Iteration (\d+): Program")
SCORE = re.compile(r"combined_score=([0-9.]+)")
CHECKPOINTS = (1, 2, 3, 5, 10, 20)


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


def main():
    runs = Path(sys.argv[1])
    thresholds = [float(t) for t in sys.argv[2:]] or [0.9]
    head = "%-13s %4s " % ("run", "n") + " ".join("best@%-3d" % c for c in CHECKPOINTS)
    head += " " + " ".join("iter>=%-5g" % t for t in thresholds) + "  zeros  final program"
    print(head)
    for log in sorted(runs.glob("*_s*.log")):
        sc = trajectory(log)
        if not sc:
            print("%-13s no completed iterations" % log.stem)
            continue
        row = "%-13s %4d " % (log.stem, len(sc))
        for c in CHECKPOINTS:
            done = [v for k, v in sc.items() if k <= c]
            row += " %-8s" % ("%.4f" % max(done) if done else "-")
        for t in thresholds:
            hit = [k for k, v in sorted(sc.items()) if v >= t]
            row += " %-10s" % (hit[0] if hit else "never")
        row += "  %5d" % sum(1 for v in sc.values() if v == 0.0)
        best = runs / log.stem / "best" / "best_program.py"
        if best.exists():
            code = best.read_text()
            tags = [name for name, pat in (("slsqp", "SLSQP"), ("scipy", "scipy"), ("hint-text", "HINT"))
                    if pat in code]
            row += "  %d lines; %s" % (code.count("\n"), ",".join(tags) or "-")
        print(row)


if __name__ == "__main__":
    main()
