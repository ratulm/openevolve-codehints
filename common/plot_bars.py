"""Grouped bar chart of % improvement over the starting program, per problem.

Each series is (label, arm, iterations, color). A bar is the mean over 3 seeds of
100 * (best score within the first N iterations - starting score) / starting score.
Blind-arm runs are joined across runs/ (1-20), runs_ext40/ (21-40) and
runs_ext80/ (41-80). The legend gives each series' average over the 10 problems.

Usage: plot_bars.py <output_stem>   (series are configured in SERIES_SETS below)
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
from plot_improvement import LABELS, SURFACE, TEXT1, TEXT2, GRID
from summarize import problem_constants, trajectory
from summarize_ext80 import blind_scores

FIVE_ARMS = [
    ("NoHints (10 iters)", "baseline", 10, "#86b6ef"),
    ("NoHints (20 iters)", "baseline", 20, "#5598e7"),
    ("NoHints (40 iters)", "baseline", 40, "#2a78d6"),
    ("NoHints (80 iters)", "baseline", 80, "#184f95"),
    ("Hints (10 iters)", "hinted", 10, "#eb6834"),
]

SIX_ARMS = FIVE_ARMS[:4] + [
    ("Hints (10 iters)", "hinted", 10, "#f0956b"),
    ("Hints (20 iters)", "hinted", 20, "#c94f1d"),
]

# legend statistic per set: "average" or "median" (across the 10 problems)
LEGEND_STAT = {"improvement_budgets": "average", "per_problem_five_arms": "median", "per_problem_six_arms": "median"}

SERIES_SETS = {
    "per_problem_five_arms": FIVE_ARMS,
    "per_problem_six_arms": SIX_ARMS,
    "improvement_budgets": [
        ("NoHints (10 iters)", "baseline", 10, "#86b6ef"),
        ("NoHints (20 iters)", "baseline", 20, "#3987e5"),
        ("NoHints (40 iters)", "baseline", 40, "#184f95"),
        ("Hints (10 iters)", "hinted", 10, "#eb6834"),
    ],
}


def best_within(init, scores, n):
    return max([init] + [v for i, v in scores.items() if i <= n])


def compute(series):
    names, means, seeds = [], [], []
    for d in sorted(H.glob("p[0-9][0-9]_*")):
        init, _ = problem_constants(d)
        row_m, row_s = [], []
        for _, arm, n, _ in series:
            vals = []
            for k in (1, 2, 3):
                sc = blind_scores(d.name, k)[0] if arm == "baseline" else trajectory(H / "runs" / d.name / f"hinted_s{k}.log")
                vals.append(100.0 * (best_within(init, sc, n) - init) / init)
            row_m.append(np.mean(vals)); row_s.append(vals)
        names.append(d.name); means.append(row_m); seeds.append(row_s)
    return names, np.array(means), seeds


def main(stem):
    series = SERIES_SETS[stem]
    names, means, seeds = compute(series)
    avg, med = means.mean(axis=0), np.median(means, axis=0)

    with open(H / "figures" / f"{stem}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["problem"] + [f"{s[0]} mean %" for s in series] + [f"{s[0]} seeds %" for s in series])
        for name, m, sv in zip(names, means, seeds):
            wr.writerow([name] + [round(x, 1) for x in m] + [" ".join("%.1f" % x for x in v) for v in sv])
        wr.writerow(["AVERAGE"] + [round(x, 1) for x in avg])
        wr.writerow(["MEDIAN"] + [round(x, 1) for x in med])

    k = len(series)
    fig, ax = plt.subplots(figsize=(15 if k > 5 else 14 if k > 4 else 13, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    gap = 0.02
    w = (0.8 - gap * (k - 1)) / k
    ymax = means.max() * 1.08
    for i in range(len(names)):
        x0 = i - 0.4
        for j, (_, _, _, color) in enumerate(series):
            x = x0 + j * (w + gap)
            m = means[i, j]
            ax.add_patch(plt.Rectangle((x, 0), w, m, linewidth=0, facecolor=color, zorder=2))
            ax.text(x + w / 2, m + ymax * 0.01, "%d" % round(m), ha="center", va="bottom", fontsize=5.5 if k > 5 else 6 if k > 4 else 6.5, color=TEXT1)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABELS[n] for n in names], fontsize=9, color=TEXT1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d%%" % v))
    ax.tick_params(axis="y", labelsize=9, colors=TEXT2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylabel("Improvement over starting program (%)", fontsize=9, color=TEXT2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=s[3]) for s in series]
    stat = LEGEND_STAT.get(stem, "average")
    vals = avg if stat == "average" else med
    ax.legend(handles, ["%s, %s %d%%" % (s[0], stat, round(a)) for s, a in zip(series, vals)],
              loc="upper left", bbox_to_anchor=(0, 1.07), ncol=k, frameon=False, fontsize=8 if k > 5 else 8.5 if k > 4 else 9,
              labelcolor=TEXT1, handlelength=1.0, handleheight=1.0, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(H / "figures" / f"{stem}.png", facecolor=SURFACE)
    print("wrote", H / "figures" / f"{stem}.png")
    for s, a, m in zip(series, avg, med):
        print("  %-20s average %4.0f%%  median %4.0f%%" % (s[0], a, m))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "improvement_budgets")
