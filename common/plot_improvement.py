"""Bar chart: improvement over the starting program, per problem.

Bars: NoHints (20 iters), NoHints (40 iters), Hints (20 iters); each bar is the
mean over 3 seeds of 100 * (best score so far - starting score) / starting score.
The legend shows each arm's average over the 10 problems.
Writes figures/improvement.png and .csv (table view).
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import FancyBboxPatch

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
from summarize import problem_constants, trajectory

LABELS = {
    "p01_circles_lshape": "Circles\nL-shape", "p02_heilbronn_trapezoid": "Heilbronn\ntrapezoid",
    "p03_dispersion_annulus": "Dispersion\nannulus", "p04_flat_basin": "Flat\nbasin",
    "p05_tour_turns": "Tour\nturns", "p06_filter_lag": "Filter\nlag", "p07_sumset_ratio": "Sumset\nratio",
    "p08_step_autocorr": "Step\nautocorr", "p09_scheduling": "Scheduling\nsetups", "p10_curve_fit": "Curve\nfit",
}
SERIES = [("NoHints (20 iters)", "#2a78d6"), ("NoHints (40 iters)", "#eb6834"), ("Hints (20 iters)", "#1baf7a")]
SURFACE, TEXT1, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
YMAX = 1.0


def best(init, scores, upto):
    b = init
    for it in range(1, upto + 1):
        v = scores.get(it)
        if v is not None and v > b:
            b = v
    return b


def collect():
    rows = []
    for d in sorted(H.glob("p[0-9][0-9]_*")):
        init, _ = problem_constants(d)
        vals = [[], [], []]
        for k in (1, 2, 3):
            b = {i: v for i, v in trajectory(H / "runs" / d.name / f"baseline_s{k}.log").items() if i <= 20}
            vals[0].append(best(init, b, 20) - init)
            b.update({i: v for i, v in trajectory(H / "runs_ext40" / d.name / f"baseline_s{k}.log").items() if 21 <= i <= 40})
            vals[1].append(best(init, b, 40) - init)
            h = trajectory(H / "runs" / d.name / f"hinted_s{k}.log")
            vals[2].append(best(init, h, 20) - init)
        rows.append((d.name, init, vals))
    return rows


def rounded_bar(ax, x, w, h, color):
    # Square at the baseline, 4px-ish rounded data end: a rounded box plus a square patch over its base.
    r = min(0.012, h / 2) if h > 0 else 0
    ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=YMAX / 1.0 * 6, linewidth=0, facecolor=color, zorder=2))
    ax.add_patch(plt.Rectangle((x, 0), w, min(h, 0.03), linewidth=0, facecolor=color, zorder=2))


def main():
    rows = collect()
    # percentage improvement over the starting program, per seed, then mean over seeds
    pct = [(name, [[100.0 * v / init for v in vals[j]] for j in range(3)]) for name, init, vals in rows]
    means = np.array([[np.mean(p[j]) for j in range(3)] for _, p in pct])
    avg = means.mean(axis=0)

    with open(H / "figures" / "improvement.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["problem", "start_score"] + [f"{s} mean % improvement" for s, _ in SERIES]
                    + [f"{s} seeds %" for s, _ in SERIES])
        for (name, init, _), (_, p), m in zip(rows, pct, means):
            wr.writerow([name, round(init, 4)] + [round(x, 1) for x in m] + [" ".join("%.1f" % x for x in v) for v in p])
        wr.writerow(["AVERAGE over problems", ""] + [round(x, 1) for x in avg])

    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n, w, gap = len(rows), 0.24, 0.025
    ymax = means.max() * 1.08
    for i in range(n):
        for j, (_, color) in enumerate(SERIES):
            x = i - 1.5 * w - gap + j * (w + gap)
            m = means[i, j]
            ax.add_patch(plt.Rectangle((x, 0), w, m, linewidth=0, facecolor=color, zorder=2))
            ax.text(x + w / 2, m + ymax * 0.01, "%d%%" % round(m), ha="center", va="bottom", fontsize=7, color=TEXT1)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(n))
    ax.set_xticklabels([LABELS[r[0]] for r in rows], fontsize=9, color=TEXT1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d%%" % v))
    ax.tick_params(axis="y", labelsize=9, colors=TEXT2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylabel("Improvement over starting program", fontsize=9, color=TEXT2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in SERIES]
    ax.legend(handles, ["%s, average %d%%" % (label, round(a)) for (label, _), a in zip(SERIES, avg)],
              loc="upper left", bbox_to_anchor=(0, 1.07), ncol=3, frameon=False, fontsize=9,
              labelcolor=TEXT1, handlelength=1.0, handleheight=1.0, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(H / "figures" / "improvement.png", facecolor=SURFACE)
    print("wrote", H / "figures" / "improvement.png")
    print("averages:", dict(zip([s for s, _ in SERIES], np.round(avg, 1))))


if __name__ == "__main__":
    main()
