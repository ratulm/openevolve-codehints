"""Overall chart with three groups: NoHints, NoHints + obfuscation notice, Hints.
Bar = median across the 10 problems of the per-problem mean (3 seeds) % improvement
over the starting program. Writes figures/overall_median_notice.png and .csv."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np

H = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(H / "common"))
from plot_improvement import SURFACE, TEXT1, TEXT2, GRID
from summarize_budgets import COLUMNS, table

GROUPS = [
    ("Code without hints\n(comments, variable names)", "NoHints", ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]),
    ("Code without hints,\nprompt says code is obfuscated", "NoHints+notice", ["#b4aaf0", "#9286e8", "#6f60db", "#4e3fb4"]),
    ("Code with hints", "Hints", ["#f0956b", "#c94f1d"]),
]


def main():
    _, _, pct, _ = table()
    med = dict(zip(COLUMNS, np.median(pct, axis=0)))
    bars, xs, x = [], [], 0.0
    for gi, (_, arm, colors) in enumerate(GROUPS):
        if gi:
            x += 0.7
        for n, c in zip([n for a, n in COLUMNS if a == arm], colors):
            bars.append((gi, arm, n, c, med[(arm, n)]))
            xs.append(x)
            x += 1.0
    with open(H / "figures" / "overall_median_notice.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "iterations", "median % improvement over start (across 10 problems)"])
        for _, arm, n, _, m in bars:
            wr.writerow([arm, n, round(m, 1)])

    fig, ax = plt.subplots(figsize=(13, 6.0), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    w = 0.62
    top = max(b[4] for b in bars)
    for xi, (_, _, _, c, m) in zip(xs, bars):
        ax.add_patch(plt.Rectangle((xi - w / 2, 0), w, m, linewidth=0, facecolor=c, zorder=2))
        ax.text(xi, m + top * 0.015, "%d%%" % round(m), ha="center", va="bottom", fontsize=14, color=TEXT1)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(0, top * 1.14)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%d iters" % b[2] for b in bars], fontsize=13, color=TEXT1)
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for gi, (label, _, _) in enumerate(GROUPS):
        gx = [xi for xi, b in zip(xs, bars) if b[0] == gi]
        lo, hi = gx[0] - w / 2, gx[-1] + w / 2
        ax.plot([lo, hi], [-0.115, -0.115], color=TEXT2, lw=1, transform=tr, clip_on=False, solid_capstyle="butt")
        for edge in (lo, hi):
            ax.plot([edge, edge], [-0.115, -0.097], color=TEXT2, lw=1, transform=tr, clip_on=False)
        ax.text((lo + hi) / 2, -0.14, label, ha="center", va="top", fontsize=14, color=TEXT1, transform=tr,
                linespacing=1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d%%" % v))
    ax.tick_params(axis="y", labelsize=12, colors=TEXT2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylabel("Median improvement over starting program\n(across 10 problems)", fontsize=12, color=TEXT2)
    fig.tight_layout()
    fig.savefig(H / "figures" / "overall_median_notice.png", facecolor=SURFACE)
    print("wrote", H / "figures" / "overall_median_notice.png")
    for _, arm, n, _, m in bars:
        print("  %-15s @%-3d median %4.0f%%" % (arm, n, m))


if __name__ == "__main__":
    main()
