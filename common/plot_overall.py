"""Overall chart: median across the 10 problems of each arm's per-problem mean
% improvement over the starting program. Same numbers as the legend of
per_problem_five_arms.png. Writes figures/overall_median.png and .csv."""
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
from plot_bars import SIX_ARMS as ARMS, compute
from plot_improvement import SURFACE, TEXT1, TEXT2, GRID


def main():
    names, means, _ = compute(ARMS)
    med = np.median(means, axis=0)
    with open(H / "figures" / "overall_median.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["arm", "median % improvement over start (across 10 problems)"])
        for s, m in zip(ARMS, med):
            wr.writerow([s[0], round(m, 1)])

    groups = [("Code without hints (comments, variable names)", "baseline"), ("Code with hints", "hinted")]
    xs, x = [], 0.0
    for i, a in enumerate(ARMS):
        if i and a[1] != ARMS[i - 1][1]:
            x += 0.6  # visual gap between the two groups
        xs.append(x)
        x += 1.0

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    w = 0.62
    for xi, s_, m in zip(xs, ARMS, med):
        ax.add_patch(plt.Rectangle((xi - w / 2, 0), w, m, linewidth=0, facecolor=s_[3], zorder=2))
        ax.text(xi, m + med.max() * 0.015, "%d%%" % round(m), ha="center", va="bottom", fontsize=14, color=TEXT1)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(0, med.max() * 1.14)
    ax.set_xticks(xs)
    ax.set_xticklabels([s_[0].split("(")[1].rstrip(")") for s_ in ARMS], fontsize=13, color=TEXT1)
    # shared group label under each set of bars, with a hairline bracket
    tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for label, arm in groups:
        gx = [xi for xi, s_ in zip(xs, ARMS) if s_[1] == arm]
        lo, hi = gx[0] - w / 2, gx[-1] + w / 2
        ax.plot([lo, hi], [-0.125, -0.125], color=TEXT2, lw=1, transform=tr, clip_on=False, solid_capstyle="butt")
        for edge in (lo, hi):
            ax.plot([edge, edge], [-0.125, -0.105], color=TEXT2, lw=1, transform=tr, clip_on=False)
        ax.text((lo + hi) / 2, -0.15, label, ha="center", va="top", fontsize=14, color=TEXT1, transform=tr)
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
    fig.savefig(H / "figures" / "overall_median.png", facecolor=SURFACE)
    print("wrote", H / "figures" / "overall_median.png")
    for s, m in zip(ARMS, med):
        print("  %-20s median %4.0f%%" % (s[0], m))


if __name__ == "__main__":
    main()
