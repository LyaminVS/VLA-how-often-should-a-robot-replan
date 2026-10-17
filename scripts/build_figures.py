"""Draw the final compact-study figure from existing statistics, without recalculating them.

Usage: .venv/bin/python scripts/build_figures.py --run outputs/compact
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

KS = (1, 5, 10, 25, 50)
TASKS = {
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"): "Open the middle drawer",
    ("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"): "Soup can into basket",
    (
        "libero_spatial",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    ): "Black bowl onto plate",
}


def values(rows, metric):
    selected = [row for row in rows if row["metric"] == metric]
    indexed = {row["k"]: row for row in selected}
    if len(selected) != len(KS) or set(indexed) != set(KS):
        raise ValueError(f"Expected exactly one {metric} estimate for each K={KS}")
    means, lows, highs = [], [], []
    for k in KS:
        mean, low, high = (indexed[k][field] for field in ("mean", "ci_low", "ci_high"))
        if any(value is None or not math.isfinite(value) for value in (mean, low, high)):
            raise ValueError(f"Missing or invalid existing interval: {metric}, K={k}")
        if not low <= mean <= high:
            raise ValueError(f"Invalid existing interval: {metric}, K={k}")
        means.append(mean)
        lows.append(low)
        highs.append(high)
    return means, lows, highs


def plot_interval(ax, rows, metric, *, color, offset=0, label=None):
    means, lows, highs = values(rows, metric)
    x = [index + offset for index in range(len(KS))]
    errors = [[mean - low for mean, low in zip(means, lows, strict=True)],
              [high - mean for mean, high in zip(means, highs, strict=True)]]
    ax.errorbar(x, means, yerr=errors, fmt="o-", color=color, linewidth=1.8,
                markersize=5, capsize=3, elinewidth=1, label=label)


def build_figure(root):
    root = Path(root).resolve()
    statistics = json.loads((root / "report" / "statistics.json").read_text())
    coverage = statistics["coverage"]
    if coverage["provisional"] or any(coverage[key] != 30 for key in
                                      ("completed_episodes", "planned_episodes", "analyzed_episodes")):
        raise ValueError("This figure requires the complete 30-episode compact study")
    task_rows = statistics["per_task"]
    if {(row["suite"], row["task"]) for row in task_rows} != set(TASKS):
        raise ValueError("Unexpected task identities in statistics.json")
    if any(row["initial_states"] != 2 for row in task_rows):
        raise ValueError("Expected two initial states per task")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.21, top=0.84,
                        hspace=0.60, wspace=0.29)
    fig.suptitle("Replanning frequency: performance and cost", fontsize=16, y=0.98)
    fig.text(0.5, 0.92, "30 episodes · 6 situations (3 tasks × 2 initial states) · 6 episodes per K",
             ha="center", fontsize=11.5)

    panels = [
        (axes[0, 0], "success", "a  Task completion", "Success rate"),
        (axes[0, 1], "compute_s", "b  Computation time", "Mean per episode, s"),
        (axes[1, 0], "model_calls", "c  Model calls", "Mean per episode"),
    ]
    for ax, metric, title, ylabel in panels:
        plot_interval(ax, statistics["estimates"], metric, color="#27658b")
        ax.set_title(title, loc="left", pad=10, fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

    tasks_ax = axes[1, 1]
    tasks_ax.set_title("d  Success by task", loc="left", pad=10, fontweight="semibold")
    for (identity, label), color, offset in zip(TASKS.items(),
                                               ("#256d9b", "#d77525", "#418657"),
                                               (-0.10, 0, 0.10), strict=True):
        rows = [row for row in task_rows if (row["suite"], row["task"]) == identity]
        plot_interval(tasks_ax, rows, "success", color=color, offset=offset, label=label)
    tasks_ax.set_ylabel("Success rate")
    handles, labels = tasks_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.53, 0.113), ncol=3,
               frameon=False, fontsize=10, handlelength=1.5, columnspacing=1.5)

    for ax in axes.flat:
        ax.set_xticks(range(len(KS)), KS)
        ax.set_xlim(-0.3, 4.3)
        ax.set_xlabel("K - actions before the next observation", labelpad=5)
        ax.grid(axis="y", color="#e0e4e8", linewidth=0.7)
        ax.set_axisbelow(True)
    for ax in (axes[0, 0], tasks_ax):
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.text(0.085, 0.060,
             "Points: means. Error bars: 95% intervals across initial states.", fontsize=10.5)
    fig.text(0.085, 0.030,
             "Descriptive intervals: only 2 states per task. A zero-width interval does not establish robustness.",
             fontsize=10.5)
    output = root / "report" / "final-comparison.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Completed compact run directory")
    args = parser.parse_args()
    build_figure(args.run)


if __name__ == "__main__":
    main()
