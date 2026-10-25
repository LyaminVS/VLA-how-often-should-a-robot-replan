"""Explain the measured success/call trade-off without interpolating policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "outputs/capped-report/analysis.json")
    parser.add_argument("--output", type=Path, default=root / "outputs/capped-report/defense-figures")
    args = parser.parse_args()
    raw = args.source.read_bytes()
    data = json.loads(raw)
    assert data["complete"] and data["episodes"] == 270
    rows = {row["method"]: row for row in data["methods"]}
    assert len(rows) == 9 and all(row["episodes"] == 30 for row in rows.values())
    # This is dominance of measured means, not a statistical superiority test.
    frontier = {
        key for key, row in rows.items()
        if not any(
            other["model_calls_mean"] <= row["model_calls_mean"]
            and other["successes"] >= row["successes"]
            and (other["model_calls_mean"] < row["model_calls_mean"]
                 or other["successes"] > row["successes"])
            for other in rows.values()
        )
    }
    assert frontier == {"fixed_50", "dvac_cap_50", "fixed_5", "dvac_cap_5"}
    ink, muted, gray = "#20303B", "#64717C", "#ADB5BC"
    blue, orange, green = "#28628F", "#CB681A", "#24695A"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16, "text.color": ink,
        "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
        "svg.fonttype": "none", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(16, 9), dpi=180)
    fig.text(.055, .94, "Similar model-call cost, very different success", size=29, weight="bold")
    fig.text(.055, .887, "The useful trade-off is higher success with fewer model calls.", size=18, color=muted)
    ax = fig.add_axes([.075, .205, .57, .62])
    ax.set_xscale("log")
    ax.set_xlim(4, 240)
    ax.set_ylim(0, 112)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, 20, 40, 60, 80, 100], [f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    ax.tick_params(length=0, pad=9, labelsize=15)
    ax.grid(color="#E7EBEE", linewidth=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Mean model calls per episode (log scale)", size=17, labelpad=16)
    ax.set_ylabel("Task success rate", size=18, labelpad=10)
    ax.annotate("", xy=(.012, .984), xytext=(.087, .907), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": green, "mutation_scale": 16})
    ax.text(.095, .95, "Better", transform=ax.transAxes, size=17, weight="bold", color=green)
    labels = {
        "fixed_1": (125, 64, "center"),
        "fixed_5": (28, 101, "center"),
        "fixed_10": (17.1, 68, "center"),
        "fixed_25": (13, 32, "left"),
        "fixed_50": (5.25, 19, "left"),
        "dvac_cap_5": (58, 98, "left"),
        "dvac_cap_10": (30, 57, "center"),
        "dvac_cap_25": (15.1, 99, "center"),
        "dvac_cap_50": (7.05, 91, "center"),
    }
    for key, row in rows.items():
        adaptive = key.startswith("dvac")
        color = (orange if adaptive else blue) if key in frontier else gray
        x, y = row["model_calls_mean"], 100 * row["success_mean"]
        ax.scatter(x, y, marker="D" if adaptive else "o", s=160 if key in frontier else 80,
                   color=color, edgecolors="white", linewidths=1.2, zorder=5)
        lx, ly, align = labels[key]
        text = ("C=" if adaptive else "K=") + key.rsplit("_", 1)[1]
        if key == "dvac_cap_50":
            text += "\n25/30"
        elif key == "fixed_25":
            text += "\n11/30"
        ax.annotate(text, xy=(x, y), xytext=(lx, ly), ha=align, va="center", size=15,
                    color=ink if key == "fixed_25" else color,
                    weight="bold" if key == "dvac_cap_50" else "normal",
                    arrowprops={"arrowstyle": "-", "lw": .8, "color": color, "shrinkA": 4, "shrinkB": 8})
    high, low = rows["dvac_cap_50"], rows["fixed_25"]
    comparison_x = (high["model_calls_mean"] * low["model_calls_mean"]) ** .5
    ax.annotate("", xy=(comparison_x, 100 * high["success_mean"] - 3),
                xytext=(comparison_x, 100 * low["success_mean"] + 3),
                arrowprops={"arrowstyle": "-|>", "color": green, "lw": 2.6, "mutation_scale": 16})
    ax.text(8.9, 60, f"+{high['successes'] - low['successes']} successes", rotation=90,
            ha="right", va="center", size=15, color=green, weight="bold")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=blue, label="Fixed K", markersize=8),
        Line2D([], [], marker="D", linestyle="none", color=orange, label="DVAC cap C", markersize=8),
    ], loc="lower right", frameon=False, fontsize=14, borderaxespad=.4)

    # A plain explanation column makes the comparison readable from a distance.
    fig.add_artist(Line2D([.685, .685], [.195, .825], color="#DCE2E6", lw=1.1, transform=fig.transFigure))
    rx = .725
    fig.text(rx, .79, "Almost the same cost", fontsize=20, weight="bold")
    fig.text(rx, .73, "DVAC C=50", fontsize=18, color=orange, weight="bold")
    fig.text(rx, .685, f"25/30 successes · {high['model_calls_mean']:.1f} calls", fontsize=18)
    fig.text(rx, .625, "Fixed K=25", fontsize=18, color=muted, weight="bold")
    fig.text(rx, .58, f"11/30 successes · {low['model_calls_mean']:.1f} calls", fontsize=18)
    fig.text(rx, .465, "Higher observed success", fontsize=20, weight="bold")
    fig.text(rx, .405, "Setting", fontsize=14, color=muted)
    fig.text(.865, .405, "Success", fontsize=14, color=muted, ha="center")
    fig.text(.962, .405, "Calls", fontsize=14, color=muted, ha="right")
    for y, key, label in [( .355, "dvac_cap_50", "C=50"), (.302, "fixed_5", "K=5"), (.249, "dvac_cap_5", "C=5")]:
        row = rows[key]
        color = orange if key.startswith("dvac") else blue
        fig.text(rx, y, label, fontsize=17, color=color, weight="bold")
        fig.text(.865, y, f"{row['successes']}/30", fontsize=17, ha="center")
        fig.text(.962, y, f"{row['model_calls_mean']:.1f}", fontsize=17, ha="right")
    fig.text(.055, .087, "Colored points: best observed trade-offs. For each gray point, a lower-cost option did at least as well.",
             fontsize=14, color=muted)
    fig.text(.055, .05, "30 matched episodes per method. Calls include failures. These are sample results, without uncertainty intervals.",
             fontsize=13, color=muted)
    fig.text(.055, .015, "K is the fixed execution interval. C is the maximum adaptive interval. No single setting wins on both objectives.",
             fontsize=13, color=muted)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in ("png", "svg", "pdf"):
        path = args.output / f"success-vs-calls-explained.{ext}"
        fig.savefig(path, dpi=180)
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    record = {"source_sha256": hashlib.sha256(raw).hexdigest(), "observed_frontier": sorted(frontier),
              "definition": "Nondominated measured sample means; not a test of statistical superiority.",
              "compared_methods": ["dvac_cap_50", "fixed_25"], "extra_successes": high["successes"] - low["successes"],
              "outputs": outputs}
    (args.output / "success-vs-calls-explained.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
