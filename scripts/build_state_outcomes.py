"""Plot all 150 episode outcomes in the unified study.

Each cell is one native LIBERO success result for a matched task, initial state,
and K. All ten initial states contribute to the main descriptive analysis.
Requires the complete, verified analysis; never runs the simulator.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.analyze_extension import sha256  # noqa: E402
from scripts.build_expanded_figures import KS, TASKS, load_bundle, publish  # noqa: E402


def outcome_grids(rows):
    """Keep states and K values explicit; refuse missing, duplicate or nonbinary cells."""
    indexed = {}
    for row in rows:
        key = (row["suite"], row["task"], row["state"], row["k"])
        if key in indexed:
            raise ValueError(f"Duplicate outcome cell: {key}")
        if type(row["success"]) not in (bool, int) or row["success"] not in (0, 1):
            raise ValueError(f"Expected binary native success: {key}")
        indexed[key] = int(row["success"])
    expected = {(suite, task, state, k) for suite, task in TASKS for state in range(10) for k in KS}
    if set(indexed) != expected:
        raise ValueError("State outcomes require exactly 3 tasks × 10 states × 5 K values")
    return {
        (suite, task): [[indexed[suite, task, state, k] for k in KS] for state in range(10)]
        for suite, task in TASKS
    }


def build_state_outcomes(args):
    bundle = load_bundle(args.run, args.pilot, args.validation)
    output = bundle["run"] / "report"
    if any(output.is_relative_to(source["root"]) for source in bundle["sources"].values()):
        raise ValueError("Derived figures must remain outside the original source runs")
    targets = [output / f"state-outcomes.{extension}" for extension in ("png", "pdf", "json")]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("State-outcome artifacts already exist; pass --overwrite to replace them")
    grids = outcome_grids(bundle["rows"]["combined"])
    statistics_digest = sha256(bundle["statistics_path"])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    failure, success = "#AD4545", "#247B62"
    cmap = ListedColormap([failure, success])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "savefig.facecolor": "white", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.8))
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.215, top=0.785, wspace=0.28)
        try:
            for ax, ((suite, task), title), panel in zip(axes, TASKS.items(), "abc", strict=True):
                grid = grids[suite, task]
                ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
                ax.set_xticks(range(len(KS)), KS)
                ax.set_yticks(range(10), range(10))
                ax.set_xticks(np.arange(-0.5, len(KS), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=0.7)
                ax.tick_params(which="both", length=0)
                ax.tick_params(which="minor", bottom=False, left=False)
                ax.set_xlabel("K: actions before replanning", labelpad=9)
                ax.set_ylabel("Initial state", labelpad=8)
                ax.set_title(f"{panel}  {title}", loc="left", pad=12)
                for state, values in enumerate(grid):
                    for column, value in enumerate(values):
                        ax.text(column, state, str(value), ha="center", va="center",
                                color="white", fontsize=12, fontweight="bold")
                for spine in ax.spines.values():
                    spine.set_visible(False)

            fig.suptitle("Native LIBERO outcomes for every matched initial state", fontsize=17, y=0.975)
            fig.text(0.5, 0.918, "150 unique episodes | one model | policy seed 11 | the same state across K",
                     ha="center", fontsize=11.5)
            fig.legend(handles=[Patch(facecolor=failure, label="0: success criterion not reached"),
                                Patch(facecolor=success, label="1: success criterion reached")],
                       loc="upper center", bbox_to_anchor=(0.55, 0.885), ncol=2, frameon=False)
            fig.text(0.12, 0.115, "Each cell is one episode, not a success probability. All 150 episodes contribute to the main analysis.",
                     fontsize=10)
            fig.text(0.12, 0.075, "Each task and K uses 10 matched initial states; each K has 30 episodes across the three tasks.",
                     fontsize=10)
            fig.text(0.12, 0.035, "LIBERO success may precede object release. This figure shows outcomes, not manually inferred grasp errors.",
                     fontsize=10, color="#435665")

            output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".state-outcomes-", dir=output) as temporary:
                staging = Path(temporary)
                entries = {}
                for extension in ("png", "pdf"):
                    path = staging / f"state-outcomes.{extension}"
                    fig.savefig(path, dpi=300)
                    entries[path.name] = sha256(path)
                if sha256(bundle["statistics_path"]) != statistics_digest:
                    raise RuntimeError("Statistics changed during figure generation")
                record = {
                    "schema_version": 1, "artifact_kind": "state_outcomes", "primary_phase": "combined",
                    "combined_analysis_role": "main_descriptive",
                    "statistics_sha256": statistics_digest, "generator_sha256": sha256(__file__),
                    "episodes": {"pilot": 30, "validation": 120, "combined": 150},
                    "states": list(range(10)), "intervals": list(KS), "metric": "native_success",
                    "files": entries,
                }
                (staging / "state-outcomes.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                for target in targets:
                    publish(staging / target.name, target, args.overwrite)
        finally:
            plt.close(fig)
    print(json.dumps({"output": str(output), "files": [path.name for path in targets],
                      "visual_review_required": True}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--pilot", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--overwrite", action="store_true")
    build_state_outcomes(parser.parse_args(argv))


if __name__ == "__main__":
    main()
