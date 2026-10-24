"""Analyze a complete, independently verified SmolVLA DVAC comparison."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.verify_dvac import load_and_verify, sha256
from vla_replan.storage import atomic_json

TIMINGS = ("inference_s", "selector_s", "simulation_s", "compute_s", "episode_wall_s", "sim_time_s")
LABELS = {"fixed_5": "Fixed K=5", "fixed_10": "Fixed K=10", "fixed_25": "Fixed K=25", "dvac": "DVAC"}
TASKS = {
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": "Soup into basket",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": "Bowl onto plate",
    "open_the_middle_drawer_of_the_cabinet": "Open middle drawer",
}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def wilson(successes, count):
    """Descriptive 95% binomial interval; does not account for paired seed clusters."""
    if type(count) is not int or count <= 0 or not 0 <= successes <= count:
        raise ValueError("Wilson interval needs 0 <= successes <= positive count")
    z = 1.959963984540054
    p, denominator = successes / count, 1 + z * z / count
    middle = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    # The exact boundary is known; avoid a rounding residue at p=0 or p=1.
    lower = 0.0 if successes == 0 else max(0.0, middle - half)
    upper = 1.0 if successes == count else min(1.0, middle + half)
    return lower, upper


def aggregate(records, method, *, task=None):
    group = [record for record in records if record["summary"]["method"] == method
             and (task is None or record["summary"]["task"] == task)]
    if not group:
        raise ValueError("Cannot aggregate an empty method group")
    rows = [record["summary"] for record in group]
    chunks = [chunk for record in group for chunk in record["chunks"]]
    horizons = [chunk["decision"]["chosen_k"] for chunk in chunks]
    nonbootstrap = [chunk for chunk in chunks if not chunk["decision"]["bootstrap"]]
    success = sum(row["success"] for row in rows)
    low, high = wilson(success, len(rows))
    total_steps = sum(row["steps"] for row in rows)
    calls = len(chunks)
    result = {
        "method": method, "task": task or "all", "episodes": len(rows), "successes": success,
        "success_rate": success / len(rows), "descriptive_wilson_low": low, "descriptive_wilson_high": high,
        "physical_settings": len({(row["suite"], row["task"], row["state"]) for row in rows}),
        "total_steps": total_steps, "total_model_calls": calls,
        "mean_model_calls_per_episode": mean(row["model_calls"] for row in rows),
        "calls_per_action_pooled": calls / total_steps,
        "mean_calls_per_action_per_episode": mean(row["model_calls"] / row["steps"] for row in rows),
        "mean_steps_per_episode": mean(row["steps"] for row in rows),
        "chosen_k_mean_per_call": mean(horizons), "chosen_k_median_per_call": statistics.median(horizons),
        "chosen_k_min": min(horizons), "chosen_k_max": max(horizons),
        "chosen_k_mean_across_episode_means": mean(row["mean_chosen_k"] for row in rows),
        "executed_prefix_mean_per_call": total_steps / calls,
        "bootstrap_calls": sum(chunk["decision"]["bootstrap"] for chunk in chunks),
        "nonbootstrap_calls": len(nonbootstrap),
        "nonbootstrap_k1_calls": sum(chunk["decision"]["chosen_k"] == 1 for chunk in nonbootstrap),
        "nonbootstrap_k50_calls": sum(chunk["decision"]["chosen_k"] == 50 for chunk in nonbootstrap),
        "dvac_no_crossing_nonbootstrap_calls": sum(chunk["decision"]["first_crossing"] is None
                                                    for chunk in nonbootstrap) if method == "dvac" else None,
        "mean_eef_path_m": mean(row["eef_path_m"] for row in rows),
        "mean_gripper_close_commands": mean(row["gripper_close_commands"] for row in rows),
    }
    for timing in TIMINGS:
        result["total_" + timing] = math.fsum(row[timing] for row in rows)
        result["mean_" + timing] = mean(row[timing] for row in rows)
    result["inference_s_per_call"] = result["total_inference_s"] / calls
    result["selector_s_per_call"] = result["total_selector_s"] / calls
    result["diagnostics_finalize_s_per_call"] = mean(chunk["diagnostics_finalize_s"] for chunk in chunks)
    result["compute_s_per_action"] = result["total_compute_s"] / total_steps
    return result


def paired_statistics(records, methods):
    by_pair = defaultdict(dict)
    for record in records:
        row = record["summary"]
        if row["method"] in by_pair[row["pair_id"]]:
            raise ValueError("Duplicate method within a pair")
        by_pair[row["pair_id"]][row["method"]] = row
    if any(set(group) != set(methods) for group in by_pair.values()):
        raise ValueError("Paired statistics require every method in every block")
    comparison, outcomes = [], []
    physical = defaultdict(list)
    for method in methods:
        if method == "dvac":
            continue
        wins = losses = both_success = both_failure = 0
        metric_differences = defaultdict(list)
        for pair_id, group in sorted(by_pair.items()):
            adaptive, fixed = group["dvac"], group[method]
            wins += int(adaptive["success"] and not fixed["success"])
            losses += int(fixed["success"] and not adaptive["success"])
            both_success += int(adaptive["success"] and fixed["success"])
            both_failure += int(not adaptive["success"] and not fixed["success"])
            row = {"reference_method": method, "pair_id": pair_id, "suite": fixed["suite"],
                   "task": fixed["task"], "state": fixed["state"], "policy_seed": fixed["policy_seed"],
                   "dvac_success": adaptive["success"], "fixed_success": fixed["success"],
                   "success_difference": int(adaptive["success"]) - int(fixed["success"]),
                   "dvac_calls": adaptive["model_calls"], "fixed_calls": fixed["model_calls"]}
            for metric in ("model_calls", "steps", *TIMINGS):
                difference = adaptive[metric] - fixed[metric]
                row[metric + "_difference"] = difference
                metric_differences[metric].append(difference)
            outcomes.append(row)
            physical[(method, fixed["suite"], fixed["task"], fixed["state"])].append(row)
        comparison.append({
            "reference_method": method, "paired_seed_runs": len(by_pair),
            "dvac_only_success": wins, "fixed_only_success": losses,
            "both_success": both_success, "both_failure": both_failure,
            "success_rate_difference": (wins - losses) / len(by_pair),
            **{metric + "_mean_paired_difference": mean(values) for metric, values in metric_differences.items()},
        })
    state_rows = []
    for (method, suite, task, state), group in sorted(physical.items()):
        state_rows.append({
            "reference_method": method, "suite": suite, "task": task, "state": state,
            "policy_seed_repetitions": len(group), "dvac_success_rate": mean(row["dvac_success"] for row in group),
            "fixed_success_rate": mean(row["fixed_success"] for row in group),
            "success_rate_difference": mean(row["success_difference"] for row in group),
            **{metric + "_mean_paired_difference": mean(row[metric + "_difference"] for row in group)
               for metric in ("model_calls", "steps", *TIMINGS)},
        })
    return comparison, outcomes, state_rows


def plots(output, cfg, records, estimates):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"fixed_5": "#547c9d", "fixed_10": "#3c8f80", "fixed_25": "#b89145", "dvac": "#a55470"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for index, row in enumerate(estimates):
        color = colors.get(row["method"], "#555555")
        axes[0].bar(index, row["success_rate"], color=color, width=.65)
        axes[0].text(index, row["success_rate"] + .025, f"{row['successes']}/{row['episodes']}", ha="center")
        axes[1].scatter(row["mean_model_calls_per_episode"], row["success_rate"], s=110, color=color)
        axes[1].annotate(LABELS.get(row["method"], row["method"]),
                         (row["mean_model_calls_per_episode"], row["success_rate"]), xytext=(5, 8),
                         textcoords="offset points", fontsize=9)
    axes[0].set(xticks=range(len(estimates)), xticklabels=[LABELS.get(row["method"], row["method"]) for row in estimates],
                ylabel="Completion fraction", ylim=(0, 1.15), title="Matched completion outcomes")
    axes[1].set(xlabel="Mean model calls per episode", ylabel="Completion fraction", ylim=(-.05, 1.15),
                title="Observed completion/computation trade-off")
    axes[1].margins(x=.3)
    fig.suptitle(f"SmolVLA: {len(records)} episodes, six task/state settings, two seeds each")
    fig.savefig(output / "completion_and_calls.png", dpi=180)
    plt.close(fig)

    adaptive = [record for record in records if record["summary"]["method"] == "dvac"]
    values = [chunk["decision"]["chosen_k"] for record in adaptive for chunk in record["chunks"]
              if not chunk["decision"]["bootstrap"]]
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    histogram = Counter(values)
    ax.bar(range(1, cfg.base.chunk_size + 1), [histogram[k] for k in range(1, cfg.base.chunk_size + 1)], color=colors["dvac"])
    ax.set(xlabel="Chosen actions before replanning (K)", ylabel="Model calls", title="DVAC interval choices after initialization",
           xlim=(0, cfg.base.chunk_size + 1))
    ax.text(.02, .95, f"{len(values)} decisions; {len(adaptive)} first-call K=1 initializations excluded",
            transform=ax.transAxes, va="top")
    fig.savefig(output / "dvac_interval_distribution.png", dpi=180)
    plt.close(fig)

    selected = [record for record in adaptive if record["summary"]["state"] == cfg.base.initial_states[0]
                and record["summary"]["policy_seed"] == cfg.base.policy_seeds[0]]
    selected.sort(key=lambda record: [task.name for task in cfg.base.tasks].index(record["summary"]["task"]))
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3.2 * len(selected)), squeeze=False, layout="constrained")
    for row_index, record in enumerate(selected):
        summary, chunks = record["summary"], record["chunks"]
        steps = [chunk["step"] for chunk in chunks] + [summary["steps"]]
        ks = [chunk["decision"]["chosen_k"] for chunk in chunks]
        axes[row_index, 0].step(steps, ks + [ks[-1]], where="post", color=colors["dvac"])
        axes[row_index, 0].set(xlabel="Executed robot action index", ylabel="Chosen K", ylim=(0, 52),
                               title=f"{TASKS.get(summary['task'], summary['task'])}: {'success' if summary['success'] else 'failure'}")
        if len(chunks) > 1:
            example = chunks[1]
            axes[row_index, 1].plot(range(cfg.base.chunk_size), example["variance"], color=colors["dvac"], label="V per future action")
            axes[row_index, 1].axhline(example["decision"]["threshold"], color="#333333", linestyle="--", label="Past-only threshold")
            crossing = example["decision"]["first_crossing"]
            if crossing is not None:
                axes[row_index, 1].axvline(crossing, color="#c57730", linestyle=":", label="First strict crossing")
            axes[row_index, 1].set(title=f"Second call at step {example['step']}: chosen K={example['decision']['chosen_k']}",
                                   xlabel="Future action index j (zero based)", ylabel="Denoising variance")
            axes[row_index, 1].legend(fontsize=8)
    fig.suptitle(f"Predetermined examples: state {cfg.base.initial_states[0]}, policy seed {cfg.base.policy_seeds[0]}")
    fig.savefig(output / "dvac_decision_examples.png", dpi=180)
    plt.close(fig)


def report_text(cfg, results):
    estimates, comparisons = results["methods"], results["paired_comparisons"]
    lines = ["# SmolVLA: adaptive replanning with DVAC", "",
             f"This comparison contains **{results['episodes']} verified episodes**, with {estimates[0]['episodes']} per method. "
             "Each method uses the same frozen checkpoint, three tasks, two initial states per task, and two policy seeds. "
             "Every fixed-interval control uses the same flow-matching diagnostic instrumentation as DVAC.", "",
             "## Completion and computation", "",
             "| Method | Completed | Completion | Descriptive Wilson 95% interval | Calls/episode | Calls/action | Inference s/episode | Selector ms/episode | Wall s/episode |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in estimates:
        lines.append(f"| {LABELS.get(row['method'], row['method'])} | {row['successes']}/{row['episodes']} | "
                     f"{100*row['success_rate']:.1f}% | {100*row['descriptive_wilson_low']:.1f}–{100*row['descriptive_wilson_high']:.1f}% | "
                     f"{row['mean_model_calls_per_episode']:.1f} | {row['calls_per_action_pooled']:.3f} | "
                     f"{row['mean_inference_s']:.2f} | {1000*row['mean_selector_s']:.3f} | {row['mean_episode_wall_s']:.2f} |")
    lines += ["", "Calls/action is the pooled number of calls divided by executed actions. Episode lengths can differ because "
              "successful episodes stop early; both per-episode and per-action measures are therefore reported. "
              "The Wilson intervals are descriptive binomial intervals. They do not account for dependence between seeds "
              "sharing an initial state and must not be interpreted as a significance test.", "",
              "![Completion and calls](completion_and_calls.png)", "", "## Results by task", "",
              "| Task | Method | Completed | Calls/episode | Compute s/episode |", "|---|---|---:|---:|---:|"]
    for row in results["tasks"]:
        lines.append(f"| {TASKS.get(row['task'], row['task'])} | {LABELS.get(row['method'], row['method'])} | "
                     f"{row['successes']}/{row['episodes']} | {row['mean_model_calls_per_episode']:.1f} | {row['mean_compute_s']:.2f} |")
    lines += ["", "## Matched outcome differences", "",
              "A pair has the same task, initial state, and policy seed. Differences below are DVAC minus the fixed method.", "",
              "| Reference | DVAC succeeds, fixed fails | Fixed succeeds, DVAC fails | Both succeed | Both fail | Completion difference | Mean call difference |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {LABELS.get(row['reference_method'], row['reference_method'])} | {row['dvac_only_success']} | "
                     f"{row['fixed_only_success']} | {row['both_success']} | {row['both_failure']} | "
                     f"{100*row['success_rate_difference']:+.1f} percentage points | {row['model_calls_mean_paired_difference']:+.1f} |")
    lines += ["", "There are six physical task/state settings, each repeated with two sampling seeds. "
              "The complete paired outcomes and the means over seeds within each physical setting are retained in "
              "`paired_outcomes.csv` and `physical_setting_differences.csv`. No p-value is reported.", "",
              "## Every matched run", "",
              "Each method cell shows outcome and model calls. These are all planned task/state/seed pairs.", "",
              "| Task | State | Seed | " + " | ".join(LABELS.get(method, method) for method in cfg.methods) + " |",
              "|---|---:|---:|" + "---:|" * len(cfg.methods)]
    for row in results["outcome_matrix"]:
        cells = [f"{'Success' if row[method + '_success'] else 'Failure'} / {row[method + '_calls']} calls"
                 for method in cfg.methods]
        lines.append(f"| {TASKS.get(row['task'], row['task'])} | {row['state']} | {row['policy_seed']} | " + " | ".join(cells) + " |")
    lines += ["", "## Adaptive interval behavior", ""]
    adaptive = next(row for row in estimates if row["method"] == "dvac")
    lines.append(f"DVAC chose a mean K of **{adaptive['chosen_k_mean_per_call']:.2f}** across all calls "
                 f"(median {adaptive['chosen_k_median_per_call']:g}, range {adaptive['chosen_k_min']}–{adaptive['chosen_k_max']}). "
                 f"The actual mean executed prefix was {adaptive['executed_prefix_mean_per_call']:.2f} actions, "
                 "which can be shorter because an episode ends inside a selected prefix. "
                 f"The {adaptive['bootstrap_calls']} mandatory first-call K=1 decisions are distinguished from later choices.")
    lines += ["", f"Among {adaptive['nonbootstrap_calls']} later calls, {adaptive['nonbootstrap_k1_calls']} chose K=1 and "
              f"{adaptive['nonbootstrap_k50_calls']} chose K=50; {adaptive['dvac_no_crossing_nonbootstrap_calls']} had no variance "
              "crossing. These counts describe how the rule behaved and are not calibrated failure probabilities.", "",
              "![Adaptive interval distribution](dvac_interval_distribution.png)", "",
              "The examples below use the predetermined first initial state and first policy seed for all three tasks. "
              "The variance panel always shows the second policy call, so its selection is independent of the outcome.", "",
              "![Adaptive decision examples](dvac_decision_examples.png)", "",
              "## Timing and evidence", "",
              "Inference time includes the instrumented policy prediction and variance calculation. Selector time "
              "is measured separately. Compute time is inference plus selection plus simulator stepping. Episode wall "
              "time includes reset, rollout, and video writer closure, but excludes JSON serialization, hashing, and "
              "publication of the evidence directory. The simulator pauses during inference. "
              "The plain-versus-instrumented equality check is an integrity check; its ordered first calls are not "
              "a valid estimate of instrumentation overhead.", "",
              f"The independent verifier checked all {results['verification']['totals']['model_calls']} generated chunks "
              f"and {results['verification']['totals']['steps']} executed actions, including variance replay, "
              "past-only rolling thresholds, selected prefixes, seed mapping, initial-state matching, final action "
              "execution, and evidence checksums. The first complete predictions match exactly across all methods "
              "within each state/seed pair. `verification.json` records the checks and evidence hashes.", "",
              "## Interpretation and limits", "",
              "These observations test transfer of the DVAC decision rule to SmolVLA. They do not reproduce the "
              "paper's complete setup or establish an official LIBERO score. The first-call K=1 initialization and "
              "use of seven valid action channels are explicit implementation choices. The model, normalization, "
              "ten-step solver, five-estimate tail, five-call history, and alpha=2 are fixed before comparison.", "",
              "Success differences must be considered together with model calls and runtime. More frequent updates "
              "may help some trajectories and cannot guarantee correction of an inaccurate grasp or an incorrect "
              "plan. Internal denoising stability is not action correctness. The comparison combines the effects of "
              "fresh observations and new stochastic generation; it does not isolate those mechanisms. "
              "Only six task/state settings are represented, so broad generalization is unsupported. "
              "Gripper close-command counts are control diagnostics, not verified grasp or failure annotations.", "",
              "See `docs/dvac-protocol.md` for the mathematical rule, complete design, and predetermined demo selection.", ""]
    return "\n".join(lines)


def analyze(root, output, *, make_plots=True, decode_videos=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Analysis output and immutable raw evidence must use separate directories")
    cfg, records, verification = load_and_verify(root, decode_videos=decode_videos)
    estimates = [aggregate(records, method) for method in cfg.methods]
    tasks = [aggregate(records, method, task=task.name) for task in cfg.base.tasks for method in cfg.methods]
    paired, outcomes, states = paired_statistics(records, cfg.methods)
    by_pair = defaultdict(dict)
    for record in records:
        summary = record["summary"]
        by_pair[(summary["suite"], summary["task"], summary["state"], summary["policy_seed"])][summary["method"]] = summary
    outcome_matrix = []
    for (suite, task, state, seed), group in sorted(by_pair.items()):
        row = {"suite": suite, "task": task, "state": state, "policy_seed": seed}
        for method in cfg.methods:
            row[method + "_success"] = group[method]["success"]
            row[method + "_calls"] = group[method]["model_calls"]
        outcome_matrix.append(row)
    decisions, episode_rows = [], []
    histogram = Counter()
    for record in records:
        summary = record["summary"]
        episode_rows.append({key: summary[key] for key in (
            "episode_id", "pair_id", "suite", "task", "state", "policy_seed", "method", "k", "success",
            "steps", "model_calls", *TIMINGS, "mean_chosen_k", "mean_executed_prefix", "eef_path_m", "gripper_close_commands",
        )})
        for chunk in record["chunks"]:
            decision = chunk["decision"]
            decisions.append({"episode_id": summary["episode_id"], "method": summary["method"], "step": chunk["step"],
                              **decision, "executed_steps": chunk["executed_steps"],
                              "current_variance_mean": mean(chunk["variance"]), "current_variance_max": max(chunk["variance"]),
                              "inference_s": chunk["inference_s"], "selector_s": chunk["selector_s"]})
            histogram[(summary["method"], decision["bootstrap"], decision["chosen_k"])] += 1
    result = {"complete": True, "episodes": len(records), "methods": estimates, "tasks": tasks,
              "paired_comparisons": paired, "physical_setting_differences": states,
              "outcome_matrix": outcome_matrix,
              "config_fingerprint": cfg.fingerprint, "analysis_source_sha256": sha256(__file__),
              "verification": {key: value for key, value in verification.items() if key != "evidence"},
              "uncertainty_note": "Wilson intervals are descriptive and do not adjust for seeds sharing physical starts; no significance test."}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "verification.json", verification)
    atomic_json(output / "analysis.json", result)
    write_csv(output / "methods.csv", estimates)
    write_csv(output / "tasks.csv", tasks)
    write_csv(output / "episodes.csv", episode_rows)
    write_csv(output / "paired_comparisons.csv", paired)
    write_csv(output / "paired_outcomes.csv", outcomes)
    write_csv(output / "outcome_matrix.csv", outcome_matrix)
    write_csv(output / "physical_setting_differences.csv", states)
    write_csv(output / "decisions.csv", decisions)
    write_csv(output / "interval_histogram.csv", [
        {"method": method, "bootstrap": bootstrap, "chosen_k": k, "calls": count}
        for (method, bootstrap, k), count in sorted(histogram.items())])
    if make_plots:
        plots(output, cfg, records, estimates)
    text = report_text(cfg, result)
    if not make_plots:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("![")) + "\n"
    (output / "report.md").write_text(text)
    artifact_names = ["analysis.json", "verification.json", "methods.csv", "tasks.csv", "episodes.csv",
                      "paired_comparisons.csv", "paired_outcomes.csv", "physical_setting_differences.csv",
                      "decisions.csv", "interval_histogram.csv", "outcome_matrix.csv", "report.md"]
    if make_plots:
        artifact_names += ["completion_and_calls.png", "dvac_interval_distribution.png", "dvac_decision_examples.png"]
    atomic_json(output / "artifacts.json", {
        "statistics_filename": "analysis.json", "statistics_sha256": sha256(output / "analysis.json"),
        "analysis_source_sha256": sha256(__file__), "config_fingerprint": cfg.fingerprint,
        "files": {name: sha256(output / name) for name in artifact_names},
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--decode-videos", action="store_true")
    args = parser.parse_args()
    result = analyze(args.input, args.output, make_plots=not args.no_plots, decode_videos=args.decode_videos)
    print(json.dumps({"episodes": result["episodes"], "complete": result["complete"],
                      "methods": [{key: row[key] for key in ("method", "successes", "episodes", "total_model_calls", "chosen_k_mean_per_call")}
                                  for row in result["methods"]]}, indent=2))


if __name__ == "__main__":
    main()
