"""Audit, summarize and render the adaptive/causal extension. No model execution."""

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_replan.analysis import cluster_estimates  # noqa: E402
from vla_replan.core import sampling_seed  # noqa: E402
from vla_replan.research import choose_interval, load_trial  # noqa: E402
from vla_replan.storage import atomic_json  # noqa: E402

LABELS = {"fixed1": "K=1", "fixed5": "K=5", "fixed10": "K=10", "fixed25": "K=25", "fixed50": "K=50", "adaptive_v1": "Adaptive v1"}
ORDER = list(LABELS)
METRICS = ["success", "model_calls", "inference_s", "compute_s", "steps", "gripper_sign_changes", "mean_action_jump"]


def deferred_transitions(steps, chunks):
    """Descriptive command-level deferral; never a contact/grasp-failure label."""
    by_step = {r["step"]: r for r in steps}
    count = eligible = deferred = 0
    for c in chunks:
        sign = c["actions"][0][6] > 0
        transition = next((i for i, a in enumerate(c["actions"][1:10], 1) if (a[6] > 0) != sign), None)
        if transition is None:
            continue
        count += 1
        s = c["step"]
        if any(s+i not in by_step for i in range(transition+1)):
            continue
        eligible += 1
        deferred += not any((by_step[s+i]["action"][6] > 0) != sign for i in range(1, transition+1))
    return {"planned_transitions": count, "evaluable_transitions": eligible, "deferred_transitions": deferred}


def command_diagnostics(steps, chunks):
    values, grip_changes = [], 0
    for old, new in zip(chunks, chunks[1:], strict=False):
        index = new["step"]-old["step"]
        if index >= len(old["actions"]):
            continue
        previous = old["actions"][index]
        current = new["actions"][0]
        values.append(math.sqrt(sum((a-b)**2 for a,b in zip(previous[:3], current[:3], strict=True))))
        grip_changes += (previous[6]>0) != (current[6]>0)
    return deferred_transitions(steps, chunks) | {
        "comparable_replans": len(values), "gripper_plan_disagreements": grip_changes,
        "mean_plan_revision_translation": sum(values)/len(values) if values else None}


def audit_trial(path, manifest, video=True):
    import imageio.v2 as imageio
    import numpy as np

    trial = load_trial(path)
    s, rows, chunks = trial["summary"], trial["steps"], trial["chunks"]
    assert s["fingerprint"] == manifest["fingerprint"] and s["data_kind"] == "real_libero"
    assert len(rows) == s["executed_steps"] == s["steps"]-s["fork_step"]
    assert len(chunks) == s["model_calls"]
    assert [r["step"] for r in rows] == list(range(s["fork_step"], s["steps"]))
    assert rows[-1]["success_after_action"] == s["success"]
    assert all(r["physics_state_sha256"] for r in rows)
    assert math.isclose(sum(c["inference_s"] for c in chunks), s["inference_s"], abs_tol=1e-8)
    by_step = {c["step"]: c for c in chunks}
    next_step = s["fork_step"]
    for c in chunks:
        assert c["step"] == next_step
        assert np.asarray(c["actions"]).shape == (50, 7)
        assert np.isfinite(c["actions"]).all()
        method = "fixed1" if s["method"] == "constant_noise" else "fixed10" if s["method"] == "commit10" else s["method"]
        assert choose_interval(c["actions"], method) == (c["selected_k"], c["reason"])
        seed_step = s["fork_step"] if s["method"] == "constant_noise" else c["step"]
        assert c["sampling_seed"] == sampling_seed(s["policy_seed"], seed_step)
        next_step += c["selected_k"]
    for r in rows:
        c = by_step[r["plan_step"]]
        assert 0 <= r["chunk_index"] == r["step"]-c["step"] < c["selected_k"]
        assert r["replanned"] == (r["step"] == c["step"])
        np.testing.assert_array_equal(r["action"], np.clip(c["actions"][r["chunk_index"]], -1, 1))
    if video:
        with imageio.get_reader(str(path / "video.mp4")) as reader:
            assert reader.get_meta_data()["fps"] == 20
            assert reader.count_frames() == len(rows)+20
    trial["diagnostics"] = command_diagnostics(rows, chunks)
    trial["directory"] = str(path)
    trial["artifact_sha256"] = {str((path / name).relative_to(ROOT) if (path / name).is_relative_to(ROOT) else path / name):
                               hashlib.sha256((path / name).read_bytes()).hexdigest()
                               for name in ("summary.json", "steps.jsonl", "chunks.jsonl", "video.mp4")}
    return trial


def load_audited(root, video=True, allow_incomplete=False):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    status = json.loads((root / "status.json").read_text())
    if status["phase"] != "completed" and not allow_incomplete:
        raise ValueError(f"Study is incomplete: {status}")
    trials = [audit_trial(p.parent, manifest, video) for p in sorted(root.glob("episodes/*/summary.json"))]
    skipped = [json.loads(p.read_text()) | {"trial": p.parent.name} for p in root.glob("episodes/*/skipped.json")]
    if status["phase"] == "completed":
        assert len(trials)+len(skipped) == manifest["scheduled_jobs"]
    return manifest, trials, skipped


def aggregate(rows, methods):
    mapped = [{**r, "k": ORDER.index(r["method"])} for r in rows]
    intervals = [ORDER.index(m) for m in methods]
    stats = []
    for metric in METRICS:
        estimates, _ = cluster_estimates(mapped, intervals, metric, seed=20261024, reference=ORDER.index("fixed10"))
        for e in estimates:
            e["method"] = ORDER[e.pop("k")]
            e["reference_method"] = ORDER[e.pop("reference_k")]
            stats.append(e)
    result = []
    for m in methods:
        subset = [r for r in rows if r["method"] == m]
        result.append({"method": m, "episodes": len(subset), "successes": sum(r["success"] for r in subset),
                       **{metric: next(s["mean"] for s in stats if s["method"] == m and s["metric"] == metric) for metric in METRICS}})
    return result, stats


def build_results(root, extra=None, video=True, allow_incomplete=False):
    import numpy as np

    manifest, trials, skipped = load_audited(root, video, allow_incomplete)
    provisional = json.loads((Path(root) / "status.json").read_text())["phase"] != "completed"
    manifests = [manifest]
    if extra is not None:
        other, more, other_skipped = load_audited(extra, video, allow_incomplete)
        provisional |= json.loads((Path(extra) / "status.json").read_text())["phase"] != "completed"
        for field in ("runtime", "hardware", "base_config"):
            assert other[field] == manifest[field], f"Supplement differs in {field}"
        trials.extend(more)
        skipped.extend(other_skipped)
        manifests.append(other)
    by_id = {t["summary"]["trial_id"]: t for t in trials}
    assert len(by_id) == len(trials)
    heldout = [t for t in trials if t["summary"]["phase"] == "heldout"]
    declared = set().union(*(set(m["plan"]["methods"]) for m in manifests))
    methods = [m for m in ORDER if m in declared]
    pairs = defaultdict(dict)
    for t in heldout:
        pairs[t["summary"]["pair_id"]][t["summary"]["method"]] = t
    pairs = {key: pair for key, pair in pairs.items() if set(pair) == set(methods)}
    omitted = sum(t["summary"]["pair_id"] not in pairs for t in heldout)
    heldout = [t for t in heldout if t["summary"]["pair_id"] in pairs]
    if not heldout:
        raise ValueError("No fully paired evaluation blocks available")
    initial_chunk_error = 0
    for pair in pairs.values():
        assert set(pair) == set(methods)
        for field in ("initial_state_sha256", "initial_observation_sha256"):
            assert len({t["summary"][field] for t in pair.values()}) == 1
        assert len({t["steps"][0]["physics_state_sha256"] for t in pair.values()}) == 1
        first = np.asarray(pair.get("fixed1", pair["fixed10"])["chunks"][0]["actions"])
        initial_chunk_error = max(initial_chunk_error, *(float(np.abs(np.asarray(t["chunks"][0]["actions"])-first).max()) for t in pair.values()))
    assert initial_chunk_error <= 2e-6
    main_rows = [t["summary"] | t["diagnostics"] for t in heldout]
    adaptive_chunks = [c for t in heldout if t["summary"]["method"] == "adaptive_v1" for c in t["chunks"]]
    summary, statistics = aggregate(main_rows, methods)
    continuation = []
    causal_pairs = defaultdict(set)
    for t in trials:
        if t["summary"]["phase"] == "causal":
            causal_pairs[t["summary"]["pair_id"]].add(t["summary"]["method"])
    for t in trials:
        s = t["summary"]
        if s["phase"] != "causal":
            continue
        if causal_pairs[s["pair_id"]] != {"constant_noise", "commit10"}:
            continue
        baseline = by_id[s["reference_trial"]]
        p = Path(baseline["directory"]) / "steps.jsonl"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == s["reference_steps_sha256"]
        fork = s["fork_step"]
        source_row = baseline["steps"][fork]
        assert source_row["observation_sha256"] == t["steps"][0]["observation_sha256"]
        assert source_row["physics_state_sha256"] == t["steps"][0]["physics_state_sha256"]
        first_error = float(np.abs(np.asarray(baseline["chunks"][fork]["actions"])-t["chunks"][0]["actions"]).max())
        assert first_error == s["first_chunk_max_abs_error"] <= 2e-6
        ref_chunks = [c for c in baseline["chunks"] if c["step"] >= fork]
        ref_rows = baseline["steps"][fork:]
        continuation.append({"trial_id": s["trial_id"], "pair_id": s["pair_id"], "method": s["method"],
                             "suite": s["suite"], "task": s["task"], "state": s["state"], "fork_step": fork,
                             "baseline_success": baseline["summary"]["success"], "success": s["success"],
                             "rescued": not baseline["summary"]["success"] and s["success"],
                             "regressed": baseline["summary"]["success"] and not s["success"],
                             "baseline_suffix_calls": len(ref_chunks), "calls": s["model_calls"],
                             "baseline_suffix_inference_s": sum(c["inference_s"] for c in ref_chunks),
                             "inference_s": s["inference_s"], "steps_after_fork": s["executed_steps"],
                             "baseline_steps_after_fork": len(ref_rows), "first_chunk_error": first_error,
                             "baseline_diagnostics": command_diagnostics(ref_rows, ref_chunks),
                             "diagnostics": t["diagnostics"]})
    causal = []
    for method in ("constant_noise", "commit10"):
        rows = [r for r in continuation if r["method"] == method]
        causal.append({"method": method, "eligible": len(rows), "baseline_successes": sum(r["baseline_success"] for r in rows),
                       "successes": sum(r["success"] for r in rows), "rescues": sum(r["rescued"] for r in rows),
                       "regressions": sum(r["regressed"] for r in rows),
                       "baseline_failures": sum(not r["baseline_success"] for r in rows)})
    result = {"schema": "research_v1_analysis", "primary_manifest": manifest,
              "input_runs": [str(Path(root))] + ([str(Path(extra))] if extra is not None else []),
              "supplementary_manifest": manifests[1] if len(manifests) > 1 else None,
              "audit": {"trials": len(trials), "skipped": len(skipped), "heldout_pairs": len(pairs),
                        "physical_state_clusters": len({(t["summary"]["suite"],t["summary"]["task"],t["summary"]["state"]) for t in heldout}),
                        "policy_seeds": sorted({t["summary"]["policy_seed"] for t in heldout}),
                        "provisional": provisional, "omitted_unpaired_heldout_trials": omitted,
                        "heldout_trials": len(heldout), "first_prediction_max_abs_error": initial_chunk_error,
                        "all_actions_and_seeds_validated": True, "all_videos_validated": video,
                        "causal_prefix_and_fork_checks": True},
              "summary": summary, "paired_statistics": statistics, "causal_summary": causal,
              "adaptive_decisions": {"interval_counts": dict(Counter(c["selected_k"] for c in adaptive_chunks)),
                                     "reason_counts": dict(Counter(c["reason"] for c in adaptive_chunks))},
              "analyzed_pair_ids": sorted(pairs),
              "artifact_sha256": {name: sha for t in trials for name, sha in t["artifact_sha256"].items()},
              "continuations": continuation, "skipped": skipped,
              "episode_summaries": [t["summary"] | t["diagnostics"] for t in trials]}
    return result, trials


def save_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_release_checks(result, trials, directory):
    import imageio.v2 as imageio

    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text())
    if status["phase"] != "completed":
        raise ValueError("Goal-persistence diagnostic is incomplete")
    release_manifest = json.loads((directory / "manifest.json").read_text())
    assert release_manifest["runtime"]["code_sha256"] == result["primary_manifest"]["runtime"]["code_sha256"]
    assert release_manifest["reference_manifests"] == [hashlib.sha256((Path(root)/"manifest.json").read_bytes()).hexdigest() for root in result["input_runs"]]
    assert (release_manifest["post_steps"],release_manifest["evaluation_tail_steps"],release_manifest["finger_open_threshold_m"]) == (40,10,.035)
    by_id = {t["summary"]["trial_id"]: t for t in trials if t["summary"]["phase"] == "heldout"}
    records = [json.loads(p.read_text()) for p in directory.glob("*/summary.json")]
    assert {r["trial_id"] for r in records} == set(by_id)
    for record in records:
        trial = by_id[record["trial_id"]]
        assert record["official_success"] == trial["summary"]["success"]
        path = Path(trial["directory"]) / "steps.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["reference_steps_sha256"]
        if record["official_success"]:
            rows = [json.loads(line) for line in (directory / record["trial_id"] / "steps.jsonl").read_text().splitlines()]
            assert [r["step"] for r in rows] == list(range(1, 41))
            stable = all(r["goal_success"] and len(r["gripper_qpos"]) == 2 and all(abs(q)>=.035 for q in r["gripper_qpos"]) for r in rows[-10:])
            assert record["replayed"] and record["post_open_stable_success"] == stable
            with imageio.get_reader(str(directory / record["trial_id"] / "video.mp4")) as reader:
                assert reader.get_meta_data()["fps"] == 20 and reader.count_frames() == 60
        else:
            assert not record["post_open_stable_success"]
    result["release_manifest"] = release_manifest
    result["release_directory"] = str(directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory)
    result["release_records"] = records
    for record in records:
        names = ["summary.json"] + (["steps.jsonl", "video.mp4"] if record["replayed"] else [])
        for name in names:
            path = directory / record["trial_id"] / name
            result["artifact_sha256"][str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["release_summary"] = [{"method": method,
        "episodes": sum(r["method"] == method for r in records),
        "official_successes": sum(r["method"] == method and r["official_success"] for r in records),
        "post_open_stable_successes": sum(r["method"] == method and r["post_open_stable_success"] for r in records)}
        for method in [r["method"] for r in result["summary"]]]


def add_replication(result, trials, directory, video=True):
    replication, new_trials = build_results(directory, video=video)
    for field in ("runtime", "hardware", "base_config"):
        assert replication["primary_manifest"][field] == result["primary_manifest"][field]
    assert not set(result["analyzed_pair_ids"]) & set(replication["analyzed_pair_ids"])
    physical = defaultdict(set)
    for t in new_trials:
        s = t["summary"]
        physical[(s["suite"],s["task"],s["state"])].add((s["initial_state_sha256"],s["initial_observation_sha256"],t["steps"][0]["physics_state_sha256"]))
    assert all(len(values) == 1 for values in physical.values())
    for row in result["episode_summaries"]:
        row["cohort"] = "primary_and_fixed_supplement"
    result["episode_summaries"].extend(dict(row, cohort="seed_replication") for row in replication["episode_summaries"])
    result["replication"] = {k:v for k,v in replication.items() if k not in ("episode_summaries", "artifact_sha256")}
    result["input_runs"].extend(replication["input_runs"])
    result["artifact_sha256"].update(replication["artifact_sha256"])
    result["audit"]["trials"] += replication["audit"]["trials"]
    result["audit"]["replication_trials"] = replication["audit"]["trials"]
    return trials + new_trials


def charts(result, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    rows = result["summary"]
    fig, ax = plt.subplots(figsize=(10.5, 5.3), constrained_layout=True)
    for r in rows:
        stat = next(s for s in result["paired_statistics"] if s["method"] == r["method"] and s["metric"] == "success")
        x, y = r["model_calls"], r["success"]*100
        error = None if stat["ci_low"] is None else np.array([[y-stat["ci_low"]*100], [stat["ci_high"]*100-y]])
        ax.errorbar(x, y, yerr=error, fmt="o", markersize=9,
                    color="#e36531" if r["method"] == "adaptive_v1" else "#2463a0", capsize=4)
        offset = 17 if r["method"] == "adaptive_v1" else -18 if r["method"] in ("fixed5", "fixed10", "fixed50") else 6
        ax.annotate(LABELS[r["method"]], (x, y), xytext=(8, offset), textcoords="offset points", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    ax.set(xlabel="Среднее число вызовов модели на эпизод (логарифмическая шкала)", ylabel="Успешность, %", xscale="log", ylim=(-5, 107),
           title=f"Качество и стоимость: {result['audit']['heldout_pairs']} парных ситуаций, один seed")
    ax.grid(alpha=.2)
    fig.savefig(out / "quality_cost.png", dpi=160)
    plt.close(fig)
    tasks = sorted({r["task"] for r in result["episode_summaries"] if r["phase"] == "heldout" and r["pair_id"] in result["analyzed_pair_ids"]})
    fig, axes = plt.subplots(1, len(tasks), figsize=(12, 4.2), sharey=True, constrained_layout=True)
    names = {"libero_object": "Банка → корзина", "libero_spatial": "Миска → тарелка", "libero_goal": "Открыть ящик"}
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks, strict=True):
        data = [r for r in result["episode_summaries"] if r["phase"] == "heldout" and r["task"] == task and r["pair_id"] in result["analyzed_pair_ids"]]
        values = [sum(r["success"] for r in data if r["method"] == method)/sum(r["method"] == method for r in data)*100 for method in [s["method"] for s in rows]]
        ax.bar(range(len(rows)), values, color=["#e36531" if r["method"] == "adaptive_v1" else "#2463a0" for r in rows])
        ax.set(xticks=range(len(rows)), xticklabels=[LABELS[r["method"]] for r in rows], ylim=(0, 110), title=names[data[0]["suite"]])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Успешность, %")
    fig.savefig(out / "tasks.png", dpi=160)
    plt.close(fig)


def contact_sheet(trials, offsets, filename):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (384*len(trials), 280*len(offsets)), "#eef2f7")
    font = ImageFont.load_default(size=13)
    for col, trial in enumerate(trials):
        s = trial["summary"]
        with imageio.get_reader(str(Path(trial["directory"]) / "video.mp4")) as reader:
            for row, step in enumerate(offsets):
                index = min(max(0, step-s["fork_step"]), s["executed_steps"])
                canvas.paste(Image.fromarray(reader.get_data(index)).resize((384, 256)), (384*col, 280*row+24))
                ImageDraw.Draw(canvas).text((384*col+8, 280*row+4), f"{s['method']} | step {min(step,s['steps'])}", font=font, fill="#15233b")
    canvas.save(filename)


def command_timeline(selected, fork, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for t, color in zip(selected, ("#26364c", "#3276ae", "#d85824"), strict=True):
        rows = [r for r in t["steps"] if r["step"] >= fork]
        axes[0].step([r["step"] for r in rows], [r["action"][6] for r in rows], where="post", label=t["summary"]["method"], color=color, alpha=.85)
        axes[1].plot([r["step"] for r in rows], [r["action_translation_jump_l2"] or 0 for r in rows], color=color, alpha=.75)
    chunk = next(c for c in selected[0]["chunks"] if c["step"] == fork)
    axes[0].step(range(fork, fork+50), np.clip(np.asarray(chunk["actions"])[:,6], -1, 1), where="post",
                 color="#8c8c8c", linestyle="--", linewidth=2, label="Первый общий план (предсказание)")
    axes[0].set(ylabel="Команда захвата", ylim=(-1.2,1.2), title="Первый общий план и реально исполненные команды после разветвления")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].set(xlabel="Абсолютный шаг симуляции", ylabel="Скачок команды\nперемещения")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def comparison_video(selected, fork, filename):
    import imageio.v2 as imageio
    import numpy as np

    readers = [imageio.get_reader(str(Path(t["directory"])/"video.mp4")) for t in selected]
    frames, indices = [None]*len(selected), [-1]*len(selected)
    try:
        with imageio.get_writer(str(filename), fps=20, codec="libx264", quality=7, ffmpeg_log_level="error") as writer:
            end = max(t["summary"]["steps"] for t in selected)
            for step in range(fork, end+20):
                for i,t in enumerate(selected):
                    s = t["summary"]
                    index = min(step-s["fork_step"], s["executed_steps"])
                    if index != indices[i]:
                        frames[i] = readers[i].get_data(index)
                        indices[i] = index
                writer.append_data(np.concatenate(frames, axis=1))
    finally:
        for reader in readers:
            reader.close()


def render_html(result, trials, out):
    out.mkdir(parents=True, exist_ok=True)
    charts(result, out)
    by_id = {t["summary"]["trial_id"]: t for t in trials}
    baseline = next(r for r in result["summary"] if r["method"] == "fixed10")
    adaptive = next(r for r in result["summary"] if r["method"] == "adaptive_v1")
    effect = next(r for r in result["paired_statistics"] if r["method"] == "adaptive_v1" and r["metric"] == "success")
    calls_change = (adaptive["model_calls"]/baseline["model_calls"]-1)*100
    intervals = "не оценивается: мало завершённых состояний" if effect['difference_ci_low'] is None else f"{effect['difference_ci_low']*100:+.1f}…{effect['difference_ci_high']*100:+.1f} п.п."
    provisional = '<p class="notice"><b>ПРЕДВАРИТЕЛЬНЫЙ ОТЧЁТ.</b> Серия ещё идёт. Показаны только полностью завершённые парные блоки; результаты могут измениться.</p>' if result["audit"]["provisional"] else ""

    def e(x):
        return html.escape(str(x))

    def image(path):
        return f'<img class="chart" src="data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}" alt="{e(Path(path).stem)}">'

    def video(t, start=0):
        path = Path(os.path.relpath(Path(t["directory"]) / "video.mp4", out))
        s = t["summary"]
        return f'<figure><video controls muted playsinline preload="metadata" data-start="{start}" src="{e(path)}"></video><figcaption>{e(s["method"])} · s{s["state"]} · {"успех" if s["success"] else "неудача"} · {s["model_calls"]} вызовов</figcaption></figure>'

    table = ''.join(f'<tr><td>{e(LABELS[r["method"]])}</td><td>{r["successes"]}/{r["episodes"]}</td><td>{r["model_calls"]:.1f}</td><td>{r["inference_s"]:.2f}</td><td>{r["compute_s"]:.2f}</td><td>{r["steps"]:.1f}</td></tr>' for r in result["summary"])
    causal_table = ''.join(f'<tr><td>{e(r["method"])}</td><td>{r["eligible"]}</td><td>{r["baseline_successes"]}</td><td>{r["successes"]}</td><td>{r["rescues"]}/{r["baseline_failures"]}</td><td>{r["regressions"]}</td></tr>' for r in result["causal_summary"])
    candidates = sorted((r for r in result["continuations"] if r["method"] == "commit10"), key=lambda r: (not r["rescued"], not r["regressed"], r["pair_id"]))
    media = ""
    if candidates:
        case = candidates[0]
        cont = by_id[case["trial_id"]]
        reference = by_id[cont["summary"]["reference_trial"]]
        fixed = next(t for t in trials if t["summary"]["pair_id"] == case["pair_id"] and t["summary"]["method"] == "constant_noise")
        selected = [reference, fixed, cont]
        endings = [t["summary"]["steps"] for t in selected]
        middle = min(endings) if min(endings) != max(endings) else (80+max(endings))//2
        contact_sheet(selected, [80, middle, max(endings)], out / "causal_frames.png")
        comparison_video(selected, 80, out / "causal_comparison.mp4")
        media = f'<h3>Иллюстрация разветвления: {e(case["suite"])} / состояние {case["state"]}</h3><p>Правило выбора: первый по идентификатору случай спасения commit10; затем регрессии; затем любой доступный случай. Это иллюстрация, вся выборка учтена в таблице выше. Единый ролик ниже: все три панели начинаются с шага 80, завершившийся эпизод показан последним кадром.</p>{image(out / "causal_frames.png")}<video controls muted playsinline preload="metadata" src="causal_comparison.mp4"></video><details><summary>Исходные ролики по отдельности</summary><p>Эталон начинается с шага 0, продолжения — с шага 80.</p><div class="videos">'+''.join(video(t) for t in selected)+'</div></details>'
    timeline_html = ""
    if candidates:
        diagnostic_case = max(candidates, key=lambda r: (r["baseline_diagnostics"]["deferred_transitions"], r["pair_id"]))
        committed = by_id[diagnostic_case["trial_id"]]
        original = by_id[committed["summary"]["reference_trial"]]
        constant = next(t for t in trials if t["summary"]["pair_id"] == diagnostic_case["pair_id"] and t["summary"]["method"] == "constant_noise")
        command_timeline([original, constant, committed], diagnostic_case["fork_step"], out / "command_timeline.png")
        timeline_html = f'<p>Для временной диаграммы выбран случай с наибольшим числом отложенных смен команды в эталонном суффиксе: {e(diagnostic_case["suite"])} / s{diagnostic_case["state"]}. Пунктир — общий первый предсказанный блок, остальные линии — реально исполненные команды. Выбор служит иллюстрации, не оценке частоты ошибок.</p>{image(out / "command_timeline.png")}'
    pairs = defaultdict(dict)
    for t in trials:
        if t["summary"]["phase"] == "heldout" and t["summary"]["pair_id"] in result["analyzed_pair_ids"]:
            pairs[t["summary"]["pair_id"]][t["summary"]["method"]] = t
    pair_key = sorted(pairs, key=lambda key: (pairs[key]["fixed10"]["summary"]["success"] == pairs[key]["adaptive_v1"]["summary"]["success"], key))[0]
    selected = [pairs[pair_key]["fixed10"], pairs[pair_key]["adaptive_v1"]]
    endings = [t["summary"]["steps"] for t in selected]
    contact_sheet(selected, [0, min(endings), max(endings)], out / "adaptive_frames.png")
    adaptive_media = f'<p>Первый по идентификатору случай с разными исходами K=10 и adaptive; при отсутствии различий — первый случай.</p>{image(out / "adaptive_frames.png")}<div class="videos">'+''.join(video(t) for t in selected)+'</div>'
    losses = sorted(key for key, pair in pairs.items() if pair["fixed10"]["summary"]["success"] and not pair["adaptive_v1"]["summary"]["success"])
    if losses:
        losing = [pairs[losses[0]]["fixed10"], pairs[losses[0]]["adaptive_v1"]]
        endings = [t["summary"]["steps"] for t in losing]
        contact_sheet(losing, [0, min(endings), max(endings)], out / "adaptive_loss_frames.png")
        adaptive_media += '<details><summary>Случай проигрыша адаптивного правила</summary><p>Первый по идентификатору случай, где K=10 успешен, а adaptive — нет.</p>'+image(out / "adaptive_loss_frames.png")+'<div class="videos">'+''.join(video(t) for t in losing)+'</div></details>'
    diag_rows = []
    for method in [r["method"] for r in result["summary"]]:
        data = [r for r in result["episode_summaries"] if r["phase"] == "heldout" and r["method"] == method and r["pair_id"] in result["analyzed_pair_ids"]]
        denom = sum(r["evaluable_transitions"] for r in data)
        missed = sum(r["deferred_transitions"] for r in data)
        diag_rows.append(f'<tr><td>{e(LABELS[method])}</td><td>{missed}/{denom}</td><td>{sum(r["gripper_sign_changes"] for r in data)/len(data):.1f}</td><td>{sum(r["mean_action_jump"] for r in data)/len(data):.3f}</td></tr>')
    release_html = ""
    if result.get("release_summary"):
        release_rows = ''.join(f'<tr><td>{e(LABELS[r["method"]])}</td><td>{r["official_successes"]}/{r["episodes"]}</td><td>{r["post_open_stable_successes"]}/{r["episodes"]}</td></tr>' for r in result["release_summary"])
        release_html = f'<section><h2>Сохраняется ли успех после открытия захвата?</h2><p>Успешные оценочные эпизоды повторены с проверкой каждой физической позиции и изображения. Затем выполнены 40 команд без перемещения с открытым захватом — две секунды симуляции. Дополнительный показатель требует сохранения цели и открытия обоих пальцев на последних десяти шагах. Официальная оценка LIBERO остаётся отдельной. Это диагностическое вмешательство после остановки политики.</p><table><tr><th>Метод</th><th>Стандартный успех</th><th>Цель сохраняется с открытыми пальцами</th></tr>{release_rows}</table></section>'
    commands = '.venv/bin/python scripts/run_research.py --plan configs/research_v1.json --output outputs/research_v1\n'
    extra_flag = ' --extra outputs/research_v1_extra' if result["supplementary_manifest"] else ''
    if extra_flag:
        commands += '.venv/bin/python scripts/run_research.py --plan configs/research_v1_extra_baselines.json --output outputs/research_v1_extra\n'
    if result.get("replication"):
        commands += '.venv/bin/python scripts/run_research.py --plan configs/research_v1_seed_replication.json --output outputs/research_v1_replication\n'
    if result.get("release_summary"):
        commands += '.venv/bin/python scripts/check_release_stability.py --run outputs/research_v1'+extra_flag+'\n'
    commands += '.venv/bin/python scripts/analyze_research.py --run outputs/research_v1'+extra_flag+(' --release-checks outputs/research_v1/release_checks' if result.get("release_summary") else '')+(' --replication outputs/research_v1_replication' if result.get("replication") else '')
    release_limit = 'Проверка открытия захвата описана отдельно выше.' if result.get("release_summary") else 'Дополнительная проверка отпускания и устойчивости не выполнена.'
    future_limit = 'Другие модели, возмущения и асинхронное исполнение остаются отдельными этапами.' if result.get("replication") else 'Дополнительные seed, возмущения и асинхронное исполнение остаются отдельными этапами.'
    histogram = result["adaptive_decisions"]["interval_counts"]
    decision_text = ", ".join(f"K={k}: {histogram.get(k, 0)}" for k in (5,10,25))
    replication_html = ""
    if result.get("replication"):
        rep = result["replication"]
        rep_rows = ''.join(f'<tr><td>{e(LABELS[r["method"]])}</td><td>{r["successes"]}/{r["episodes"]}</td><td>{r["model_calls"]:.1f}</td></tr>' for r in rep["summary"])
        rep_effect = next(r for r in rep["paired_statistics"] if r["metric"] == "success" and r["method"] == "adaptive_v1")
        replication_html = f'<section><h2>Повторение на других состояниях и seed</h2><p>Отдельная серия после первичной оценки: состояния 8–9 каждой задачи и seed 29,47. Правило и пороги не менялись. Это <b>{rep["audit"]["physical_state_clusters"]} физических состояний</b> с двумя повторами каждое; они не добавлены к основной таблице всех K. Bootstrap усредняет seed внутри состояния.</p><table><tr><th>Метод</th><th>Успехи</th><th>Среднее число вызовов</th></tr>{rep_rows}</table><p>Парная разность успешности adaptive − K=10: {rep_effect["paired_difference"]*100:+.1f} п.п.; интервал {rep_effect["difference_ci_low"]*100:+.1f}…{rep_effect["difference_ci_high"]*100:+.1f} п.п. Выборка мала и условна на этих задачах.</p></section>'
    html_text = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLA: адаптивное перепланирование и причины ошибок</title>
<style>:root{{--ink:#15233b;--muted:#5c6c82;--paper:#f2f5f9;--orange:#d85824}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 24px 80px}}header{{background:#15233b;color:white;border-radius:22px;padding:42px}}h1{{font-size:clamp(30px,4vw,48px);line-height:1.15;margin:12px 0 22px}}h2{{margin:0 0 14px;font-size:28px}}h3{{font-size:20px}}section{{background:white;padding:30px;margin-top:24px;border-radius:18px;overflow:auto}}.kicker{{color:#ffbe89;letter-spacing:.12em;font-size:13px;text-transform:uppercase}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}}.card{{padding:20px;background:white;border-radius:16px}}.big{{font-size:30px;font-weight:750;display:block}}.muted,figcaption{{color:var(--muted);font-size:14px}}.notice{{border-left:4px solid var(--orange);background:#fff4ec;padding:16px 20px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:15px}}td,th{{text-align:left;padding:12px 10px;border-bottom:1px solid #dbe3ee}}th{{background:#edf2f8}}.chart{{width:100%;height:auto;border-radius:10px;margin:14px 0}}.videos{{display:flex;gap:14px;flex-wrap:wrap}}figure{{margin:0;flex:1;min-width:240px}}video{{width:100%;border-radius:10px}}a{{color:#1760a3}}code{{font-size:14px;background:#eaf0f7;padding:2px 5px}}pre{{background:#15233b;color:#e4eefc;padding:18px;white-space:pre-wrap;border-radius:10px}}@media(max-width:650px){{main{{padding:18px 12px}}header,section{{padding:24px 18px}}.cards{{grid-template-columns:1fr}}table{{font-size:13px}}}}</style>
<main><header><div class="kicker">SmolVLA · LIBERO · 24 октября 2026</div><h1>Когда роботу<br>перепланировать движение?</h1><p>Новая проверка фиксированного и адаптивного K и контролируемые продолжения K=1 из одной и той же точки.</p></header>{provisional}
<div class="cards"><div class="card"><span class="big">{result['audit']['trials']}</span>проверенных реальных прогонов</div><div class="card"><span class="big">{adaptive['successes']}/{adaptive['episodes']}</span>успехов adaptive на оценочной выборке</div><div class="card"><span class="big">{calls_change:+.1f}%</span>вызовов adaptive относительно K=10</div></div>
<section><h2>Что показал эксперимент</h2><p>Адаптивное правило: <b>{adaptive['successes']}/{adaptive['episodes']}</b> успехов; K=10: <b>{baseline['successes']}/{baseline['episodes']}</b>. Разность успешности: {(adaptive['success']-baseline['success'])*100:+.1f} п.п.; 95% парный bootstrap-интервал <b>{intervals}</b>.</p><p class="notice">Малая исследовательская выборка: план включает три фиксированные задачи, четыре состояния на задачу и один seed. Интервалы условны на выбранных задачах и могут быть вырожденными. Отсутствие различия не устанавливает эквивалентность; результат не определяет универсальный лучший K.</p>{image(out/'quality_cost.png')}</section>
<section><h2>Качество и вычисления</h2><p>Только оценочные состояния 4–7, равные веса задач и состояний. Неудачи включены в средние. Compute = инференс + шаги среды; загрузка, сброс, запись видео и хеширование исключены.</p><table><tr><th>Метод</th><th>Успехи</th><th>Вызовы</th><th>Инференс, с</th><th>Compute, с</th><th>Шаги</th></tr>{table}</table>{image(out/'tasks.png')}</section>
<section><h2>Как выбирается K</h2><p>При смене знака команды захвата в следующих десяти предсказаниях или повороте ненулевых векторов перемещения более чем на 60° выбирается <b>K=5</b>. Иначе при команде закрытия — <b>K=10</b>, при открытом захвате — <b>K=25</b>. Горизонт всегда H=50. Дополнительных генераций нет; координаты предмета, контакты и успех симулятора правилу недоступны.</p><p>Выборы в оценочной выборке: <b>{decision_text}</b>.</p><p>Правило и выборка зафиксированы до этих результатов. Разработка использует состояния 2–3; они не включены в основную таблицу. Пороговые значения по результатам не менялись. Это простой baseline рядом с существующим <a href="https://arxiv.org/abs/2604.04161">Adaptive Action Chunking</a>.</p>{adaptive_media}</section>
<section><h2>Почему K=1 может проигрывать</h2><p>Для заранее выбранных состояний 4,5,6 каждой задачи повторены первые 80 команд обычного K=1. Сверены все изображения и полные состояния MuJoCo до точки разветвления. Первая полная генерация 50×7 также совпала с исходной в пределах 2×10⁻⁶.</p><p><b>constant_noise:</b> новые наблюдения каждый шаг, один и тот же шум генерации. <b>commit10:</b> исполнение десяти действий предсказанного блока. Обычный K=1 берётся из записанного эталона. Включены и успешные, и неуспешные эталоны; завершившиеся до шага 80 явно исключены как неподходящие.</p><table><tr><th>Продолжение</th><th>Случаев</th><th>Успехов K=1</th><th>Успехов после</th><th>Спасено из неудач</th><th>Регрессий</th></tr>{causal_table}</table><p class="muted">Эффекты относятся к этим состояниям и точке 80. Исполнение commit10 одновременно меняет частоту наблюдений и сохраняет продолжение плана: это не изолированный тест единственной физической причины.</p>{media}</section>
<section><h2>Что видно в командах</h2><p>«Отложенная смена захвата» — модель предсказала смену знака в пределах следующих девяти действий, но она не была исполнена к предсказанному шагу. Учитываются только моменты, для которых есть последующие записанные действия. Это диагностический показатель команд, не число ошибок захвата или падений предмета. Предсказания перекрываются во времени, поэтому события не независимы. Скачок измеряется в координатах команд управления.</p><table><tr><th>Метод</th><th>Отложено / можно проверить</th><th>Смен знака / эпизод</th><th>Средний скачок команды</th></tr>{''.join(diag_rows)}</table>{timeline_html}</section>{replication_html}{release_html}
<section><h2>Проверка и воспроизведение</h2><p>Проверены все сохранённые действия, полные 50×7 предсказания, интервалы исполнения, seed, начальные условия и количество кадров видео на 20 fps. Максимальное различие первой генерации между методами: <b>{result['audit']['first_prediction_max_abs_error']:.2g}</b>. Пропущено веток из-за раннего завершения эталона: {len(result['skipped'])}. Незавершённых оценочных эпизодов вне полных парных блоков: {result['audit']['omitted_unpaired_heldout_trials']}.</p><p>Checkpoint: <code>{e(result['primary_manifest']['base_config']['model_revision'])}</code>; код: <code>{e(result['primary_manifest']['runtime']['code_sha256'])}</code>. Более ранние 30- и 300-эпизодные серии представлены отдельно и не добавлены к этой статистике.</p><pre>{e(commands)}</pre><p><a href="analysis.json">Полные результаты и provenance (JSON)</a> · <a href="episodes.csv">Таблица эпизодов</a> · <a href="paired_statistics.csv">Парные интервалы</a></p><p class="muted">Симулятор ждёт модель; реальная задержка управления не проверялась. Успех — стандартный LIBERO. {release_limit} Контактные силы не использованы для установления причины. {future_limit}</p></section></main>
</html>'''
    (out / "index.html").write_text(html_text)


def write_markdown(result, out):
    adaptive = next(r for r in result["summary"] if r["method"] == "adaptive_v1")
    reference = next(r for r in result["summary"] if r["method"] == "fixed10")
    delta = next(r for r in result["paired_statistics"] if r["method"] == "adaptive_v1" and r["metric"] == "success")
    plan = result["primary_manifest"]["plan"]
    states = ",".join(map(str, plan["heldout_states"]))
    seeds = ",".join(map(str, plan["policy_seeds"]))
    planned_cases = len(result["primary_manifest"]["base_config"]["tasks"])*len(plan["causal_states"])*len(plan["policy_seeds"])
    skipped_cases = len({row["trial"].rsplit("__",1)[0] for row in result["skipped"]})
    eligible = result["causal_summary"][0]["eligible"]
    if adaptive["success"] <= reference["success"] and adaptive["model_calls"] >= reference["model_calls"]:
        verdict = "The frozen adaptive heuristic did not improve the primary sample."
    elif adaptive["success"] >= reference["success"] and adaptive["model_calls"] <= reference["model_calls"]:
        verdict = "The frozen heuristic has favorable point estimates here; uncertainty still limits the conclusion."
    else:
        verdict = "The frozen heuristic trades observed success against computational cost in this sample."
    lines = ["# Adaptive replanning follow-up — 24 October 2026", "",
        f"{result['audit']['trials']} new VLA rollouts across the separately described cohorts below. "
        "The original 30-episode and previous 300-episode datasets are not pooled into these estimates.", "",
        "## Primary evaluation", "",
        f"Three fixed tasks, states {states}, policy seed(s) {seeds}: {result['audit']['heldout_pairs']} paired situations. H=50, ten denoising steps, "
        "FP32 and 20 Hz are constant. The adaptive rule uses only predicted actions. Development states 2–3 are excluded.", "",
        "| Method | Successes | Mean model calls | Mean inference, s | Mean model + simulation, s |",
        "|---|---:|---:|---:|---:|"]
    lines += [f"| {LABELS[r['method']]} | {r['successes']}/{r['episodes']} | {r['model_calls']:.2f} | {r['inference_s']:.2f} | {r['compute_s']:.2f} |" for r in result["summary"]]
    lines += ["", f"Adaptive v1 achieved {adaptive['successes']}/{adaptive['episodes']} successes versus "
        f"{reference['successes']}/{reference['episodes']} for K=10, with {adaptive['model_calls']:.2f} versus "
        f"{reference['model_calls']:.2f} mean calls. Its paired success difference is {delta['paired_difference']*100:+.1f} pp "
        f"(95% task-stratified state-bootstrap interval {delta['difference_ci_low']*100:+.1f} to {delta['difference_ci_high']*100:+.1f} pp). "
        "This small conditional sample does not establish universal superiority, inferiority or equivalence.", "",
        verdict+" It can cut a chunk at K=5 "
        "even when the predicted gripper transition is at index 5–9, replacing the plan before that transition. "
        "This is a possible mechanism to test, not a physical failure label. Thresholds were not retuned after the outcomes.", "",
        "## Controlled continuations", "",
        f"{planned_cases} cases were specified before outcomes. {skipped_cases} reference episodes ended before step 80, leaving {eligible} eligible cases. "
        "Every replayed observation and full MuJoCo state matched, and the first 50×7 prediction matched within 2e-6.", "",
        "| Continuation | Reference successes | New successes | Rescued reference failures | Regressions |",
        "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['baseline_successes']}/{r['eligible']} | {r['successes']}/{r['eligible']} | {r['rescues']}/{r['baseline_failures']} | {r['regressions']} |" for r in result["causal_summary"]]
    lines += ["", "Constant noise retains fresh observations at K=1. Commit10 executes ten actions from each new chunk. "
        "The table records rescues and regressions for the whole eligible sample, including successful references. "
        "Changing K after a fixed prefix is a conditional intervention, not the same experiment as choosing K from step zero. "
        "Earlier selected-case rescues are preserved in `results/previous_300/selected_failure_probes.json`, "
        "with their different checkpoint, noise protocol and fork steps explicitly documented.", "",
        "Gripper deferral and command revision are descriptive diagnostics. For K>=10, executing the first ten actions "
        "mechanically prevents deferral of transitions at indices 1–9; zero deferral is not an independent quality result. "
        "Repeated predictions overlap and are not independent statistical events. Contact forces were not recorded."]
    if result.get("replication"):
        rep = result["replication"]
        lines += ["", "## Unchanged-rule replication", "",
            "Separate follow-up on states 8–9 and seeds 29,47: six physical states with two seeds each. "
            "It was specified after the primary evaluation and before its own outcomes. The rule and thresholds remain unchanged. "
            "Seed repeats stay inside the state cluster; this cohort is not pooled into the all-K primary table.", "",
            "| Method | Successes | Mean calls |", "|---|---:|---:|"]
        lines += [f"| {LABELS[r['method']]} | {r['successes']}/{r['episodes']} | {r['model_calls']:.2f} |" for r in rep["summary"]]
    if result.get("release_summary"):
        lines += ["", "## Goal persistence after opening the gripper", "",
            "Successful primary/supplemental episodes were replayed exactly, then followed by 40 zero-motion/open-gripper "
            "commands. The extra score requires goal success and both finger |qpos|>=0.035 m on all final ten steps. "
            "Official failures are false by definition. This is a post-success diagnostic, not a changed LIBERO score "
            "or a physical-robot stability guarantee.", "", "| Method | Official successes | Post-open persistent goal |", "|---|---:|---:|"]
        lines += [f"| {LABELS[r['method']]} | {r['official_successes']}/{r['episodes']} | {r['post_open_stable_successes']}/{r['episodes']} |" for r in result["release_summary"]]
    lines += ["", "## Evidence and reproduction", "",
        "[Interactive report with videos](index.html), [full analysis/provenance](analysis.json), "
        "[per-episode CSV](episodes.csv), [paired bootstrap estimates](paired_statistics.csv).", "",
        f"Runner source SHA-256: `{result['primary_manifest']['runtime']['code_sha256']}`.", "",
        "All saved actions, full chunks, sampling seeds, paired initial conditions and 20 fps frame counts were audited. "
        "Raw logs/videos are in the portable evidence archive or can be regenerated using the commands in the HTML report. "
        "The simulator pauses for inference; real-time latency, perturbations and other model families remain untested. "
        "Video illustrations are selected examples, not an independent annotation sample.", ""]
    (out / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-video-audit", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true", help="Clearly marked preview; only complete paired blocks")
    parser.add_argument("--release-checks", type=Path)
    parser.add_argument("--replication", type=Path)
    args = parser.parse_args()
    out = args.output or args.run / "report"
    result, trials = build_results(args.run, args.extra, not args.skip_video_audit, args.allow_incomplete)
    if args.release_checks:
        add_release_checks(result, trials, args.release_checks)
    if args.replication:
        trials = add_replication(result, trials, args.replication, not args.skip_video_audit)
    result["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "analysis.json", result)
    save_csv(out / "episodes.csv", result["episode_summaries"])
    save_csv(out / "paired_statistics.csv", result["paired_statistics"])
    render_html(result, trials, out)
    if result["audit"]["provisional"]:
        (out / "README.md").write_text("# Provisional preview\n\nThe study is incomplete. See index.html for explicitly marked paired-block results.\n")
    else:
        write_markdown(result, out)
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["causal_summary"], indent=2))


if __name__ == "__main__":
    main()
