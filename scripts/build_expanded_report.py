"""Build an eight-page English report after all 150 real episodes are complete.

The main analysis uses all 150 episodes with matched initial states across K.
The original run manifests remain separate for provenance. No model is run and no
result or interpretation is fabricated by this script.

Usage: python scripts/build_expanded_report.py --run outputs/expanded \
    --discussion outputs/expanded/discussion.json --overwrite

Run the PDF skill's authoring marker before producing a PDF. This script does not
run that marker automatically. Render and visually inspect all pages after building.
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_expanded_figures import (  # noqa: E402
    FIGURES,
    KS,
    PHASES,
    TASKS,
    load_bundle,
    mean,
    publish,
    read_json_snapshot,
    sha256,
    validate_expanded_discussion,
    verify_unchanged,
)
from scripts.build_report import (  # noqa: E402
    BLUE,
    INK,
    MUTED,
    PALE,
    TEAL,
    clean,
    font_paths,
    num,
)


def _figure_paths(bundle, snapshots):
    directory = bundle["run"] / "report"
    record_path = directory / "expanded-figures.json"
    record, snapshots[record_path] = read_json_snapshot(record_path)
    figure_generator = Path(__file__).with_name("build_expanded_figures.py").resolve()
    snapshots[figure_generator] = sha256(figure_generator)
    if (record["statistics_sha256"] != bundle["statistics_sha256"]
            or record["generator_sha256"] != snapshots[figure_generator]):
        raise ValueError("Figures are stale; rerun scripts/build_expanded_figures.py")
    expected = {f"{name}.{extension}" for name in FIGURES for extension in ("png", "pdf")}
    if set(record["files"]) != expected or record.get("primary_phase") != "combined":
        raise ValueError("Figure provenance is incomplete")
    for name, digest in record["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Figure changed since publication: {name}")
        snapshots[directory / name] = digest
    state_record_path = directory / "state-outcomes.json"
    state_record, snapshots[state_record_path] = read_json_snapshot(state_record_path)
    state_generator = Path(__file__).with_name("build_state_outcomes.py").resolve()
    snapshots[state_generator] = sha256(state_generator)
    if (state_record.get("statistics_sha256") != bundle["statistics_sha256"]
            or state_record.get("generator_sha256") != snapshots[state_generator]
            or state_record.get("primary_phase") != "combined"
            or state_record.get("artifact_kind") != "state_outcomes"
            or set(state_record.get("files", {})) != {"state-outcomes.png", "state-outcomes.pdf"}):
        raise ValueError("State-outcome figure is stale or incomplete; rerun scripts/build_state_outcomes.py")
    for name, digest in state_record["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"State-outcome figure changed since publication: {name}")
        snapshots[directory / name] = digest
    return {name: directory / f"{name}.png" for name in (*FIGURES, "state-outcomes")}


def _discussion_size(data):
    values = []
    for case in data["cases"]:
        values.extend(case[key] for key in ("title", "observation", "hypothesis", "evidence"))
        values.extend(case.get("caveats", []))
        values.extend(case.get("protocol_refs", []))
    return sum(map(len, values))


def build_report(args):
    generator = Path(__file__).resolve()
    snapshots = {generator: sha256(generator)}
    bundle = load_bundle(args.run, args.pilot, args.validation)
    snapshots[bundle["statistics_path"]] = bundle["statistics_sha256"]
    stats, rows, labels = (bundle[key] for key in ("stats", "rows", "labels"))
    discussion, snapshots[args.discussion.resolve()] = read_json_snapshot(args.discussion)
    validate_expanded_discussion(discussion, rows["combined"])
    for phase in ("pilot", "validation"):
        selected = [label for label in labels[phase] if label["selected_for_visual_review"]]
        if len(selected) != 30 or any(label["inspected"] is not True for label in selected):
            raise ValueError(f"{phase}: the final report requires inspection evidence for all 30 selected recordings")
    if _discussion_size(discussion) > 3400:
        raise ValueError("Keep case prose, including optional metadata, within 3400 characters for one readable page")
    if sum(map(len, discussion["findings"])) + len(discussion["conclusion"]) > 2500:
        raise ValueError("Keep findings plus conclusion within 2500 characters for the findings page")
    charts = _figure_paths(bundle, snapshots)
    output = args.output.resolve()
    provenance_output = output.with_suffix(".json")
    if output.suffix.lower() != ".pdf":
        raise ValueError("The final report output must have a .pdf suffix")
    protected_inputs = [bundle["statistics_path"], args.discussion.resolve(),
                        bundle["run"] / "report/expanded-figures.json", bundle["run"] / "report/state-outcomes.json"]
    if output in protected_inputs or provenance_output in protected_inputs:
        raise ValueError("Report output and provenance must not overwrite their inputs")
    if (output.exists() or provenance_output.exists()) and not args.overwrite:
        raise FileExistsError("Final PDF or provenance exists; pass --overwrite to rebuild")
    # Prevent accidental replacement of raw data or either phase's original report.
    if any(output.is_relative_to(source["root"]) for source in bundle["sources"].values()):
        raise ValueError("The expanded PDF must be outside both original source-run directories")

    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Flowable,
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    regular, bold = font_paths(args.font, args.font_bold)
    pdfmetrics.registerFont(TTFont("Expanded", regular))
    pdfmetrics.registerFont(TTFont("ExpandedBold", bold))
    pdfmetrics.registerFontFamily("Expanded", normal="Expanded", bold="ExpandedBold", italic="Expanded")
    width, height = A4
    body_width = width - 36 * mm
    styles = {
        "body": ParagraphStyle("body", fontName="Expanded", fontSize=9.2, leading=12.5,
                               textColor=colors.HexColor(INK), spaceAfter=6),
        "small": ParagraphStyle("small", fontName="Expanded", fontSize=8, leading=10.5,
                                textColor=colors.HexColor(MUTED), spaceAfter=5),
        "title": ParagraphStyle("title", fontName="ExpandedBold", fontSize=25, leading=29,
                                textColor=colors.HexColor(BLUE), spaceAfter=10),
        "h1": ParagraphStyle("h1", fontName="ExpandedBold", fontSize=18, leading=22,
                             textColor=colors.HexColor(BLUE), spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="ExpandedBold", fontSize=11, leading=14,
                             textColor=colors.HexColor(TEAL), spaceBefore=7, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="Expanded", fontSize=7.9, leading=10.4,
                               textColor=colors.HexColor(INK), wordWrap="CJK"),
        "head": ParagraphStyle("head", fontName="ExpandedBold", fontSize=7.9, leading=10.4,
                               textColor=colors.white),
        "code": ParagraphStyle("code", fontName="Expanded", fontSize=7.3, leading=10,
                               textColor=colors.HexColor(INK), wordWrap="CJK", spaceAfter=7,
                               backColor=colors.HexColor(PALE), borderPadding=5),
        "kicker": ParagraphStyle("kicker", fontName="ExpandedBold", fontSize=8.4, leading=11,
                                 textColor=colors.HexColor(TEAL), spaceAfter=8),
    }

    def p(text, style="body", markup=False):
        return Paragraph(str(text) if markup else clean(text), styles[style])

    def table(headers, data, fractions):
        result = Table([[p(v, "head") for v in headers]] + [[p(v, "cell") for v in row] for row in data],
                       colWidths=[body_width * value for value in fractions], repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(PALE), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return result

    def figure(name, caption):
        w, h = (13.2, 6.8) if name == "state-outcomes" else FIGURES[name]
        return [Spacer(1, 8), Image(str(charts[name]), width=body_width, height=body_width * h / w),
                p(caption, "small")]

    def interval(estimate, scale=1, digits=1, difference=False):
        low, high = ("difference_ci_low", "difference_ci_high") if difference else ("ci_low", "ci_high")
        return f"[{num(scale * estimate[low], digits)}, {num(scale * estimate[high], digits)}]"

    def counts(group):
        return f"{sum(r['success'] for r in group)}/{len(group)}"

    def coverage_value(value):
        return "unknown" if value is None else str(value)

    def event_count(item, field):
        known, total = item[f"{field}_known_episodes"], item[f"{field}_total_known"]
        return f"{coverage_value(total)} [{known}/{item['total_episodes']}]"

    class ReplanDiagram(Flowable):
        def __init__(self):
            super().__init__()
            self.width, self.height = body_width, 54

        def draw(self):
            canvas = self.canv
            box_width, gap = (self.width - 3 * 11) / 4, 11
            items = [("OBSERVATION", "2 cameras + state + instruction"),
                     ("PREDICT", "H = 50 actions"), ("EXECUTE", "First K actions"),
                     ("REPLAN", "New observation; replace tail")]
            for index, (title, subtitle) in enumerate(items):
                x = index * (box_width + gap)
                canvas.setFillColor(colors.HexColor(PALE))
                canvas.roundRect(x, 8, box_width, 42, 5, fill=1, stroke=0)
                canvas.setFillColor(colors.HexColor(TEAL))
                canvas.setFont("ExpandedBold", 8)
                canvas.drawCentredString(x + box_width / 2, 34, title)
                canvas.setFillColor(colors.HexColor(INK))
                canvas.setFont("Expanded", 6.9)
                canvas.drawCentredString(x + box_width / 2, 20, subtitle)
                if index < 3:
                    canvas.drawCentredString(x + box_width + gap / 2, 26, ">")

    cfg = bundle["sources"]["validation"]["config"]
    estimates = {phase: {(r["metric"], r["k"]): r for r in stats["phases"][phase]["estimates"]}
                 for phase in PHASES}
    by_k = {phase: {k: [r for r in rows[phase] if r["k"] == k] for k in KS} for phase in PHASES}
    by_id = {r["episode_id"]: r for r in rows["combined"]}
    source = stats["provenance"]
    draw_count, reference = source["bootstrap_draws"], source["reference_k"]
    story = []

    # Page 1: one complete study, fixed inference policy and matched starts.
    story.extend([
        p("SMOLVLA / LIBERO   |   150-EPISODE EXPERIMENT", "kicker"),
        p("How often should a robot replan?", "title"),
        p("Three tasks, ten initial states, five replanning intervals", "h2"),
        p("We vary the number of actions K executed before obtaining a fresh observation and replacing the plan. "
          "The study measures task completion, motion, grasping behavior, model calls and runtime for one pretrained model."),
        ReplanDiagram(),
        p("Experimental design", "h2"),
        table(["Factor", "Setting", "Comparison"], [
            ["Tasks", "Drawer, soup can, black bowl", "Three different manipulation behaviors"],
            ["Initial states", "0-9 for each task", "The same ten states for every K"],
            ["Replanning interval", "K = 1, 5, 10, 25, 50", "30 episodes per K; 150 in total"],
        ], [0.22, 0.35, 0.43]),
        Spacer(1, 7),
        p("All 150 episodes contribute to the main analysis. Each setting has the same task and initial-state "
          "coverage; differences reflect the executed prefix length K under the fixed protocol.", "small"),
        table(["Task", "Suite", "Step limit"],
              [[TASKS[(t.suite, t.name)], t.suite, t.max_steps] for t in cfg.tasks], [0.53, 0.29, 0.18]),
        p("Model and controlled conditions", "h2"),
        p(f"SmolVLA uses flow matching to predict H={cfg.chunk_size} actions from two images, robot state and a language "
          "instruction. Each action specifies relative end-effector translation, rotation and a gripper command. "
          "K is an executed prefix length, not repeated execution of one action."),
        p(f"K=1,5,10,25,50; one policy seed ({cfg.policy_seeds[0]}); {cfg.control_hz} Hz simulated control; "
          f"{cfg.num_denoising_steps} generation iterations; batch=1; precision {cfg.precision}; "
          f"{cfg.settle_steps} settling steps; {cfg.warmup_calls} warmup calls. Checkpoint, preprocessing, simulator, "
          "hardware and logging are held fixed. Physical initial states and initial observations match across K by SHA-256.", "small"),
        p("Run order is randomized within each source run. Model sampling noise is tied to policy seed and "
          "simulation step. The simulator pauses during inference.", "small"),
        PageBreak(),
    ])

    # Page 2: all 150 episodes form the main result.
    success_rows, difference_rows = [], []
    for k in KS:
        selected = by_k["combined"][k]
        entry = estimates["combined"][("success", k)]
        task_counts = [counts([r for r in selected if (r["suite"], r["task"]) == identity]) for identity in TASKS]
        success_rows.append([k, *task_counts, counts(selected), interval(entry, 100, 1) + "%"])
        difference_rows.append([k, f"{100 * entry['paired_difference']:+.1f}", interval(entry, 100, 1, True)])
    story.extend([
        p("Task completion across 150 episodes", "h1"),
        p("3 tasks x 10 initial states x 5 K values. Each K has 30 episodes; each task/K has 10. "
          "Success is LIBERO's native signal before the step limit."),
        table(["K", "Drawer", "Soup can", "Bowl", "Total", "95% interval"], success_rows,
              [0.07, 0.16, 0.16, 0.16, 0.15, 0.30]),
        *figure("study-by-task", "Figure 1. Success by task across all ten initial states. Every K is evaluated from the same states within each task."),
        p(f"Matched contrasts against K={reference}", "h2"),
        table(["K", "Success difference, percentage points", "95% paired interval"], difference_rows,
              [0.10, 0.48, 0.42]),
        Spacer(1, 7),
        p(f"Means weight tasks equally, then initial states equally. {draw_count:,} bootstrap resamples draw initial-state "
          "clusters with replacement within each task; all K outcomes stay paired in every draw. The contrast interval "
          "is computed from paired differences, not by subtracting two marginal interval bounds. Tasks and the policy "
          "seed remain fixed. Intervals are exploratory and are not adjusted for multiple comparisons.", "small"),
        PageBreak(),
    ])

    # Page 3: show all matched outcomes instead of splitting the main study.
    discordance = []
    reference_outcomes = {(r["suite"], r["task"], r["state"]): r["success"]
                          for r in by_k["combined"][reference]}
    for k in KS:
        paired = [(r["success"], reference_outcomes[r["suite"], r["task"], r["state"]])
                  for r in by_k["combined"][k]]
        discordance.append([k, sum(a and not b for a, b in paired),
                            sum(not a and b for a, b in paired), sum(a == b for a, b in paired)])
    story.extend([
        p("Which initial situations change outcome?", "h1"),
        p("Each row below holds the task and initial state fixed while K changes. Each cell shows one native "
          "LIBERO outcome, not a probability. Reading across a row makes disagreements visible that an aggregate "
          "success percentage can hide."),
        *figure("state-outcomes", "Figure 2. All 150 episode outcomes. Green cells reach the simulator's success condition; red cells reach the step limit without success."),
        p(f"Paired outcomes compared with K={reference}", "h2"),
        table(["K", "K succeeds; reference fails", "K fails; reference succeeds", "Same outcome"],
              discordance, [0.08, 0.32, 0.34, 0.26]),
        Spacer(1, 7),
        p("No interval succeeds in every initial situation. The result depends on the task and initial geometry; "
          "different settings can have similar totals while failing in different scenes. The paired intervals on "
          "page 2 quantify uncertainty across these states, conditional on the fixed tasks and policy seed."),
        p("The examples on page 6 examine visible behavior in selected recordings. They illustrate particular "
          "failures and recoveries without replacing the complete outcome grid.", "small"),
        PageBreak(),
    ])

    # Page 4: episode-level means and call-weighted latency have different denominators.
    cost_rows, motion_rows = [], []
    for k in KS:
        selected = by_k["combined"][k]
        cost_rows.append([k, *[num(mean(selected, field)) for field in
                              ("model_calls", "inference_s", "simulation_s", "compute_s", "episode_wall_s")]])
        motion_rows.append([k, num(mean(selected, "eef_path_m"), 2),
                            "undefined" if k == 50 else num(mean(selected, "mean_plan_revision_translation_l2"), 3),
                            num(mean(selected, "mean_replan_jump_translation_l2"), 3),
                            num(mean(selected, "gripper_close_commands"))])
    story.extend([
        p("Computational cost and robot motion", "h1"),
        p("Runtime values below are means per episode. Early termination affects calls, duration and path length; "
          "calls = ceil(executed steps / K). A low cost is not a substitute for successful completion."),
        table(["K", "Calls", "Model, s", "Simulator, s", "Sum, s", "Episode, s"], cost_rows,
              [0.07, 0.15, 0.19, 0.20, 0.19, 0.20]),
        *figure("study-cost", "Figure 3. Calls and wall time across all 150 episodes, with state-bootstrap intervals; model latency is total inference time divided by total calls, without an uncertainty interval."),
        p("Motion proxies", "h2"),
        table(["K", "EEF path, m", "Plan revision", "Replan jump", "Gripper closes"], motion_rows,
              [0.07, 0.22, 0.24, 0.24, 0.23]),
        Spacer(1, 6),
        p("Plan revision: L2 difference between the new first translation command and the next command in the previous "
          "plan. At K=50 the previous plan is exhausted, so revision is undefined. Replan jump: L2 change between "
          "consecutive executed translation commands at replanning. Both use action units, not meters. Changes and "
          "gripper-closing transitions are proxies; they do not count grasp errors or prove useful correction. "
          "Each proxy mean uses only episodes where that value is defined.", "small"),
        p("Model time includes preprocessing, inference, postprocessing and GPU synchronization. Simulator time "
          "includes stepping and rendering. Episode wall time adds reset, frame recording and video finalization, "
          "but excludes model loading, warmup and JSON-log writes. Simulated duration is steps / 20, not wall time.", "small"),
        p(f"Measured totals: {sum(r['model_calls'] for r in rows['combined']):,} model calls and "
          f"{sum(r['episode_wall_s'] for r in rows['combined']) / 60:.1f} min summed episode time across all 150 episodes. "
          "Setup and transfer time are excluded.", "small"),
        PageBreak(),
    ])

    # Page 5: separate selection, actual inspection and resolvable event counts.
    coverage = stats["phases"]["combined"]["annotation_coverage"]
    inspected = None if any(item.get("inspected_episodes") is None for item in coverage) else sum(
        item["inspected_episodes"] for item in coverage)
    totals = [[len(rows["combined"]), sum(x["selected_for_visual_review"] for x in coverage),
               coverage_value(inspected), sum(x["fully_reviewed_episodes"] for x in coverage)]]
    label_rows = [[item["k"], f"{item['selected_for_visual_review']}/{item['total_episodes']}",
                   coverage_value(item.get("inspected_episodes")), item["fully_reviewed_episodes"],
                   event_count(item, "missed_grasps"), event_count(item, "dropped_objects")]
                  for item in coverage]
    selected_labels = [label for label in labels["combined"] if label["selected_for_visual_review"]]
    corrections = Counter(label["correction_after_replan"] or "unavailable" for label in selected_labels)
    correction_rows = [[len(selected_labels), *[corrections[key] for key in ("yes", "no", "unclear", "unavailable")]]]
    story.extend([
        p("Visual annotation: coverage and errors", "h1"),
        p(f"The review subset contains {len(selected_labels)} episodes: every task and K for states 0-3. "
          f"{coverage_value(inspected)} recordings were inspected; both missed-grasp and drop counts are "
          f"resolvable in {sum(x['fully_reviewed_episodes'] for x in coverage)}. States 4-9 contribute quantitative "
          "outcomes and cost but are outside this event-annotation subset. Unresolved contacts remain unknown."),
        table(["All episodes", "Selected", "Inspected", "Both counts known"], totals,
              [0.25, 0.22, 0.23, 0.30]),
        p("Review coverage and observed events by K", "h2"),
        table(["K", "Selected / all", "Inspected", "Both known", "Misses: sum [n/all]", "Drops: sum [n/all]"],
              label_rows, [0.06, 0.18, 0.15, 0.15, 0.23, 0.23]),
        Spacer(1, 7),
        p("Inspected counts come only from inspection.json. If no inspection record is supplied, the report says "
          "unknown; it does not infer inspection from a CSV row. reviewed=yes denotes both event counts being "
          "resolvable. reviewed=no can mean an inspected but ambiguous episode. Each event has its own known "
          "denominator: a drop may be clear while missed grasps remain unknown. Unknown events are never zero.", "small"),
        p("Correction labels in the selected subset", "h2"),
        table(["Selected episodes", "Yes", "No", "Unclear", "Unavailable"], correction_rows,
              [0.26, 0.15, 0.15, 0.19, 0.25]),
        p("Review method and interpretation", "h2"),
        p("The assistant reviews both camera views using sampled frame sheets, transitions of the gripper command, "
          "the final action and outcome frames; ambiguous contacts receive denser frame checks where available. "
          "Exact reviewed steps and evidence are recorded in notes and inspection metadata. There is no independent "
          "human review. Counts are observations from this subset, not estimated rates for all 150 episodes."),
        p("A missed grasp means closing without retaining the target. Failure to hook a drawer handle with an open "
          "gripper is not a missed finger grasp under this definition. A drop requires unintended loss after retention; "
          "planned release during placement is excluded. A correction label requires visible redirection plus relevant "
          "replanning steps in logs; temporal alignment does not establish a causal benefit."),
        PageBreak(),
    ])

    # Page 6: authored observations are checked against the complete source data.
    story.extend([p("Illustrative cases: behavior and errors", "h1"),
                  p("These paired examples use the same initial situation within each task. Outcomes below come "
                    "from episode summaries; the accompanying observations are grounded in reviewed videos and logs.")])
    for index, case in enumerate(discussion["cases"], 1):
        members = [by_id[identity] for identity in case["episode_ids"]]
        first = members[0]
        block = [p(f"{index}. {case['title']}", "h2"),
                 p(f"{TASKS[(first['suite'], first['task'])]}; state {first['state']}; seed {first['policy_seed']}. "
                   + "; ".join(f"K={r['k']}: {'success' if r['success'] else 'failure'}, "
                               f"{r['steps']} steps, {r['model_calls']} calls" for r in members) + ".", "small"),
                 p("Observation. " + case["observation"]),
                 p("Hypothesis. " + case["hypothesis"]),
                 p("Evidence. " + case["evidence"], "small")]
        if case.get("caveats"):
            block.append(p("Case limitations. " + " ".join(case["caveats"]), "small"))
        if case.get("protocol_refs"):
            block.append(p("Protocol references. " + "; ".join(case["protocol_refs"]), "small"))
        story.append(KeepTogether(block))
    story.extend([p("Examples illustrate mechanisms and uncertainty; they do not replace the complete outcome "
                    "table. A REPLAN marker shows plan replacement, not proof that a new image caused an improvement.", "small"),
                  PageBreak()])

    # Page 7: conclusions refer to all 150 actual episodes.
    story.extend([p("Findings and limits of inference", "h1"), p("Findings from the complete experiment", "h2")])
    for index, finding in enumerate(discussion["findings"], 1):
        story.append(p(f"{index}. {finding}"))
    story.extend([
        p("What the comparison identifies", "h2"),
        p("Changing K changes both the observation interval and how often stochastic action chunks are regenerated. "
          "The comparison identifies the effect of the execution policy K; it cannot isolate fresh visual feedback "
          "from plan replacement or stochastic regeneration. Matching sampling noise at shared simulation steps "
          "does not create common subsequent trajectories."),
        p("Remaining limitations", "h2"),
        p("The experiment covers ten states per task, but only three selected tasks, one checkpoint and one "
          "policy seed. Bootstrap intervals describe variability across these states conditional on those choices. "
          "They do not estimate variation across new tasks, models, seeds or physical robots. The multiple K "
          "comparisons are exploratory. These simulator states are not claimed to be unseen during checkpoint training."),
        p("Native LIBERO success can occur while an object is still held in the target region. Episodes stop at that "
          "signal; stable placement after explicit release was not tested separately. Early termination also changes "
          "cost and path length. Visual event counts cover a subset and retain unresolved contacts as unknown."),
        p("Physics pauses while inference runs. A 20 Hz simulated controller is not evidence of real-time operation "
          "under model latency. Sessions can also differ in system load. No conclusion about a physical robot is "
          "supported by these simulations alone."),
        p("Conclusion", "h2"), p(discussion["conclusion"]),
        p("Next controlled tests", "h2"),
        p("Multiple policy seeds and additional tasks would test robustness. A separate intervention comparing fresh "
          "and stale observations at the same regeneration schedule would help separate feedback from chunk "
          "replacement. Adaptive K and physical-robot latency require their own evaluation protocols.", "small"),
        PageBreak(),
    ])

    # Page 8: the original raw source runs stay separate and reproducible.
    hardware, runtime = source["hardware"], source["runtime"]
    packages = runtime["packages"]
    story.extend([
        p("Reproducibility and source provenance", "h1"),
        p(f"Device: {hardware['device']}; CUDA {hardware['cuda']}; Python {runtime['python']}; {runtime['system']}. "
          f"LeRobot {packages['lerobot']}; hf-libero {packages['hf-libero']}; MuJoCo {packages['mujoco']}; "
          f"PyTorch {packages['torch']}.", "small"),
        p(f"Model: {cfg.model_id}\nRevision: {cfg.model_revision}\n"
          f"Backbone: {cfg.backbone_id}\nRevision: {cfg.backbone_revision}", "code"),
        p("Rebuild results from the repository root", "h2"),
        p("python scripts/analyze_extension.py\n"
          "python scripts/build_expanded_figures.py --overwrite\n"
          "python scripts/build_state_outcomes.py --overwrite\n"
          "python scripts/build_expanded_report.py --overwrite\n"
          "python scripts/build_expanded_demo.py --overwrite", "code"),
        p("See docs/linux-checklist.md for collecting the episodes and setting up the pinned runtime.", "small"),
        p("The analysis requires complete paired episodes and matching runtime, checkpoint and frozen core code. "
          "Finish visual review and save annotations.csv plus inspection.json before rebuilding statistics. "
          "Then author discussion.json from actual videos. Plot regeneration after an edit uses --overwrite. "
          "PDF generation needs ReportLab and pypdf, followed by page rendering and visual inspection.", "small"),
        p(f"Frozen core SHA-256: {runtime['code_sha256']}\n"
          f"Dependency lock SHA-256: {runtime['dependency_lock_sha256']}\n"
          f"Expanded statistics SHA-256: {bundle['statistics_sha256']}\n"
          f"Study discussion SHA-256: {snapshots[args.discussion.resolve()]}", "code"),
        p("The statistics record every source manifest hash and every raw summary, action log, chunk log and "
          "video hash. Original episode files and their fingerprints remain unchanged. PDF provenance binds "
          "the statistics, discussion and figures to this report through SHA-256 digests.", "small"),
        p("References", "h2"),
    ])
    references = [
        ("SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics", "https://arxiv.org/abs/2506.01844"),
        ("Evaluated SmolVLA LIBERO checkpoint (pinned revision)", f"https://huggingface.co/{cfg.model_id}/tree/{cfg.model_revision}"),
        ("LeRobot 0.6.1 SmolVLA implementation", "https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py"),
        ("LeRobot 0.6.1 LIBERO adapter", "https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/envs/libero.py"),
        ("LIBERO environment documentation", "https://huggingface.co/docs/lerobot/en/libero"),
    ]
    for index, (title, url) in enumerate(references, 1):
        story.append(p(f'{index}. <link href="{clean(url)}" color="{TEAL}">{clean(title)}</link>', "small", markup=True))
    story.append(p("References describe the model and software. All experimental numbers come from the actual "
                   "source episodes; interpretation is supplied in the reviewed discussion files.", "small"))

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D4E0E8"))
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont("Expanded", 7.3)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, height - 11 * mm, "SMOLVLA / LIBERO | K = 1, 5, 10, 25, 50 | 150 EPISODES")
        canvas.drawString(18 * mm, 11 * mm, "3 tasks x 10 initial states x 5 replanning intervals")
        canvas.drawRightString(width - 18 * mm, 11 * mm, f"{document.page} / 8")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expanded-report-", dir=output.parent) as directory:
        staged = Path(directory) / output.name
        document = SimpleDocTemplate(str(staged), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                                     topMargin=21 * mm, bottomMargin=20 * mm,
                                     title="How often should a robot replan?",
                                     author="VLA replanning experiment",
                                     subject="SmolVLA / LIBERO: 150 matched episodes across five replanning intervals")
        document.build(story, onFirstPage=decorate, onLaterPages=decorate)
        reader = PdfReader(staged)
        if len(reader.pages) != 8 or any(not page.extract_text().strip() for page in reader.pages):
            raise ValueError(f"Expected eight nonempty pages, got {len(reader.pages)}; revise layout or shorten discussion before publishing")
        record = {
            "schema_version": 1, "primary_phase": "combined", "pages": 8,
            "pdf_sha256": sha256(staged),
            "statistics_sha256": bundle["statistics_sha256"],
            "discussion_sha256": snapshots[args.discussion.resolve()],
            "figures_record_sha256": snapshots[bundle["run"] / "report/expanded-figures.json"],
            "state_outcomes_record_sha256": snapshots[bundle["run"] / "report/state-outcomes.json"],
            "generator_sha256": snapshots[generator],
        }
        staged_record = Path(directory) / provenance_output.name
        staged_record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        verify_unchanged(snapshots)
        publish(staged, output, args.overwrite)
        publish(staged_record, provenance_output, args.overwrite)
    print(json.dumps({"output": str(output), "provenance": str(provenance_output),
                      "pages": 8, "primary_phase": "combined",
                      "episodes": {"pilot": 30, "validation": 120, "combined": 150},
                      "visual_review_required": True}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--pilot", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--discussion", type=Path, default=Path("outputs/expanded/discussion.json"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/vla-replanning-report.pdf"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-bold", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    build_report(parser.parse_args(argv))


if __name__ == "__main__":
    main()
