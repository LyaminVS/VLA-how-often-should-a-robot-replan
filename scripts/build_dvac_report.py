#!/usr/bin/env python3
"""Build a six-page English DVAC report from verified, complete experimental evidence.

Run the PDF skill's artifact-operation marker before authoring. This script does
not run the marker. Render all pages and inspect them after generation.
No model is loaded and no source experiment files are modified.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.analyze_dvac import LABELS, TASKS, aggregate, paired_statistics
from scripts.verify_dvac import load_and_verify, sha256

COLORS = {"navy": "#193A56", "teal": "#087F8C", "ink": "#203245", "muted": "#5A6B7B", "pale": "#EDF4F8"}
FIGURES = ("completion_and_calls.png", "dvac_interval_distribution.png", "dvac_decision_examples.png")


def clean(value):
    text = str(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return escape(text).replace("\n", "<br/>")


def read_snapshot(path, snapshots):
    path = Path(path).resolve()
    snapshots[path] = sha256(path)
    return json.loads(path.read_text())


def fonts(regular=None, bold=None):
    if bool(regular) != bool(bold):
        raise ValueError("Provide both --font and --font-bold")
    candidates = [(regular, bold)] if regular else [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for normal, heavy in candidates:
        if Path(normal).is_file() and Path(heavy).is_file():
            return Path(normal), Path(heavy)
    raise FileNotFoundError("Suitable TTF fonts not found; provide --font and --font-bold")


def load_bundle(args):
    root, analysis = args.run.resolve(), args.analysis.resolve()
    snapshots = {Path(__file__).resolve(): sha256(__file__)}
    cfg, records, checked = load_and_verify(root)
    if (len(records) != 48 or len(cfg.base.tasks) != 3 or cfg.base.initial_states != (15, 16)
            or cfg.base.policy_seeds != (11, 22) or cfg.methods != ("fixed_5", "fixed_10", "fixed_25", "dvac")
            or cfg.base.chunk_size != 50 or cfg.base.num_denoising_steps != 10
            or cfg.base.control_hz != 20 or cfg.base.settle_steps != 10
            or cfg.base.warmup_calls != 2 or cfg.base.precision != "fp32"
            or cfg.dvac.tail_length != 5 or cfg.dvac.window_size != 5 or cfg.dvac.alpha != 2
            or checked["physical_task_state_settings"] != 6
            or checked["matched_task_state_seed_blocks"] != 12):
        raise ValueError("This report is specific to the frozen 48-episode DVAC comparison")
    statistics_path = args.statistics.resolve() if args.statistics else analysis / "analysis.json"
    stats = read_snapshot(statistics_path, snapshots)
    if (stats.get("complete") is not True or stats.get("episodes") != 48
            or stats.get("config_fingerprint") != cfg.fingerprint
            or stats.get("analysis_source_sha256") != sha256(PROJECT / "scripts/analyze_dvac.py")):
        raise ValueError("The statistics are incomplete or stale")
    expected_methods = [aggregate(records, method) for method in cfg.methods]
    expected_tasks = [aggregate(records, method, task=task.name) for task in cfg.base.tasks for method in cfg.methods]
    expected_pairs, _, expected_states = paired_statistics(records, cfg.methods)
    for key, expected in (("methods", expected_methods), ("tasks", expected_tasks),
                          ("paired_comparisons", expected_pairs), ("physical_setting_differences", expected_states)):
        if stats.get(key) != expected:
            raise ValueError(f"Statistics do not match independently verified records: {key}")
    saved_verification = read_snapshot(analysis / "verification.json", snapshots)
    core_fields = ("complete", "episodes", "planned_episodes", "matched_task_state_seed_blocks",
                   "physical_task_state_settings", "manifest_sha256", "config_fingerprint",
                   "invariance_sha256", "verifier_source_sha256", "collection_source_checks",
                   "first_prediction_equality", "max_variance_error", "totals")
    for key in core_fields:
        if saved_verification.get(key) != checked[key] or stats.get("verification", {}).get(key) != checked[key]:
            raise ValueError(f"The saved verification is stale: {key}")
    artifacts = read_snapshot(analysis / "artifacts.json", snapshots)
    if (artifacts.get("statistics_sha256") != snapshots[statistics_path]
            or artifacts.get("analysis_source_sha256") != stats["analysis_source_sha256"]
            or artifacts.get("config_fingerprint") != cfg.fingerprint):
        raise ValueError("Analysis artifact provenance does not match the statistics")
    for name in (*FIGURES, "report.md", "verification.json"):
        path = analysis / name
        if artifacts.get("files", {}).get(name) != sha256(path):
            raise ValueError(f"Analysis artifact changed: {name}")
        snapshots[path] = sha256(path)
    manifest = read_snapshot(root / "manifest.json", snapshots)
    invariance = read_snapshot(root / "invariance.json", snapshots)
    if any(row.get("initial_state") != 21 or row.get("sampling_seed") != 101 for row in invariance["checks"]):
        raise ValueError("Unexpected inputs in the excluded-state invariance check")
    for item in checked["evidence"]:
        directory = root / "episodes" / item["episode_id"]
        snapshots[directory / "summary.json"] = item["summary_sha256"]
        for name, digest in item["files"].items():
            snapshots[directory / name] = digest
    for relative in ("scripts/analyze_dvac.py", "scripts/verify_dvac.py", "docs/dvac-protocol.md"):
        path = PROJECT / relative
        snapshots[path] = sha256(path)
    for relative, digest in manifest["adaptive_source"].items():
        snapshots[PROJECT / relative] = digest
    diagnostic_review = None
    review_path = analysis / "review/posthoc-review.json"
    if review_path.exists():
        diagnostic_review = read_snapshot(review_path, snapshots)
        if diagnostic_review.get("status") != "reviewed" or diagnostic_review.get("independent_human_review") is not False:
            raise ValueError("Expected explicitly assistant-generated diagnostic review")
        known = {record["summary"]["episode_id"]: record["summary"] for record in records}
        for entry in diagnostic_review["episodes"]:
            summary = known[entry["episode_id"]]
            if entry["video_sha256"] != summary["evidence_sha256"]["video.mp4"]:
                raise ValueError("Diagnostic review references changed source video")
    return {
        "root": root, "analysis": analysis, "statistics_path": statistics_path,
        "cfg": cfg, "records": records, "stats": stats, "verification": checked,
        "manifest": manifest, "invariance": invariance, "snapshots": snapshots,
        "diagnostic_review": diagnostic_review,
    }


def build(args):
    bundle = load_bundle(args)
    cfg, records, stats, checked = (bundle[key] for key in ("cfg", "records", "stats", "verification"))
    output = args.output.resolve()
    sidecar = output.with_suffix(".json")
    if output.suffix.lower() != ".pdf":
        raise ValueError("Report output must use a .pdf extension")
    if output.is_relative_to(bundle["root"]) or output.is_relative_to(bundle["analysis"]):
        raise ValueError("The PDF must be outside the immutable evidence and analysis directories")
    if (output.exists() or sidecar.exists()) and not args.overwrite:
        raise FileExistsError("Report or provenance exists; pass --overwrite to rebuild")
    if output in bundle["snapshots"] or sidecar in bundle["snapshots"]:
        raise ValueError("Report output cannot overwrite an input")

    from PIL import Image as PILImage
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular, bold = fonts(args.font, args.font_bold)
    pdfmetrics.registerFont(TTFont("DVAC", str(regular)))
    pdfmetrics.registerFont(TTFont("DVACBold", str(bold)))
    pdfmetrics.registerFontFamily("DVAC", normal="DVAC", bold="DVACBold", italic="DVAC")
    bundle["snapshots"][regular] = sha256(regular)
    bundle["snapshots"][bold] = sha256(bold)
    page_width, page_height = A4
    width = page_width - 36 * mm
    ink = colors.HexColor(COLORS["ink"])
    styles = {
        "body": ParagraphStyle("body", fontName="DVAC", fontSize=9, leading=12.2, textColor=ink, spaceAfter=6),
        "small": ParagraphStyle("small", fontName="DVAC", fontSize=7.8, leading=10.3,
                                textColor=colors.HexColor(COLORS["muted"]), spaceAfter=5),
        "title": ParagraphStyle("title", fontName="DVACBold", fontSize=25, leading=29,
                                textColor=colors.HexColor(COLORS["navy"]), spaceAfter=10),
        "h1": ParagraphStyle("h1", fontName="DVACBold", fontSize=18, leading=22,
                             textColor=colors.HexColor(COLORS["navy"]), spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="DVACBold", fontSize=10.5, leading=14,
                             textColor=colors.HexColor(COLORS["teal"]), spaceBefore=6, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="DVAC", fontSize=7.8, leading=10.2, textColor=ink),
        "head": ParagraphStyle("head", fontName="DVACBold", fontSize=7.8, leading=10.2, textColor=colors.white),
        "formula": ParagraphStyle("formula", fontName="DVAC", fontSize=10.1, leading=16,
                                  textColor=ink, backColor=colors.HexColor(COLORS["pale"]),
                                  borderPadding=7, spaceBefore=3, spaceAfter=9),
        "code": ParagraphStyle("code", fontName="DVAC", fontSize=7.2, leading=10.2,
                               textColor=ink, wordWrap="CJK", backColor=colors.HexColor(COLORS["pale"]),
                               borderPadding=5, spaceAfter=8),
        "kicker": ParagraphStyle("kicker", fontName="DVACBold", fontSize=8.3, leading=11,
                                 textColor=colors.HexColor(COLORS["teal"]), spaceAfter=7),
    }

    def p(text, style="body", markup=False):
        return Paragraph(str(text) if markup else clean(text), styles[style])

    def table(headers, rows, fractions, *, tight=False):
        body = [[p(value, "head") for value in headers]] + [[p(value, "cell") for value in row] for row in rows]
        item = Table(body, colWidths=[width * fraction for fraction in fractions], repeatRows=1, hAlign="LEFT")
        pad = 4 if tight else 6
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLORS["navy"])),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(COLORS["pale"]), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ]))
        return item

    def figure(name, max_height=None):
        path = bundle["analysis"] / name
        with PILImage.open(path) as image:
            w, h = image.size
        target_width = width
        target_height = width * h / w
        if max_height and target_height > max_height:
            target_width *= max_height / target_height
            target_height = max_height
        return Image(str(path), width=target_width, height=target_height, hAlign="CENTER")

    methods = {row["method"]: row for row in stats["methods"]}
    adaptive = methods["dvac"]
    fixed = [methods[method] for method in cfg.methods if method != "dvac"]
    paired = {row["reference_method"]: row for row in stats["paired_comparisons"]}
    summaries = [record["summary"] for record in records]
    by_pair = defaultdict(dict)
    for summary in summaries:
        by_pair[(summary["task"], summary["state"], summary["policy_seed"])][summary["method"]] = summary
    task_order = {task.name: index for index, task in enumerate(cfg.base.tasks)}
    pair_keys = sorted(by_pair, key=lambda key: (task_order[key[0]], key[1], key[2]))
    reference = paired["fixed_10"]
    story = []

    # 1: research question and exact measured outcomes.
    story.extend([
        p("SMOLVLA / LIBERO  |  ADAPTIVE REPLANNING  |  48 MATCHED EPISODES", "kicker"),
        p("Can the model decide<br/>when to replan?", "title", markup=True),
        p("A denoising-variance decision rule compared with fixed execution intervals", "h2"),
        p("A robot predicts a block of future actions, executes a prefix, observes the scene again and replans. "
          "This experiment tests whether the model's internal flow-matching trajectory can choose that prefix "
          "without additional model evaluations. The target is task completion with fewer policy calls."),
        p(f"DVAC completed {adaptive['successes']}/12 episodes. The fixed controls completed between "
          f"{min(row['successes'] for row in fixed)} and {max(row['successes'] for row in fixed)}/12. "
          f"DVAC used {adaptive['total_model_calls']} policy calls in total, averaging "
          f"{adaptive['mean_model_calls_per_episode']:.1f} calls per episode. All outcomes are retained."),
        table(["Method", "Completed", "Calls total", "Calls / action", "Compute s / episode", "Wall s / episode"], [
            [LABELS[row["method"]], f"{row['successes']}/{row['episodes']}", row["total_model_calls"],
             f"{row['calls_per_action_pooled']:.3f}", f"{row['mean_compute_s']:.2f}", f"{row['mean_episode_wall_s']:.2f}"]
            for row in stats["methods"]
        ], [.19, .14, .14, .16, .19, .18]),
        Spacer(1, 6),
        p("Calls/action is pooled over executed actions. Successful episodes may stop earlier, so call totals "
          "and wall time also depend on episode duration. Compute time includes instrumented inference, "
          "DVAC selection and simulator stepping.", "small"),
        figure("completion_and_calls.png", 192),
        p("Each point uses 12 matched task/state/seed episodes. There are six physical task/state settings, "
          "each repeated with two sampling seeds. These observations do not establish a general benchmark ranking.", "small"),
        PageBreak(),
    ])

    # 2: mathematical method with explicit iteration/action indices and adapters.
    story.extend([
        p("From flow matching to an execution horizon", "h1"),
        p("H=50 future actions are generated in M=10 internal Euler updates. The robot executes K actions "
          "before receiving its next observation. Internal iteration i and future action index j are different axes."),
        p("1. Generate the complete action block", "h2"),
        p("The input c contains both camera images, robot state and the instruction. Starting from Gaussian "
          "noise X(0), the model updates the entire action block. The input c stays fixed; the simulator is paused."),
        p("t<sub>i</sub> = 1 - i/M,  v<sub>i</sub> = v<sub>θ</sub>(X<sup>(i)</sup>, t<sub>i</sub>, c)<br/>"
          "X<sup>(i+1)</sup> = X<sup>(i)</sup> - v<sub>i</sub>/M,  i = 0,...,9", "formula", markup=True),
        p("2. Measure consistency inside that one generation", "h2"),
        p("Capture a clean-action estimate before each Euler update. Keep the final L=5 estimates "
          "with iteration set I={5,6,7,8,9} (flow times approximately 0.5, 0.4, 0.3, 0.2, 0.1). For each future action j, compute "
          "population variance across estimates and sum over seven valid action coordinates:"),
        p("Z<sub>i</sub> = X<sup>(i)</sup> - t<sub>i</sub>v<sub>i</sub><br/>"
          "μ<sub>j,d</sub> = (1/L) Σ<sub>i∈I</sub> Z<sub>i,j,d</sub><br/>"
          "V<sub>j</sub> = Σ<sub>d=0..6</sub> (1/L) Σ<sub>i∈I</sub> "
          "(Z<sub>i,j,d</sub> - μ<sub>j,d</sub>)<sup>2</sup>", "formula", markup=True),
        p("The estimates concern the same future action inside one call. Normalized action channels "
          "are used and padding is excluded. Float32 estimates are reduced in float64."),
        p("3. Choose a prefix using the preceding calls", "h2"),
        p("W contains all 50 variance values from each of up to five preceding calls, including repeated "
          "values. Add the current call only after selecting K. Use population standard deviation."),
        p("τ = mean(W) + 2 std<sub>pop</sub>(W)<br/>"
          "j* = first j with V<sub>j</sub> &gt; τ,  j = 0,...,49<br/>"
          "K = 50 if no crossing exists; otherwise K = max(1, j*)", "formula", markup=True),
        p("Execute the first K actions from the final X(10), with the original postprocessing. For example, "
          "j*=4 executes indices 0-3. The minimum K=1 can force execution even when index 0 crosses. "
          "The mean estimate is never used as a control command.", "small"),
        p("Paper defaults used: L=5, history=5, alpha=2, minimum=1, maximum=H. Explicit adaptation choices: "
          "reset history each episode; use K=1 and no threshold for its first call; analyze seven valid "
          "SmolVLA channels. The paper does not prescribe the empty-history rule. [1]", "small"),
        PageBreak(),
    ])

    # 3: paired design, every result and sampler integrity.
    outcome_rows = []
    for task, state, seed in pair_keys:
        group = by_pair[(task, state, seed)]
        outcome_rows.append([TASKS[task], state, seed] + [
            f"{'S' if group[method]['success'] else 'F'} / {group[method]['model_calls']}" for method in cfg.methods
        ])
    invariant_rows = [
        [TASKS[row["task"]], "Exactly equal", f"{row['max_absolute_action_error']:g}"]
        for row in sorted(bundle["invariance"]["checks"], key=lambda row: task_order[row["task"]])
    ]
    story.extend([
        p("Matched design and complete outcomes", "h1"),
        p("Three tasks x two initial states (15, 16) x two policy seeds (11, 22) x four methods = 48 episodes. "
          "The six physical settings produce 12 task/state/seed blocks. Every block contains all four methods; "
          "block order and method order are randomized with seed 20261024."),
        p("All methods use the same checkpoint, 50-action horizon, ten solver iterations, FP32 precision, "
          "20 Hz control, ten settling steps and two warmup calls. Soup and bowl episodes allow 280 actions; "
          "drawer episodes allow 300. The RTX 2060 runs methods sequentially. Parameters were fixed before collection."),
        table(["Task", "State", "Seed", "Fixed 5", "Fixed 10", "Fixed 25", "DVAC"],
              outcome_rows, [.31, .08, .08, .13, .13, .13, .14], tight=True),
        Spacer(1, 5),
        p("Every cell is outcome / model calls; S = success, F = failure. These are all 12 paired blocks, "
          "not selected examples. Initial simulator-state and observation hashes match across all methods "
          "and both seeds within each physical setting.", "small"),
        p("Instrumentation preserves the sampled actions", "h2"),
        p("Before collection, plain and instrumented inference were run on excluded initial state 21, "
          "with sampling seed 101, for all three tasks. Every final 50 x 7 action value matched exactly. "
          "The wrapper observes existing velocity evaluations and returns each original tensor unchanged."),
        table(["Held-out task", "Final action arrays", "Maximum absolute error"],
              invariant_rows, [.43, .32, .25], tight=True),
        Spacer(1, 5),
        p(f"The independent verifier replayed {checked['totals']['model_calls']} generated chunks and "
          f"{checked['totals']['steps']} executed actions: variance, thresholds, chosen horizons, clipped "
          "prefixes, seeds and raw-file hashes. First complete predictions also match across all methods "
          "within each state/seed block. Exact equality on these inputs is an integrity check, not a proof "
          "of universal numerical equivalence.", "small"),
        PageBreak(),
    ])

    # 4: distribution, paired differences and task stratification.
    task_rows = []
    for task in cfg.base.tasks:
        group = {row["method"]: row for row in stats["tasks"] if row["task"] == task.name}
        task_rows.append([TASKS[task.name]] + [f"{group[method]['successes']}/4" for method in cfg.methods])
    story.extend([
        p("What execution intervals did DVAC choose?", "h1"),
        p(f"Across all calls, mean K was {adaptive['chosen_k_mean_per_call']:.2f}, median "
          f"{adaptive['chosen_k_median_per_call']:g}, and range {adaptive['chosen_k_min']}-{adaptive['chosen_k_max']}. "
          f"The actual executed prefix averaged {adaptive['executed_prefix_mean_per_call']:.2f} actions "
          "because success or termination may interrupt a selected prefix."),
        figure("dvac_interval_distribution.png", 187),
        p(f"The histogram excludes the {adaptive['bootstrap_calls']} required first-call K=1 choices. "
          f"Among {adaptive['nonbootstrap_calls']} later decisions, {adaptive['nonbootstrap_k1_calls']} "
          f"chose K=1 and {adaptive['nonbootstrap_k50_calls']} chose K=50. "
          f"There were {adaptive['dvac_no_crossing_nonbootstrap_calls']} calls without any strict crossing.", "small"),
        p("Matched outcome differences", "h2"),
        table(["Reference", "DVAC only succeeds", "Fixed only succeeds", "Both succeed", "Both fail", "Mean call difference"], [
            [LABELS[row["reference_method"]], row["dvac_only_success"], row["fixed_only_success"],
             row["both_success"], row["both_failure"], f"{row['model_calls_mean_paired_difference']:+.1f}"]
            for row in stats["paired_comparisons"]
        ], [.20, .16, .16, .15, .14, .19], tight=True),
        Spacer(1, 5),
        p("Call differences are DVAC minus the reference on the same 12 blocks. Seeds share physical "
          "initial situations; no p-value or independence-based significance claim is made.", "small"),
        p("Completion by task", "h2"),
        table(["Task", "Fixed 5", "Fixed 10", "Fixed 25", "DVAC"], task_rows,
              [.40, .15, .15, .15, .15], tight=True),
        Spacer(1, 6),
        p(f"Relative to fixed K=10, DVAC's completion difference was "
          f"{100 * reference['success_rate_difference']:+.1f} percentage points and its mean call difference "
          f"was {reference['model_calls_mean_paired_difference']:+.1f}. The measured selector cost was "
          f"{1000 * adaptive['selector_s_per_call']:.3f} ms per DVAC call; instrumented inference averaged "
          f"{adaptive['inference_s_per_call']:.3f} s per call. These are observed sample means."),
        PageBreak(),
    ])

    # 5: predetermined examples, no outcome-based selection.
    selected = sorted([
        record for record in records if record["summary"]["method"] == "dvac"
        and record["summary"]["state"] == 15 and record["summary"]["policy_seed"] == 11
    ], key=lambda record: task_order[record["summary"]["task"]])
    example_rows = []
    for record in selected:
        summary, chunks = record["summary"], record["chunks"]
        example = chunks[1] if len(chunks) > 1 else None
        example_rows.append([
            TASKS[summary["task"]], "Success" if summary["success"] else "Failure",
            str(example["decision"]["chosen_k"]) if example else "n/a",
            f"{example['decision']['threshold']:.3g}" if example else "n/a",
            f"{max(example['variance']):.3g}" if example else "n/a",
        ])
    story.extend([
        p("Three examples selected before the outcomes", "h1"),
        p("Each task is shown at initial state 15 and policy seed 11. The left panels trace selected "
          "execution horizons against robot action index. The right panels always inspect the second "
          "policy call, immediately after the first-call initialization."),
        figure("dvac_decision_examples.png", 365),
        p("A horizontal threshold is computed from earlier calls. A vertical marker identifies the first "
          "strict crossing, when one exists. The final action trajectory is generated before this prefix "
          "decision. The plots reveal the decision rule; they do not identify the physical cause of a failure.", "small"),
        table(["Task", "Outcome", "Second-call K", "Threshold", "Maximum V"], example_rows,
              [.36, .16, .16, .16, .16], tight=True),
        Spacer(1, 7),
        p("The companion recording, outputs/dvac-report/demo.mp4, shows these same three task/state/seed "
          "pairs: fixed K=10 on the left and DVAC on the right, aligned by simulation time. It includes "
          "instructions, robot motion, replanning markers, selected K and final outcomes. All three "
          "preselected pairs are included regardless of success."),
    ])
    story.append(PageBreak())

    # 6: careful interpretation, runtime definitions and reproducibility.
    fixed10 = methods["fixed_10"]
    call_relation = "fewer" if adaptive["total_model_calls"] < fixed10["total_model_calls"] else "more" if adaptive["total_model_calls"] > fixed10["total_model_calls"] else "the same number of"
    story.extend([
        p("Interpretation, limits and reproducibility", "h1"),
        p(f"In this sample, DVAC completed {adaptive['successes']}/12 episodes, compared with "
          f"{fixed10['successes']}/12 for fixed K=10, and used "
          f"{call_relation} policy calls. Its internal signal therefore needs to be assessed jointly with "
          "the observed outcomes and computational cost. A stable denoising trajectory may still produce "
          "an incorrect action; large variance is not a calibrated probability of failure."),
    ])
    if bundle["diagnostic_review"]:
        story.append(p("An outcome-selected failure example", "h2"))
        story.append(p(bundle["diagnostic_review"]["summary"], "small"))
        story.append(p("Assistant inspection of sampled frames; no independent human review. Source frames and "
                       "uncertain observations are retained in outputs/dvac-report/review/.", "small"))
    story.extend([
        p("Scope of the evidence", "h2"),
        p("Only two initial states per task and two sampling seeds per state were tested. Six physical "
          "situations are too few for broad generalization. The Wilson intervals in the analysis files "
          "are descriptive and do not adjust for seed dependence. This study transfers the DVAC rule "
          "to SmolVLA; it does not reproduce the paper's full benchmark or establish an official LIBERO score."),
        p("The comparison combines fresh observations and stochastic regeneration. It does not isolate "
          "which mechanism caused a success difference. The variance scale depends on the checkpoint, "
          "normalization, solver grid and estimate window. Gripper close-command counts and path length "
          "are control diagnostics, not verified grasp-error labels. Neither the plots nor aggregate "
          "outcomes justify a causal claim about grasping or motion correction."),
        p("Timing definitions", "h2"),
        p("All four methods collect identical diagnostics. Inference includes the policy prediction and "
          "variance reduction; selector time is separate. Compute is inference + selection + simulation. "
          "Episode wall time includes reset, rollout and video closure, excluding JSON writing and hashing. "
          "The simulator pauses during inference. This is not a real-time latency or asynchronous control study. "
          "The ordered equality checks are unsuitable for estimating instrumentation overhead.", "small"),
        p("Reproduce from the repository", "h2"),
        p("bash scripts/collect_dvac.sh check configs/dvac.json \\\n  --output outputs/dvac-check/invariance.json\n"
          "bash scripts/collect_dvac.sh --config configs/dvac.json --output outputs/dvac\n"
          "python scripts/analyze_dvac.py outputs/dvac outputs/dvac-report --decode-videos\n"
          "python scripts/build_dvac_demo.py\n"
          "python scripts/build_dvac_report.py", "code"),
        p("Use the pinned Linux collection environment and the artifact environment for analysis/PDF tooling. "
          "Render the generated PDF and inspect every page before delivery. The manifest preserves "
          "checkpoint/backbone revisions, runtime, hardware and sampler hashes. The PDF's adjacent JSON "
          "binds its statistics, figures, source files and raw evidence to SHA-256 checksums.", "small"),
        p(f"Checkpoint: {cfg.base.model_id}\nRevision: {cfg.base.model_revision}\n"
          f"Statistics SHA-256: {bundle['snapshots'][bundle['statistics_path']]}", "code"),
        p("References", "h2"),
    ])
    references = [
        ("Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies. Section 3 and Algorithm 1.", "https://arxiv.org/html/2606.03847v1"),
        ("SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.", "https://arxiv.org/abs/2506.01844"),
        ("Frozen SmolVLA LIBERO checkpoint.", f"https://huggingface.co/{cfg.base.model_id}/tree/{cfg.base.model_revision}"),
        ("LeRobot 0.6.1 SmolVLA implementation.", "https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py"),
    ]
    for index, (title, url) in enumerate(references, 1):
        story.append(p(f'{index}. <link href="{clean(url)}" color="{COLORS["teal"]}">{clean(title)}</link>', "small", markup=True))

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D4E0E8"))
        canvas.line(18 * mm, page_height - 14 * mm, page_width - 18 * mm, page_height - 14 * mm)
        canvas.setFont("DVAC", 7.1)
        canvas.setFillColor(colors.HexColor(COLORS["muted"]))
        canvas.drawString(18 * mm, page_height - 11 * mm, "SMOLVLA / LIBERO | DVAC AND FIXED EXECUTION HORIZONS")
        canvas.drawString(18 * mm, 11 * mm, "48 episodes | 6 physical settings | 12 matched blocks")
        canvas.drawRightString(page_width - 18 * mm, 11 * mm, f"{document.page} / 6")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dvac-report-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        document = SimpleDocTemplate(
            str(staged), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=21 * mm, bottomMargin=20 * mm,
            title="Can the model decide when to replan?", author="VLA replanning experiment",
            subject="SmolVLA DVAC transfer: 48 matched LIBERO episodes",
        )
        document.build(story, onFirstPage=decorate, onLaterPages=decorate)
        reader = PdfReader(staged)
        if len(reader.pages) != 6 or any(not page.extract_text().strip() for page in reader.pages):
            raise ValueError(f"Expected six nonempty pages, got {len(reader.pages)}; revise layout before publishing")
        for path, expected in bundle["snapshots"].items():
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Input changed while rendering: {path}")
        input_hashes = {
            str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path): digest
            for path, digest in sorted(bundle["snapshots"].items())
        }
        provenance = {
            "schema_version": 1, "artifact_kind": "dvac_report", "pages": 6, "episodes": 48,
            "config_fingerprint": cfg.fingerprint,
            "pdf_sha256": sha256(staged),
            "statistics_sha256": bundle["snapshots"][bundle["statistics_path"]],
            "generator_sha256": sha256(__file__), "manifest_sha256": checked["manifest_sha256"],
            "invariance_sha256": checked["invariance_sha256"],
            "sources": input_hashes,
            "visual_review_required": True,
        }
        staged_sidecar = Path(temporary) / sidecar.name
        staged_sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
        os.replace(staged, output)
        os.replace(staged_sidecar, sidecar)
    print(json.dumps({"output": str(output), "provenance": str(sidecar), "pages": 6,
                      "episodes": 48, "visual_review_required": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/dvac"))
    parser.add_argument("--analysis", type=Path, default=Path("outputs/dvac-report"))
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf/vla-dvac-report.pdf"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-bold", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
