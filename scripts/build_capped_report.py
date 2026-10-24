#!/usr/bin/env python3
"""Build the complete eight-page fixed/capped-DVAC research report.

A reviewed discussion file with one or two illustrated cases adds a ninth page.
Run the PDF skill authoring marker first, then render and inspect every page.
This builder reads completed evidence, independently recomputes statistics, and
never changes old reports, episodes, manifests or collection sources.
"""

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.analyze_capped import (CAP_METHODS, DESIGNATED, METHODS, PLOT_CONTRASTS, TASK_NAMES,
                                   cap_behavior, cluster_statistics, frontier, label, sha256)
from scripts.verify_capped import load_and_verify

FIGURES = ("success_calls_frontier.png", "task_success.png", "cap_horizon_behavior.png",
           "paired_success_differences.png", "outcome_matrix.png")
SHORT = {method: ("A" if method.startswith("dvac_cap_") else "F") + method.rsplit("_", 1)[1] for method in METHODS}
PALETTE = {"navy": "#193A56", "teal": "#087F8C", "ink": "#203245", "muted": "#5A6B7B", "pale": "#EDF4F8"}


def clean(value):
    value = str(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return escape(value).replace("\n", "<br/>")


def read_snapshot(path, snapshots):
    path = Path(path).resolve()
    snapshots[path] = sha256(path)
    return json.loads(path.read_text())


def load_bundle(args):
    root, analysis = args.run.resolve(), args.analysis.resolve()
    snapshots = {Path(__file__).resolve(): sha256(__file__)}
    references = args.reference_root or [PROJECT / "outputs/compact", PROJECT / "outputs/validation"]
    cfg, records, checked = load_and_verify(root, reference_roots=references)
    if (len(records) != 270 or checked["fixed_episodes"] != 150 or checked["capped_episodes"] != 120
            or cfg.base.initial_states != tuple(range(10)) or cfg.base.policy_seeds != (11,)
            or cfg.methods != CAP_METHODS or cfg.base.intervals != (1, 5, 10, 25, 50)
            or cfg.base.chunk_size != 50 or cfg.base.num_denoising_steps != 10
            or cfg.base.control_hz != 20 or cfg.base.settle_steps != 10
            or cfg.base.warmup_calls != 2 or cfg.base.precision != "fp32"
            or cfg.dvac.tail_length != 5 or cfg.dvac.window_size != 5 or cfg.dvac.alpha != 2
            or checked["matched_task_state_seed_blocks"] != 30):
        raise ValueError("This report requires the complete frozen nine-method, 270-episode design")
    stats_path = analysis / "analysis.json"
    stats = read_snapshot(stats_path, snapshots)
    if (stats.get("complete") is not True or stats.get("episodes") != 270
            or stats.get("config_fingerprint") != cfg.fingerprint
            or stats.get("analysis_source_sha256") != sha256(PROJECT / "scripts/analyze_capped.py")):
        raise ValueError("Analysis is incomplete or stale")
    expected = cluster_statistics(records, draws=stats["bootstrap"]["draws"], seed=stats["bootstrap"]["seed"])
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"Statistics differ from the independently verified records: {key}")
    behavior, _, _, timing = cap_behavior(records)
    for key, value in (("cap_behavior", behavior), ("adaptive_timing", timing),
                       ("observed_frontier", frontier(expected["methods"]))):
        if stats.get(key) != value:
            raise ValueError(f"Adaptive statistics differ from raw evidence: {key}")
    verification = read_snapshot(analysis / "verification.json", snapshots)
    for key in ("complete", "fixed_episodes", "capped_episodes", "planned_capped_episodes",
                "matched_task_state_seed_blocks", "physical_task_state_settings", "manifest_sha256",
                "config_fingerprint", "invariance_sha256", "verifier_source_sha256",
                "verification_dependency_sha256", "collection_source_checks",
                "first_prediction_equality_all_methods", "max_variance_error", "capped_totals"):
        if verification.get(key) != checked[key] or stats.get("verification", {}).get(key) != checked[key]:
            raise ValueError(f"Saved verification differs from fresh verification: {key}")
    artifacts = read_snapshot(analysis / "artifacts.json", snapshots)
    if (artifacts.get("statistics_sha256") != snapshots[stats_path]
            or artifacts.get("analysis_source_sha256") != stats["analysis_source_sha256"]
            or artifacts.get("config_fingerprint") != cfg.fingerprint):
        raise ValueError("Figure/table provenance does not match the statistics")
    required_artifacts = (*FIGURES, "analysis.json", "verification.json", "report.md", "outcome_matrix.csv")
    for name in required_artifacts:
        path = analysis / name
        if artifacts.get("files", {}).get(name) != sha256(path):
            raise ValueError(f"Changed or unbound analysis artifact: {name}")
        snapshots[path] = sha256(path)
    manifest = read_snapshot(root / "manifest.json", snapshots)
    invariance = read_snapshot(root / "invariance.json", snapshots)
    bridge = read_snapshot(args.reference_check, snapshots)
    expected_pairs = {episode.pair_id for episode in cfg.episodes()}
    checks = bridge.get("checks", [])
    if (bridge.get("passed") is not True or bridge.get("pairs") != 30 or len(checks) != 30
            or bridge.get("robot_actions_executed") != 0
            or {row.get("pair_id") for row in checks} != expected_pairs
            or bridge.get("config_fingerprint") != cfg.fingerprint
            or bridge.get("source_manifest") != manifest["capped_source"]
            or bridge.get("runtime") != manifest["runtime"]
            or bridge.get("hardware") != manifest["hardware"]
            or bridge.get("sampler_manifest") != manifest["upstream_sampler"]
            or bridge.get("references") != manifest["reference_sources"]
            or any(row.get("exact_action_equality") is not True
                   or row.get("max_absolute_action_error") != 0 for row in checks)):
        raise ValueError("The historical compatibility check does not match the complete collection")
    for relative, digest in manifest["capped_source"].items():
        snapshots[PROJECT / relative] = digest
    for relative in ("scripts/analyze_capped.py", "scripts/verify_capped.py", "scripts/verify_dvac.py",
                     "scripts/verify_evidence.py", "docs/capped-protocol.md"):
        snapshots[PROJECT / relative] = sha256(PROJECT / relative)
    for item in checked["evidence"]:
        directory = root / "episodes" / item["episode_id"]
        snapshots[directory / "summary.json"] = item["summary_sha256"]
        for name, digest in item["files"].items():
            snapshots[directory / name] = digest
    # Retained audits bind their original videos and full traces, beyond the first-call reference gate.
    for historical in checked["historical_controls"]:
        old_root = Path(historical["root"])
        read_snapshot(old_root / "manifest.json", snapshots)
        read_snapshot(old_root / "report/evidence-verification.json", snapshots)
    for record in records:
        if record["origin"] == "reused_fixed":
            directory = Path(record["raw_source_root"]) / "episodes" / record["summary"]["episode_id"]
            for name in ("summary.json", "steps.jsonl", "chunks.jsonl", "video.mp4"):
                path = directory / name
                snapshots[path] = sha256(path)
    discussion = None
    if args.discussion:
        discussion = read_snapshot(args.discussion, snapshots)
        cases = discussion.get("cases", [])
        if not 1 <= len(cases) <= 2:
            raise ValueError("Reviewed discussion must contain one or two illustrated cases")
        by_id = {record["summary"]["episode_id"]: record["summary"] for record in records}
        for case in cases:
            for field, limit in (("title", 100), ("observation", 650), ("hypothesis", 450)):
                if not isinstance(case.get(field), str) or not case[field].strip() or len(case[field]) > limit:
                    raise ValueError(f"Invalid reviewed-case text: {field}")
            ids = case.get("episode_ids", [])
            if not ids or len(ids) != len(set(ids)) or any(identity not in by_id for identity in ids):
                raise ValueError("Reviewed cases must identify real comparison episodes")
            if len({by_id[identity]["pair_id"] for identity in ids}) != 1:
                raise ValueError("An illustrated case must compare the same task/state/seed")
            figure = (PROJECT / case["figure"]).resolve()
            if not figure.is_relative_to(PROJECT) or not figure.is_file():
                raise ValueError("Reviewed figure must be a saved project file")
            snapshots[figure] = sha256(figure)
    return {"cfg": cfg, "records": records, "stats": stats, "verification": checked,
            "saved_verification": verification, "manifest": manifest, "invariance": invariance,
            "bridge": bridge, "snapshots": snapshots, "root": root, "analysis": analysis,
            "statistics_path": stats_path, "discussion": discussion}


def font_paths(regular=None, bold=None):
    if bool(regular) != bool(bold):
        raise ValueError("Provide both --font and --font-bold")
    candidates = [(regular, bold)] if regular else [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            return Path(regular), Path(bold)
    raise FileNotFoundError("Suitable fonts not found; provide --font and --font-bold")


def build(args):
    bundle = load_bundle(args)
    cfg, records, stats = (bundle[key] for key in ("cfg", "records", "stats"))
    output, page_count = args.output.resolve(), 9 if bundle["discussion"] else 8
    provenance_output = output.with_suffix(".json")
    if output.suffix.lower() != ".pdf":
        raise ValueError("Output must be a PDF")
    protected_roots = [bundle["root"], bundle["analysis"]] + [Path(row["root"]) for row in bundle["verification"]["historical_controls"]]
    if any(output.is_relative_to(root) for root in protected_roots):
        raise ValueError("PDF output must be outside evidence and analysis directories")
    if (output.exists() or provenance_output.exists()) and not args.overwrite:
        raise FileExistsError("PDF or provenance exists; use --overwrite to rebuild")
    if output in bundle["snapshots"] or provenance_output in bundle["snapshots"]:
        raise ValueError("Output would overwrite an input")

    from PIL import Image as PILImage
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular, bold = font_paths(args.font, args.font_bold)
    pdfmetrics.registerFont(TTFont("Capped", str(regular)))
    pdfmetrics.registerFont(TTFont("CappedBold", str(bold)))
    pdfmetrics.registerFontFamily("Capped", normal="Capped", bold="CappedBold", italic="Capped")
    for path in (regular, bold):
        bundle["snapshots"][path] = sha256(path)
    page_width, page_height = A4
    width = page_width - 36 * mm
    styles = {
        "body": ParagraphStyle("body", fontName="Capped", fontSize=9, leading=12.2, textColor=colors.HexColor(PALETTE["ink"]), spaceAfter=6),
        "small": ParagraphStyle("small", fontName="Capped", fontSize=7.8, leading=10.3, textColor=colors.HexColor(PALETTE["muted"]), spaceAfter=5),
        "title": ParagraphStyle("title", fontName="CappedBold", fontSize=25, leading=29, textColor=colors.HexColor(PALETTE["navy"]), spaceAfter=10),
        "h1": ParagraphStyle("h1", fontName="CappedBold", fontSize=18, leading=22, textColor=colors.HexColor(PALETTE["navy"]), spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="CappedBold", fontSize=10.5, leading=14, textColor=colors.HexColor(PALETTE["teal"]), spaceBefore=6, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName="Capped", fontSize=7.6, leading=10, textColor=colors.HexColor(PALETTE["ink"])),
        "matrix": ParagraphStyle("matrix", fontName="Capped", fontSize=7.1, leading=9.3, textColor=colors.HexColor(PALETTE["ink"])),
        "head": ParagraphStyle("head", fontName="CappedBold", fontSize=7.6, leading=10, textColor=colors.white),
        "formula": ParagraphStyle("formula", fontName="Capped", fontSize=10, leading=16, textColor=colors.HexColor(PALETTE["ink"]),
                                  backColor=colors.HexColor(PALETTE["pale"]), borderPadding=7, spaceBefore=3, spaceAfter=9),
        "code": ParagraphStyle("code", fontName="Capped", fontSize=7.2, leading=10.2, textColor=colors.HexColor(PALETTE["ink"]),
                               wordWrap="CJK", backColor=colors.HexColor(PALETTE["pale"]), borderPadding=5, spaceAfter=8),
        "kicker": ParagraphStyle("kicker", fontName="CappedBold", fontSize=8.2, leading=11, textColor=colors.HexColor(PALETTE["teal"]), spaceAfter=7),
    }

    def p(text, style="body", markup=False):
        return Paragraph(str(text) if markup else clean(text), styles[style])

    def table(headers, rows, fractions, *, tight=False, matrix=False):
        item = Table([[p(value, "head") for value in headers]] + [[p(value, "matrix" if matrix else "cell") for value in row] for row in rows],
                     colWidths=[width * fraction for fraction in fractions], repeatRows=1, hAlign="LEFT")
        padding = 3 if matrix else 4 if tight else 6
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PALETTE["navy"])),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(PALETTE["pale"]), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), padding), ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ]))
        return item

    def figure(name, max_height=None, path=None):
        source = Path(path) if path else bundle["analysis"] / name
        with PILImage.open(source) as image:
            w, h = image.size
        out_width, out_height = width, width * h / w
        if max_height and out_height > max_height:
            out_width *= max_height / out_height
            out_height = max_height
        return Image(str(source), width=out_width, height=out_height, hAlign="CENTER")

    def interval(row, metric, scale=1, digits=1, signed=False):
        spec = f"{'+' if signed else ''}.{digits}f"
        return f"[{format(scale * row[metric + '_ci_low'], spec)}, {format(scale * row[metric + '_ci_high'], spec)}]"

    estimates = {row["method"]: row for row in stats["methods"]}
    comparisons = {(row["method"], row["reference"]): row for row in stats["contrasts"]}
    behavior = {row["method"]: row for row in stats["cap_behavior"]}
    task_order = {task.name: index for index, task in enumerate(cfg.base.tasks)}
    outcomes = sorted(stats["outcome_matrix"], key=lambda row: (task_order[row["task"]], row["state"], row["policy_seed"]))
    cap10, cap50 = estimates["dvac_cap_10"], estimates["dvac_cap_50"]
    designated = comparisons[DESIGNATED[0]]
    story = []

    # Page 1: complete nine-method result, measured success and common call cost.
    story += [
        p("SMOLVLA / LIBERO | NINE EXECUTION RULES | 270 MATCHED EPISODES", "kicker"),
        p("How often should<br/>a robot replan?", "title", markup=True),
        p("Fixed intervals and denoising-variance control with an execution cap", "h2"),
        p("One pretrained model generates 50 future actions from images, robot state and an instruction. "
          "Five fixed intervals and four adaptive rules choose when to observe and replan. "
          "The generated horizon stays fixed; the executed prefix changes."),
        p(f"DVAC cap 10 completed {cap10['successes']}/30 episodes; fixed K=10 completed "
          f"{estimates['fixed_10']['successes']}/30 and DVAC cap 50 completed {cap50['successes']}/30. "
          "All nine variants are reported. Completion and model calls are the common comparison measures."),
        table(["Method", "Completed", "Completion % [95% interval]", "Calls / episode [95% interval]", "Calls / 100 actions"], [
            [label(row["method"]), f"{row['successes']}/30", f"{100 * row['success_mean']:.1f} {interval(row, 'success', 100)}",
             f"{row['model_calls_mean']:.1f} {interval(row, 'model_calls')}", f"{row['calls_per_100_actions_mean']:.2f}"]
            for row in stats["methods"]
        ], [.22, .12, .24, .27, .15], tight=True),
        Spacer(1, 5),
        p("Intervals use a descriptive, task-stratified paired bootstrap. Every method uses the same "
          "30 task/state settings and policy seed 11. See page 4 and the final page for interpretation.", "small"),
        figure("success_calls_frontier.png", 205),
        p("Circles: fixed intervals; diamonds: adaptive caps. Error bars describe sampling variation over "
          "observed initial states. An apparent frontier of sample means is not proof of a superior method.", "small"),
        PageBreak(),
    ]

    # Page 2: rigorous definitions and the cap intervention.
    story += [
        p("What changes: H, C and K", "h1"),
        table(["Symbol", "Meaning", "Setting"], [
            ["H", "Future actions generated by one policy call", "50 for every method"],
            ["M", "Internal Euler updates within that call", "10 for every method"],
            ["C", "Maximum actions an adaptive call may execute", "5, 10, 25 or 50"],
            ["K", "Actions selected for execution before a new observation", "Fixed constant, or adaptive 1 through C"],
        ], [.10, .64, .26], tight=True),
        p("Flow matching inside one policy call", "h2"),
        p("With the observation/instruction c held fixed, Gaussian noise X(0) becomes an action block. "
          "Internal iteration i is separate from future action index j. The simulator pauses during generation."),
        p("t<sub>i</sub> = 1 - i/M,  v<sub>i</sub> = v<sub>θ</sub>(X<sup>(i)</sup>,t<sub>i</sub>,c)<br/>"
          "X<sup>(i+1)</sup> = X<sup>(i)</sup> - v<sub>i</sub>/M,  i = 0,...,9", "formula", markup=True),
        p("The denoising-variance signal", "h2"),
        p("Record clean-action estimates before the Euler update. For I={5,6,7,8,9}, L=5, future "
          "action j=0,...,49 and valid coordinate d=0,...,6, use population variance:"),
        p("Z<sub>i</sub> = X<sup>(i)</sup> - t<sub>i</sub>v<sub>i</sub>,  "
          "μ<sub>j,d</sub> = (1/L) Σ<sub>i∈I</sub> Z<sub>i,j,d</sub><br/>"
          "V<sub>j</sub> = Σ<sub>d=0..6</sub> (1/L) Σ<sub>i∈I</sub> (Z<sub>i,j,d</sub> - μ<sub>j,d</sub>)<sup>2</sup>", "formula", markup=True),
        p("V measures disagreement about the same future action within one generation. It uses normalized "
          "valid channels, excludes padding and reduces saved float32 estimates in float64. This is a "
          "proxy for internal stability, not calibrated failure risk or variance across successive replanning calls.", "small"),
        p("Threshold, original rule and execution cap", "h2"),
        p("W contains every value from all 50 positions in each of up to five preceding calls. Repeated "
          "values remain. Add the current V only after its decision; reset history every episode."),
        p("τ = mean(W) + 2 std<sub>pop</sub>(W);  j* = first j with V<sub>j</sub> &gt; τ<br/>"
          "K<sub>raw</sub> = 50 if no crossing; otherwise max(1,j*)<br/>"
          "K = min(C, K<sub>raw</sub>)", "formula", markup=True),
        p("The first call uses K=1 and no threshold. Execute the original final X(10) prefix, never the "
          "mean intermediate estimate. A crossing at j=4 permits indices 0-3; the minimum of one action "
          "can force execution when j=0 crosses. The cap applies to both crossing and no-crossing branches.", "small"),
        p("C=50 preserves the uncapped rule. C=1 always replans after one action, duplicating fixed K=1 "
          "under identical inputs and sampling, so it is not collected as a separate treatment. The cap "
          "does not shorten H or reduce M; its purpose is to limit open-loop execution.", "small"),
        PageBreak(),
    ]

    # Page 3: design, reuse provenance and exact compatibility checks.
    invariant_rows = [[TASK_NAMES[row["task"]], "State 21 / seed 101", f"{row['max_absolute_action_error']:g}"]
                      for row in sorted(bundle["invariance"]["checks"], key=lambda row: task_order[row["task"]])]
    story += [
        p("A matched comparison using retained controls", "h1"),
        p("Three tasks, ten initial states per task (0-9), and policy seed 11 form 30 matched blocks. "
          "Each block contains fixed K=1,5,10,25,50 and DVAC C=5,10,25,50: 270 episodes in total. "
          "The existing 150 fixed-control episodes are reused; only 120 capped episodes are newly collected."),
        table(["Task", "LIBERO suite", "Action limit", "Starts / method"], [
            [TASK_NAMES[task.name], task.suite, task.max_steps, "10"] for task in cfg.base.tasks
        ], [.38, .25, .17, .20]),
        p("Controlled configuration", "h2"),
        p("Checkpoint and backbone revisions are pinned. Every method uses H=50, M=10, FP32, 20 Hz "
          "simulated control, ten settling steps and the same sampling-seed rule. All new adaptive caps "
          "share L=5, alpha=2, five-call history, minimum K=1 and first-call K=1 initialization. "
          "The latter initialization and seven-channel mask are explicit SmolVLA adaptation choices."),
        p("The 30 new task/state blocks and four methods within each block are shuffled with seed "
          "20261024, then run sequentially on one RTX 2060. The cap variants have identical resource "
          "limits and diagnostic instrumentation. Hyperparameters and comparison rules were fixed before these runs."),
        p("Compatibility checks", "h2"),
        table(["Check", "Coverage", "Required result"], [
            ["Pinned sources and environment", "Both historical control roots", "Matching checkpoint, base code, software, assets and task limits"],
            ["Plain versus instrumented inference", "Three excluded task inputs", "Every final 50 x 7 action value exactly equal"],
            ["Historical reset / first prediction", "All 30 matched situations", "Identical reset hashes, noise seed and complete first chunk"],
            ["Collection-time gate", "Each of 120 capped episodes", "The same reset and first-chunk check before any robot action"],
        ], [.31, .25, .44], tight=True),
        Spacer(1, 5),
        table(["Instrumentation check", "Input", "Maximum error"], invariant_rows, [.43, .37, .20], tight=True),
        Spacer(1, 6),
        p("The separate historical check generated one chunk per situation and executed no robot actions. "
          "Exact equality supports compatibility on the checked inputs; it does not prove equivalence on "
          "every future observation. All original raw controls and their saved evidence checksums remain intact.", "small"),
        p("Timing boundary", "h2"),
        p("Historical controls used six Torch CPU threads and did not collect internal FM diagnostics; "
          "new adaptive variants use one thread and record those diagnostics. Therefore no fixed-versus-adaptive "
          "wall-time or inference-speed claim is made. Model calls are the common cost measure. Page 6 reports "
          "timings only among the four newly collected adaptive caps.", "small"),
        PageBreak(),
    ]

    # Page 4: paired effects and statistical interpretation.
    contrast_rows = []
    for key in PLOT_CONTRASTS:
        row = comparisons[key]
        contrast_rows.append([
            f"{SHORT[row['method']]} - {SHORT[row['reference']]}",
            f"{100 * row['success_difference']:+.1f} {interval(row, 'success', 100, signed=True)}",
            f"{row['model_calls_difference']:+.1f} {interval(row, 'model_calls', signed=True)}",
            f"{row['method_only_success']} / {row['reference_only_success']}",
        ])
    story += [
        p("Paired differences: does the cap help?", "h1"),
        p("Two comparisons were designated: A10 versus F10 tests whether earlier updates help under "
          "the same maximum interval; A10 versus A50 tests the effect of restricting long prefixes. "
          "A denotes adaptive cap C and F denotes fixed K. Every cap-versus-matching-fixed result is also shown."),
        table(["Comparison", "Completion difference, pp [95% interval]", "Call difference / episode [95% interval]", "Only first / only second succeeds"],
              contrast_rows, [.18, .29, .30, .23], tight=True),
        Spacer(1, 6),
        figure("paired_success_differences.png", 191),
        p("Differences are first method minus second on the same starts. Discordant counts show where "
          "the methods disagree; shared successes and shared failures remain in the complete matrix on page 7. "
          "The complete 36 pairwise comparisons are retained in paired_contrasts.csv.", "small"),
        p("How uncertainty is estimated", "h2"),
        p(f"Each of {stats['bootstrap']['draws']:,} bootstrap draws samples ten initial states with replacement "
          "within each task, keeping all nine methods together. Task means receive equal weight. Ratios "
          "are recomputed in each draw; 2.5th and 97.5th percentiles form the displayed intervals. "
          "The random seed is 20261024. The experimental unit is a physical task/initial-state setting."),
        p("These are descriptive, exploratory intervals on previously examined states. They are not "
          "multiplicity-adjusted confirmatory tests. An interval containing zero does not show equivalence; "
          "one excluding zero does not remove selection or generalization limits. When every observed "
          "state has the same outcome, a bootstrap interval can collapse without guaranteeing future success."),
        p("Calls and episode length must be read together", "h2"),
        p("For a realized episode of T actions, a cap-C schedule needs at least ceil(T/C) calls. It cannot "
          "use longer prefixes than fixed K=C. However, the methods may reach success at different times, "
          "so calls per episode and calls per 100 executed actions answer different questions.", "small"),
        PageBreak(),
    ]

    # Page 5: task heterogeneity and honest motion diagnostics.
    task_table = []
    for task in cfg.base.tasks:
        rows = {row["method"]: row for row in stats["by_task"] if row["task"] == task.name}
        task_table.append([TASK_NAMES[task.name]] + [f"{rows[method]['successes']}/10" for method in METHODS])
    story += [
        p("Task-specific behavior and motion diagnostics", "h1"),
        p("Each method sees the same ten initial states for each task. The three manipulation behaviors "
          "can respond differently to an execution cap; an aggregate mean can hide those differences."),
        figure("task_success.png", 166),
        table(["Task"] + [SHORT[method] for method in METHODS], task_table, [.28] + [.08] * 9, tight=True),
        Spacer(1, 7),
        p("F1-F50 are fixed intervals; A5-A50 are adaptive maximum intervals. Exact per-task estimates, "
          "call counts and descriptive intervals are available in by_task.csv.", "small"),
        p("Measured trajectories and gripper commands", "h2"),
        table(["Method", "Mean actions / episode", "Mean end-effector path, m", "Mean close-command events"], [
            [label(row["method"]), f"{row['steps_mean']:.1f}", f"{row['eef_path_m_mean']:.3f}", f"{row['gripper_close_commands_mean']:.2f}"]
            for row in stats["methods"]
        ], [.25, .24, .26, .25], tight=True),
        Spacer(1, 6),
        p("Path length sums the end-effector displacement after executed actions. A close-command event "
          "is a transition to a positive gripper command; it is not a successful grasp. Longer paths or "
          "repeated closing may reflect retries, task progress or an unsuccessful trajectory. Early "
          "termination also changes these totals."),
        p("Replanning changes the action chunk after a new observation. That can enable a correction, "
          "but a changed command alone does not prove that an error was corrected. Failure mechanisms "
          "need video/contact evidence. This report does not convert command counts into grasp-error rates.", "small"),
        p("Success uses the native LIBERO predicate and stops the episode immediately. It may occur "
          "before an object is released; post-release stability is not evaluated.", "small"),
        PageBreak(),
    ]

    # Page 6: cap mechanics and only genuinely comparable timing measurements.
    story += [
        p("How often did the cap actually bind?", "h1"),
        p("The raw DVAC choice is computed over all 50 predicted actions, then clipped by C. A cap binds "
          "only when this clipping changes K. Selected K and the actually executed prefix differ when "
          "success or termination interrupts a plan."),
        figure("cap_horizon_behavior.png", 185),
        table(["Cap C", "Mean raw K", "Mean chosen K", "Mean executed", "Calls limited by cap", "K=1 after first call"], [
            [row["execution_cap"], f"{row['raw_chosen_k_mean_per_call']:.2f}", f"{row['chosen_k_mean_per_call']:.2f}",
             f"{row['executed_prefix_mean_per_call']:.2f}", f"{100 * row['cap_applied_fraction']:.1f}%",
             f"{100 * row['nonbootstrap_k1_fraction']:.1f}%" if row['nonbootstrap_k1_fraction'] is not None else "n/a"]
            for row in stats["cap_behavior"]
        ], [.10, .15, .17, .17, .21, .20], tight=True),
        Spacer(1, 6),
        p("Mean K weights policy calls equally. Cap-binding fractions include first calls; the final "
          "column excludes the mandatory first-call K=1 initialization. Different caps create different "
          "trajectories and observations, so their raw-K distributions are not evaluations of one shared "
          "variance sequence.", "small"),
        p("Calls without a variance crossing after initialization: " + "; ".join(
            f"C={row['execution_cap']}: {row['nonbootstrap_no_crossing_calls']}/{row['nonbootstrap_calls']}"
            for row in stats["cap_behavior"]) + ". These raw choices are still limited by C.", "small"),
        p("Measured runtime among adaptive caps only", "h2"),
        table(["Cap C", "Inference s / call", "Selector ms / call", "Compute s / episode", "Wall s / episode"], [
            [row["method"].rsplit("_", 1)[1], f"{row['inference_s_per_call']:.3f}",
             f"{1000 * row['selector_s_per_call']:.3f}", f"{row['compute_s_mean_per_episode']:.2f}",
             f"{row['episode_wall_s_mean_per_episode']:.2f}"]
            for row in stats["adaptive_timing"]
        ], [.12, .22, .22, .22, .22], tight=True),
        Spacer(1, 6),
        p("Inference includes the instrumented policy prediction and variance calculation. Selector time "
          "is measured separately. Compute time is inference + selection + simulator stepping. Episode "
          "wall time includes reset, rollout and video closure, excluding JSON serialization, hashing and "
          "publication of the evidence directory."),
        p("The synchronous simulator pauses during inference. Playback follows simulation time and "
          "does not show compute pauses as physical motion. These measurements do not establish "
          "real-time control performance, asynchronous latency tolerance, or fixed-versus-adaptive speedup.", "small"),
        PageBreak(),
    ]

    # Page 7: complete 30-by-nine outcome/call matrix in readable compact notation.
    short_task = {cfg.base.tasks[0].name: "Soup", cfg.base.tasks[1].name: "Bowl", cfg.base.tasks[2].name: "Drawer"}
    matrix_rows = [[short_task[row["task"]], row["state"]] + [
        f"{'S' if row[method + '_success'] else 'F'}/{row[method + '_calls']}" for method in METHODS
    ] for row in outcomes]
    story += [
        p("Every matched outcome and call count", "h1"),
        p("Each cell is outcome / model calls: S = success, F = failure. All rows use policy seed 11. "
          "F1,F5,F10,F25,F50 denote fixed K; A5,A10,A25,A50 denote adaptive cap C. Every method has "
          "exactly one episode in every row."),
        table(["Task", "State"] + [SHORT[method] for method in METHODS], matrix_rows,
              [.12, .07] + [.09] * 9, matrix=True),
        Spacer(1, 7),
        p("All 270 outcomes are shown. The underlying numerical matrix is retained in outcome_matrix.csv; "
          "the color-coded overview is outcome_matrix.png. These complete records prevent favorable "
          "examples from substituting for the overall comparison.", "small"),
        p("Predetermined demonstration", "h2"),
        p("The main recording uses state 0 and seed 11 for all three tasks, showing fixed K=10 beside "
          "DVAC C=10 and C=50. Instructions, robot motion, replanning markers, chosen K, cap, call count "
          "and outcome are displayed. Every preselected task is included regardless of outcome.", "small"),
        PageBreak(),
    ]

    if bundle["discussion"]:
        story += [p("Reviewed illustrations of outcome disagreements", "h1"),
                  p("These illustrations are selected using the outcomes and are not an unbiased error sample. "
                    "Observed motion is separated from hypotheses about its cause. The complete outcome matrix "
                    "remains the basis for comparing methods.", "small")]
        by_id = {record["summary"]["episode_id"]: record["summary"] for record in records}
        for case in bundle["discussion"]["cases"]:
            row = by_id[case["episode_ids"][0]]
            story += [p(case["title"], "h2"),
                      p(f"{TASK_NAMES[row['task']]} | state {row['state']} | policy seed {row['policy_seed']}", "small"),
                      figure(None, 150, path=PROJECT / case["figure"]),
                      p("Observed: " + case["observation"]),
                      p("Hypothesis: " + case["hypothesis"], "small")]
        story.append(PageBreak())

    # Final page: explicit scope, conclusions grounded in actual contrasts and reproduction.
    nondominated = [label(row["method"]) for row in stats["observed_frontier"] if row["observed_nondominated"]]
    story += [
        p("Interpretation, limits and reproducibility", "h1"),
        p(f"For the designated A10-F10 comparison, completion changed by "
          f"{100 * designated['success_difference']:+.1f} percentage points and calls by "
          f"{designated['model_calls_difference']:+.1f} per episode. The sample-mean frontier contains "
          f"{', '.join(nondominated)}: no other method matches or improves both completion and call count "
          "while strictly improving one. This describes the observed trade-off, not a universally optimal rule."),
        p("What the experiment cannot establish", "h2"),
        p("The fixed outcomes were already examined and informed the choice to study caps. This is an "
          "exploratory comparison on seen states, not a held-out confirmation. Only one policy seed and "
          "three fixed tasks are included. The intervals describe variation across these initial situations; "
          "they do not estimate uncertainty over arbitrary new tasks or robustness to generation noise."),
        p("A stable internal flow trajectory can still lead to an inaccurate action. More frequent replanning "
          "cannot guarantee recovery from a missed grasp or systematic model error. Changing K combines "
          "new observations and stochastic regeneration; this design does not isolate their causal effects. "
          "Video illustrations can suggest mechanisms but do not provide contact-mechanical proof or error rates.", "small"),
        p("This comparison evaluates the full DVAC rule and its cap. Isolating the contribution of the "
          "variance signal would require a control schedule matched on model-call budget or execution-interval "
          "distribution. Better completion alone would not establish that V predicts errors.", "small"),
        p("Reproduce the retained-control comparison", "h2"),
        p("bash scripts/collect_capped.sh check configs/capped.json \\\n  --output outputs/capped-check/invariance.json\n"
          "bash scripts/collect_capped.sh reference-check configs/capped.json \\\n  --reference-roots outputs/compact outputs/validation \\\n  --output outputs/capped-check/historical-reference.json\n"
          "bash scripts/collect_capped.sh --config configs/capped.json --output outputs/capped \\\n  --reference-roots outputs/compact outputs/validation\n"
          "python scripts/analyze_capped.py outputs/capped outputs/capped-report --decode-videos\n"
          "python scripts/build_capped_report.py"
          + (" --discussion " + shlex.quote(str(args.discussion)) if args.discussion else ""), "code"),
        p("Use the pinned Linux/CUDA runtime for collection, with the documented CPU, memory and thread "
          "limits. Analysis/PDF tools use the artifact environment. The report rebuild verifies every "
          "episode and recomputes the paired bootstrap. The adjacent JSON binds all used statistics, figures, "
          "raw evidence and source files to hashes. Render and inspect all PDF pages before delivery.", "small"),
        p(f"Checkpoint: {cfg.base.model_id}\nRevision: {cfg.base.model_revision}\n"
          f"Statistics SHA-256: {bundle['snapshots'][bundle['statistics_path']]}", "code"),
        p("References", "h2"),
    ]
    for index, (title, url) in enumerate([
        ("Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies. Section 3.", "https://arxiv.org/html/2606.03847v1"),
        ("SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.", "https://arxiv.org/abs/2506.01844"),
        ("Pinned SmolVLA LIBERO checkpoint.", f"https://huggingface.co/{cfg.base.model_id}/tree/{cfg.base.model_revision}"),
        ("LeRobot 0.6.1 SmolVLA implementation.", "https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py"),
    ], 1):
        story.append(p(f'{index}. <link href="{clean(url)}" color="{PALETTE["teal"]}">{clean(title)}</link>', "small", markup=True))

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D4E0E8"))
        canvas.line(18 * mm, page_height - 14 * mm, page_width - 18 * mm, page_height - 14 * mm)
        canvas.setFont("Capped", 7.1)
        canvas.setFillColor(colors.HexColor(PALETTE["muted"]))
        canvas.drawString(18 * mm, page_height - 11 * mm, "SMOLVLA / LIBERO | FIXED INTERVALS AND CAPPED DVAC")
        canvas.drawString(18 * mm, 11 * mm, "270 matched episodes | 30 physical settings | 9 execution rules")
        canvas.drawRightString(page_width - 18 * mm, 11 * mm, f"{document.page} / {page_count}")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".capped-report-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        document = SimpleDocTemplate(str(staged), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                                     topMargin=21 * mm, bottomMargin=20 * mm,
                                     title="How often should a robot replan? Fixed intervals and capped DVAC",
                                     author="VLA replanning experiment", subject="270 matched SmolVLA / LIBERO episodes")
        document.build(story, onFirstPage=decorate, onLaterPages=decorate)
        reader = PdfReader(staged)
        if len(reader.pages) != page_count or any(not page.extract_text().strip() for page in reader.pages):
            headings = [page.extract_text()[120:240] for page in reader.pages]
            raise ValueError(f"Expected {page_count} nonempty pages, got {len(reader.pages)}. Page excerpts: {headings}")
        for path, expected in bundle["snapshots"].items():
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Input changed while building the PDF: {path}")
        source_hashes = {str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path): digest
                         for path, digest in sorted(bundle["snapshots"].items())}
        provenance = {
            "schema_version": 1, "artifact_kind": "capped_dvac_full_report", "pages": page_count,
            "episodes": 270, "retained_fixed_episodes": 150, "new_capped_episodes": 120,
            "config_fingerprint": cfg.fingerprint, "pdf_sha256": sha256(staged),
            "statistics_sha256": bundle["snapshots"][bundle["statistics_path"]],
            "generator_sha256": sha256(__file__), "manifest_sha256": bundle["verification"]["manifest_sha256"],
            "historical_reference_check_sha256": bundle["snapshots"][args.reference_check.resolve()],
            "sources": source_hashes, "visual_review_required": True,
        }
        staged_record = Path(temporary) / provenance_output.name
        staged_record.write_text(json.dumps(provenance, indent=2) + "\n")
        os.replace(staged, output)
        os.replace(staged_record, provenance_output)
    print(json.dumps({"output": str(output), "provenance": str(provenance_output), "pages": page_count,
                      "episodes": 270, "visual_review_required": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/capped"))
    parser.add_argument("--analysis", type=Path, default=Path("outputs/capped-report"))
    parser.add_argument("--reference-root", type=Path, action="append")
    parser.add_argument("--reference-check", type=Path, default=Path("outputs/capped-check/historical-reference.json"))
    parser.add_argument("--discussion", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf/vla-capped-replanning-report.pdf"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-bold", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
