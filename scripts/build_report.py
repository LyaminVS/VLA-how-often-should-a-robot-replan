"""Build a six-page English report from a completed, real compact experiment.

Requires reportlab and pypdf. Does not run the model or modify experimental data.

Example:
    python scripts/build_report.py --run outputs/compact \
        --discussion outputs/compact/discussion.json

Discussion JSON is deliberately separate from quantitative results. It must contain:
    findings: 1-4 strings, each at most 650 characters;
    cases: 1-3 objects containing title, episode_ids, observation, hypothesis, evidence;
    conclusion: a string of at most 1100 characters.
Case observation/hypothesis/evidence should identify visible steps or video times.
Hypotheses are always explicitly labelled as hypotheses in the report.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_replan.analysis import load_results, summarize_annotations  # noqa: E402

TASK_NAMES = {
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": "Soup can into basket",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": (
        "Black bowl onto plate"
    ),
    "open_the_middle_drawer_of_the_cabinet": "Open the middle drawer",
}
BLUE = "#193A56"
TEAL = "#087F8C"
PALE = "#EDF4F8"
INK = "#203245"
MUTED = "#5A6B7B"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    """Escape ReportLab markup and normalize dash glyphs for portable PDF rendering."""
    value = str(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return escape(value).replace("\n", "<br/>")


def num(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def checked_text(value, field, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field} must be a nonempty string of at most {limit} characters")
    return value.strip()


def validate_discussion(data, rows):
    if not isinstance(data, dict):
        raise ValueError("Discussion must be a JSON object")
    findings = data.get("findings")
    cases = data.get("cases")
    if not isinstance(findings, list) or not 1 <= len(findings) <= 4:
        raise ValueError("discussion.findings must contain 1-4 strings")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 3:
        raise ValueError("discussion.cases must contain 1-3 cases")
    for index, value in enumerate(findings):
        checked_text(value, f"findings[{index}]", 650)
    checked_text(data.get("conclusion"), "conclusion", 1100)
    identities = {row["episode_id"]: row for row in rows}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{index}] must be an object")
        for field, limit in (("title", 100), ("observation", 650), ("hypothesis", 450), ("evidence", 300)):
            checked_text(case.get(field), f"cases[{index}].{field}", limit)
        ids = case.get("episode_ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 5 or len(set(ids)) != len(ids):
            raise ValueError(f"cases[{index}].episode_ids must contain 1-5 distinct episode IDs")
        if any(identity not in identities for identity in ids):
            raise ValueError(f"Unknown or out-of-cohort episode ID in cases[{index}]")
        if len({identities[identity]["pair_id"] for identity in ids}) != 1:
            raise ValueError("A case must compare the same task, initial state and policy seed")


def validate_statistics(data, rows, coverage, intervals):
    if data.get("coverage") != coverage:
        raise ValueError("Statistics coverage is stale; rerun vla_replan analyze")
    estimates = data.get("estimates", [])
    for metric in ("success", "compute_s", "model_calls", "eef_path_m", "episode_wall_s"):
        for k in intervals:
            matching = [r for r in estimates if r["metric"] == metric and r["k"] == k]
            actual = mean([r for r in rows if r["k"] == k], metric)
            if len(matching) != 1 or not math.isclose(matching[0]["mean"], actual, abs_tol=1e-9):
                raise ValueError(f"Statistics do not match completed episodes: {metric}, K={k}")


def font_paths(regular=None, bold=None):
    if bool(regular) != bool(bold):
        raise ValueError("Provide both --font and --font-bold")
    candidates = (
        [(regular, bold)]
        if regular
        else [
            (
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            ),
            (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ),
            (
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ),
        ]
    )
    for normal, heavy in candidates:
        if Path(normal).is_file() and Path(heavy).is_file():
            return str(normal), str(heavy)
    raise FileNotFoundError("No suitable TTF font found; pass --font and --font-bold")


def build_report(args):
    # Load and validate everything before creating an output file.
    root = args.run.resolve()
    cfg, rows, coverage = load_results(root, allow_incomplete=False)
    if (
        cfg.name != "compact"
        or len(cfg.tasks) != 3
        or len(cfg.initial_states) != 2
        or len(cfg.policy_seeds) != 1
        or tuple(sorted(cfg.intervals)) != (1, 5, 10, 25, 50)
        or cfg.chunk_size != 50
    ):
        raise ValueError("This report requires the 30-episode compact protocol; smoke is not research data")
    manifest = read_json(root / "manifest.json")
    stats = read_json(root / "report/statistics.json")
    discussion = read_json(args.discussion)
    intervals = sorted(cfg.intervals)
    validate_statistics(stats, rows, coverage, intervals)
    validate_discussion(discussion, rows)
    chart = root / "report/final-comparison.png"
    if not chart.is_file():
        raise FileNotFoundError(f"Run scripts/build_figures.py first: missing {chart}")
    annotation_file = root / "annotations.csv"
    if annotation_file.is_file():
        annotations = summarize_annotations(annotation_file, rows)
        with annotation_file.open(newline="", encoding="utf-8") as stream:
            all_labels = list(csv.DictReader(stream))
            reviewed = [r for r in all_labels if r["reviewed"] == "yes"]
    else:
        annotations = [
            {
                "k": k,
                "reviewed": 0,
                "total_episodes": sum(r["k"] == k for r in rows),
                "missed_grasps_per_reviewed_episode": None,
                "drops_per_reviewed_episode": None,
            }
            for k in intervals
        ]
        reviewed = []
        all_labels = []

    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
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
    pdfmetrics.registerFont(TTFont("Report", regular))
    pdfmetrics.registerFont(TTFont("ReportBold", bold))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold", italic="Report")
    width, height = A4
    body_width = width - 36 * mm
    styles = {
        "body": ParagraphStyle(
            "body",
            fontName="Report",
            fontSize=9.4,
            leading=13.5,
            textColor=colors.HexColor(INK),
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Report",
            fontSize=8,
            leading=10.8,
            textColor=colors.HexColor(MUTED),
            spaceAfter=5,
        ),
        "title": ParagraphStyle(
            "title",
            fontName="ReportBold",
            fontSize=25,
            leading=29,
            textColor=colors.HexColor(BLUE),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="ReportBold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(BLUE),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="ReportBold",
            fontSize=11.4,
            leading=15,
            textColor=colors.HexColor(TEAL),
            spaceBefore=9,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "cell",
            fontName="Report",
            fontSize=8,
            leading=10.6,
            textColor=colors.HexColor(INK),
            wordWrap="CJK",
        ),
        "head": ParagraphStyle(
            "head", fontName="ReportBold", fontSize=8, leading=10.6, textColor=colors.white, alignment=TA_LEFT
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Report",
            fontSize=7.4,
            leading=10.1,
            textColor=colors.HexColor(INK),
            wordWrap="CJK",
            spaceAfter=5,
            backColor=colors.HexColor(PALE),
            borderPadding=6,
        ),
        "kicker": ParagraphStyle(
            "kicker",
            fontName="ReportBold",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor(TEAL),
            spaceAfter=9,
        ),
    }

    def p(value, style="body", markup=False):
        return Paragraph(str(value) if markup else clean(value), styles[style])

    def table(headers, data, widths=None):
        content = [[p(value, "head") for value in headers]] + [
            [p(value, "cell") for value in row] for row in data
        ]
        result = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(PALE), colors.white]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#D4E0E8")),
                ]
            )
        )
        return result

    class ReplanDiagram(Flowable):
        def __init__(self):
            super().__init__()
            self.width = body_width
            self.height = 78

        def draw(self):
            canvas = self.canv
            labels = ["Observation", "SmolVLA: H = 50", "Execute K", "New observation"]
            subtitles = ["cameras + robot state", "action sequence", "K = 1, 5, 10, 25, 50", "discard the old tail"]
            box = (self.width - 3 * 16) / 4
            for index, (label, subtitle) in enumerate(zip(labels, subtitles, strict=True)):
                x = index * (box + 16)
                canvas.setFillColor(colors.HexColor(PALE))
                canvas.roundRect(x, 22, box, 47, 6, fill=1, stroke=0)
                for text, size, y, font in ((label, 8.3, 52, "ReportBold"), (subtitle, 7, 36, "Report")):
                    canvas.setFillColor(colors.HexColor(BLUE))
                    canvas.setFont(font, size)
                    canvas.drawCentredString(x + box / 2, y, text)
                if index < 3:
                    canvas.setStrokeColor(colors.HexColor(TEAL))
                    canvas.line(x + box + 3, 46, x + box + 13, 46)
                    canvas.line(x + box + 9, 49, x + box + 13, 46)
                    canvas.line(x + box + 9, 43, x + box + 13, 46)
            canvas.setFont("Report", 7.5)
            canvas.setFillColor(colors.HexColor(MUTED))
            canvas.drawCentredString(
                self.width / 2, 8, "One instruction per episode; repeat until success or the step limit."
            )

    by_k = {k: [row for row in rows if row["k"] == k] for k in intervals}
    estimates = {(r["metric"], r["k"]): r for r in stats["estimates"]}
    by_id = {row["episode_id"]: row for row in rows}
    names = [TASK_NAMES.get(task.name, task.name.replace("_", " ")) for task in cfg.tasks]
    story = []

    # 1. Protocol. All numerical settings are taken from the saved manifest.
    story.extend(
        [
            p("VLA / LIBERO · AN EXPERIMENT IN FEEDBACK CONTROL", "kicker"),
            p("How often should a robot replan?", "title"),
            p("A compact study: 30 episodes across three tasks", "h2"),
            p(
                "Objective: measure how the number of actions K between observation updates affects success, "
                "motion and computational cost for one pretrained model. Five preliminary smoke-test episodes "
                "are excluded from the results in this report."
            ),
            ReplanDiagram(),
            p("What the model predicts", "h2"),
            p(
                "SmolVLA receives images from two cameras, the robot state and a language instruction. "
                "Its action expert uses flow matching to predict a sequence of 50 actions. Each action "
                "specifies relative translation and rotation of the end effector, plus a gripper command. "
                "The controller executes the first K actions, obtains a fresh observation and replaces the remaining plan."
            ),
            p("Tasks and initial situations", "h2"),
            table(
                ["Task", "LIBERO suite", "Limit"],
                [[names[i], task.suite, f"{task.max_steps} steps"] for i, task in enumerate(cfg.tasks)],
                [body_width * 0.50, body_width * 0.29, body_width * 0.21],
            ),
            Spacer(1, 9),
            p(
                f"3 tasks × {len(cfg.initial_states)} states ({', '.join(map(str, cfg.initial_states))}) "
                f"× 1 seed ({cfg.policy_seeds[0]}) × 5 values of K = {len(rows)} episodes. "
                "All K values are evaluated from each initial state. Identical physical states and "
                "initial observations are verified using SHA-256 hashes."
            ),
            p("Controlled conditions", "h2"),
            p(
                f"H = {cfg.chunk_size}; {cfg.num_denoising_steps} generation iterations; "
                f"{cfg.control_hz} Hz in simulation; batch = 1; precision {cfg.precision}; "
                f"{cfg.settle_steps} settling steps; {cfg.warmup_calls} warmup calls. "
                "Camera images are 256 × 256 before the checkpoint's standard preprocessing. The model, "
                "normalization, GPU and video recording are identical. Blocks and K values are shuffled using seed "
                f"{cfg.order_seed}. Model noise is tied to the policy seed and simulation step."
            ),
            p(
                "K changes the length of the executed prefix. The prediction horizon H stays fixed; a single action "
                "is not repeated K times. The simulator waits for model inference to finish.",
                "small",
            ),
            PageBreak(),
        ]
    )

    # 2. Outcome counts and the analysis plot.
    success_data = []
    for k in intervals:
        selected = by_k[k]
        outcome = estimates[("success", k)]
        counts = []
        for task in cfg.tasks:
            group = [row for row in selected if row["suite"] == task.suite and row["task"] == task.name]
            counts.append(f"{sum(row['success'] for row in group)}/{len(group)}")
        ci = f"{num(100 * outcome['ci_low'], 0)}-{num(100 * outcome['ci_high'], 0)}%"
        success_data.append([str(k), *counts, f"{sum(r['success'] for r in selected)}/{len(selected)}", ci])
    story.extend(
        [
            p("Results: task completion", "h1"),
            p(
                "Success means satisfying LIBERO's native criterion before the step limit. The table reports "
                "successful episodes divided by all episodes; each K has six episodes."
            ),
            table(
                ["K", "Soup can", "Bowl", "Drawer", "Overall", "95% interval"],
                success_data,
                [body_width * x for x in (0.07, 0.18, 0.16, 0.16, 0.14, 0.29)],
            ),
            Spacer(1, 11),
            Image(str(chart), width=body_width, height=body_width * 7 / 11),
            Spacer(1, 4),
            p(
                "Figure 1. Success, computational cost, model calls and results by task. "
                "Lines connect the tested K values; behavior at intermediate values was not measured.",
                "small",
            ),
            p("Interpreting uncertainty", "h2"),
            p(
                "Means are computed over states within each task, then over tasks with equal weights. "
                "The 95% intervals use 5,000 paired bootstrap resamples of states within each task. "
                "With only two states, these intervals are very coarse: a narrow or zero-width interval does not "
                "establish reliability or equivalence between settings. The conclusions are descriptive."
            ),
            PageBreak(),
        ]
    )

    # 3. Cost, motion proxies and annotation denominator.
    cost_rows = [
        [
            k,
            num(mean(by_k[k], "model_calls")),
            num(mean(by_k[k], "inference_s")),
            num(mean(by_k[k], "simulation_s")),
            num(mean(by_k[k], "compute_s")),
            num(mean(by_k[k], "episode_wall_s")),
        ]
        for k in intervals
    ]
    motion_rows = [
        [
            k,
            num(mean(by_k[k], "eef_path_m"), 2),
            num(mean(by_k[k], "mean_plan_revision_translation_l2"), 3)
            if mean(by_k[k], "mean_plan_revision_translation_l2") is not None
            else "undefined",
            num(mean(by_k[k], "mean_replan_jump_translation_l2"), 3),
            num(mean(by_k[k], "gripper_close_commands")),
        ]
        for k in intervals
    ]
    label_rows = []
    for group in annotations:
        labels = [label for label in all_labels if by_id[label["episode_id"]]["k"] == group["k"]]
        fields = []
        for field in ("missed_grasps", "dropped_objects"):
            known = [int(label[field]) for label in labels if label.get(field, "").strip()]
            if any(count < 0 for count in known):
                raise ValueError("Visual event counts cannot be negative")
            fields.append(f"{sum(known)} [{len(known)}/{group['total_episodes']}]")
        label_rows.append([group["k"], f"{group['reviewed']}/{group['total_episodes']}", *fields])
    story.extend(
        [
            p("Cost, motion and errors", "h1"),
            p(
                "Values are means per episode. The number of calls depends on both K and how early "
                "the episode ends: calls = ceil(steps / K)."
            ),
            table(
                ["K", "Calls", "Model, s", "Simulation, s", "Sum, s", "Episode, s"],
                cost_rows,
                [body_width * x for x in (0.07, 0.15, 0.20, 0.19, 0.19, 0.20)],
            ),
            Spacer(1, 7),
            p(
                f"Entire study: {sum(r['model_calls'] for r in rows)} model calls; "
                f"{sum(r['steps'] for r in rows)} steps; sum of episode durations "
                f"{num(sum(r['episode_wall_s'] for r in rows) / 60)} min. "
                "This excludes installation, weight downloads and process startup with model loading.",
                "small",
            ),
            p(
                "Model: preprocessing + inference + postprocessing, with GPU synchronization. Simulation: "
                "env.step and rendering. Sum = model + simulation. Episode time includes reset, execution, "
                "frame recording and video finalization; it excludes model loading, warmup and JSON-log writes. "
                "Simulated duration = steps / 20; this differs from wall time.",
                "small",
            ),
            p("Motion proxies", "h2"),
            table(
                ["K", "EEF path, m", "Plan revision", "Replan jump", "Gripper closes"],
                motion_rows,
                [body_width * x for x in (0.07, 0.22, 0.24, 0.24, 0.23)],
            ),
            Spacer(1, 6),
            p(
                "Plan revision is the L2 difference between the new first translation vector and the next vector "
                "in the old plan; K = 50 leaves no tail. Replan jump is the L2 difference between consecutive "
                "translation commands at replanning. These are command units, not meters. A changed command "
                "alone does not demonstrate a useful correction.",
                "small",
            ),
            p("Video annotation: coverage and events", "h2"),
            table(
                ["K", "Complete labels", "Misses: sum [n]", "Drops: sum [n]"],
                label_rows,
                [body_width * x for x in (0.08, 0.26, 0.33, 0.33)],
            ),
            Spacer(1, 6),
            p(
                f"Complete labels: {len(reviewed)}/{len(rows)} episodes. Brackets give the number of episodes "
                "with a determinate count for that event, out of six. Sums over known episodes do not imply "
                "zero events in the remaining episodes and do not estimate the event rate for the entire study. "
                "A drop can be clear even when the number of misses is not. A miss means closing the fingers "
                "without retaining the target; failures to hook a handle with an open gripper are excluded. "
                "An assistant annotated frames and logs; there was no independent human review.",
                "small",
            ),
            PageBreak(),
        ]
    )

    # 4. Visually reviewed evidence. No automatically invented explanations.
    story.extend(
        [
            p("Observations from paired episodes", "h1"),
            p(
                "The observations below come from discussion.json. Each comparison uses the same task, "
                "initial state and seed. Possible causes are distinguished from observed behavior."
            ),
        ]
    )
    for index, case in enumerate(discussion["cases"], 1):
        selected = [by_id[identity] for identity in case["episode_ids"]]
        first = selected[0]
        outcomes = "; ".join(
            f"K={r['k']}: {'success' if r['success'] else 'failure'}, "
            f"{r['steps']} steps, {r['model_calls']} calls"
            for r in selected
        )
        short = TASK_NAMES.get(first["task"], first["task"])
        block = [
            p(f"{index}. {case['title']}", "h2"),
            p(f"{short}; state={first['state']}; seed={first['policy_seed']}. {outcomes}.", "small"),
            p("Observation. " + case["observation"]),
            p("Possible explanation. " + case["hypothesis"]),
            p("Verifiable evidence. " + case["evidence"], "small"),
        ]
        story.append(KeepTogether(block))
    correction_counts = Counter(r["correction_after_replan"] for r in reviewed)
    story.extend(
        [
            p("Interpreting changes in motion", "h2"),
            p(
                "The REPLAN marker identifies when a plan is replaced. A subsequent change in direction "
                "establishes a temporal relationship, but does not by itself show that replanning caused "
                "an improvement. A causal claim would require additional controlled interventions."
            ),
            p(
                "correction_after_replan among episodes with complete labels: "
                f"yes={correction_counts['yes']}, no={correction_counts['no']}, "
                f"unclear={correction_counts['unclear']}. A yes label requires specific steps "
                "in annotations.csv, cross-checked against chunks.jsonl.",
                "small",
            ),
            PageBreak(),
        ]
    )

    # 5. Interpretation and limitations.
    story.extend([p("Findings and limitations", "h1"), p("What this study shows", "h2")])
    for index, finding in enumerate(discussion["findings"], 1):
        story.append(p(f"{index}. {finding}"))
    story.extend(
        [
            p("Limitations", "h2"),
            p(
                "There are only two initial situations per task and one generation seed. A difference of one "
                "episode corresponds to 50 percentage points within a task. This study illustrates behavior and "
                "cost, but does not precisely estimate the probability of success in arbitrary situations."
            ),
            p(
                "Three tasks from one simulator are evaluated with one checkpoint. Generalization to other "
                "models, task sets, object properties or a physical robot requires new evaluation. "
                "Contact and grasping errors may have several possible causes."
            ),
            p(
                "Success is LIBERO's native signal, and the episode ends immediately afterward. The can "
                "entering the target region may count as success while it is still held by the fingers. "
                "Stable placement after release and subsequent seconds of motion were not separately tested."
            ),
            p(
                "Calls are synchronous: physics pauses while the GPU predicts a plan. Thus, 20 Hz is the "
                "simulated control frequency. This experiment does not evaluate a physical robot under "
                "inference latency or demonstrate real-time operation."
            ),
            p(
                "Early termination changes path length, call count and episode duration. A shorter path does "
                "not necessarily mean better motion, and lower cost does not substitute for success. "
                "Visual annotation is subjective; unreviewed or ambiguous events must remain unknown."
            ),
            p("Conclusion", "h2"),
            p(discussion["conclusion"]),
            p("Possible extensions", "h2"),
            p(
                "A next step would be more independent initial states, followed by multiple seeds. "
                "Object perturbations and adaptive K require a separate, predefined protocol. "
                "These additional experiments are outside the scope of the current compact study."
            ),
            PageBreak(),
        ]
    )

    # 6. Exact provenance and commands for rerunning the delivered experiment.
    hardware = manifest["hardware"]
    runtime = manifest["runtime"]
    packages = runtime["packages"]
    macos = hardware.get("backend") == "mps"
    setup_script = "macos_setup.sh" if macos else "remote_setup.sh"
    run_script = "run_macos.sh" if macos else "run_remote.sh"
    reproduction_output = "outputs/mac-compact" if macos else "outputs/compact"
    story.extend(
        [
            p("Reproducibility and materials", "h1"),
            p(
                f"Run created: {manifest['created_utc']}. Device: {hardware['device']}; "
                f"{('MPS / Metal' if hardware.get('backend') == 'mps' else 'CUDA ' + str(hardware['cuda']))}; "
                f"Python {runtime['python']}; {runtime['system']}. "
                f"LeRobot {packages['lerobot']}; hf-libero {packages['hf-libero']}; "
                f"MuJoCo {packages['mujoco']}; PyTorch {packages['torch']}.",
                "small",
            ),
            p(
                f"Model: {cfg.model_id}\nrevision: {cfg.model_revision}\n"
                f"Backbone: {cfg.backbone_id}\nrevision: {cfg.backbone_revision}",
                "code",
            ),
            p("Commands from the repository root", "h2"),
            p(
                f"bash scripts/{setup_script}\n"
                f"bash scripts/{run_script} configs/compact.json {reproduction_output}\n"
                f".venv/bin/python scripts/finish_run.py {reproduction_output}",
                "code",
            ),
            p(
                "After reviewing videos and completing annotations.csv, rerun the analysis and record "
                "observations and testable hypotheses in discussion.json. PDF generation requires reportlab and pypdf.",
                "small",
            ),
            p(
                f".venv/bin/python -m vla_replan analyze {reproduction_output} "
                f"--output {reproduction_output}/report --annotations {reproduction_output}/annotations.csv\n"
                f".venv/bin/python scripts/build_figures.py --run {reproduction_output}\n"
                f"python scripts/build_report.py --run {reproduction_output} "
                f"--discussion {reproduction_output}/discussion.json",
                "code",
            ),
            p("Saved materials", "h2"),
            p(
                "manifest.json - conditions, versions and hashes; episodes/*/summary.json - outcomes and costs; "
                "steps.jsonl and chunks.jsonl - motion, actions and replanning steps; video.mp4 - two camera "
                "views, instruction and outcome; report/ - tables and figures; demos/ - paired K=1 and K=50 "
                "videos and motion traces; annotations.csv - visual labels; discussion.json - interpretation.",
                "small",
            ),
            p(
                f"Config SHA-256: {manifest['config_fingerprint']}\n"
                f"Code SHA-256: {runtime['code_sha256']}\n"
                f"Lock SHA-256: {runtime.get('dependency_lock_sha256', 'not recorded')}\n"
                f"Discussion SHA-256: {hashlib.sha256(args.discussion.read_bytes()).hexdigest()}",
                "code",
            ),
            p("References", "h2"),
        ]
    )
    references = [
        (
            "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics",
            "https://arxiv.org/abs/2506.01844",
        ),
        (
            "SmolVLA LIBERO checkpoint, pinned revision",
            f"https://huggingface.co/{cfg.model_id}/tree/{cfg.model_revision}",
        ),
        (
            "LeRobot 0.6.1: SmolVLA implementation",
            "https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py",
        ),
        (
            "LeRobot 0.6.1: LIBERO adapter",
            "https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/envs/libero.py",
        ),
        ("LIBERO environment documentation", "https://huggingface.co/docs/lerobot/en/libero"),
    ]
    for index, (label, url) in enumerate(references, 1):
        story.append(
            p(
                f'{index}. <link href="{escape(url)}" color="{TEAL}">{clean(label)}</link>',
                "small",
                markup=True,
            )
        )
    story.append(
        p(
            "All numerical results in this report come from completed episodes in this study. "
            "The references above describe the model and software environment, not the outcomes measured here.",
            "small",
        )
    )

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D4E0E8"))
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont("Report", 7.4)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, height - 11 * mm, "SMOLVLA / LIBERO   |   K = 1, 5, 10, 25, 50")
        canvas.drawString(18 * mm, 11 * mm, "Compact study · 30 episodes · matched initial states")
        canvas.drawRightString(width - 18 * mm, 11 * mm, f"{doc.page} / 6")
        canvas.restoreState()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        doc = SimpleDocTemplate(
            str(staged),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=21 * mm,
            bottomMargin=20 * mm,
            title="How often should a robot replan?",
            author="VLA replanning experiment",
            subject="SmolVLA / LIBERO: compact paired experiment",
        )
        doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
        reader = PdfReader(staged)
        if len(reader.pages) != 6:
            raise ValueError(
                f"Report overflow: expected 6 pages, got {len(reader.pages)}. "
                "Shorten the discussion text before publishing."
            )
        if any(not page.extract_text().strip() for page in reader.pages):
            raise ValueError("Report contains an empty page")
        staged.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "pages": 6,
                "episodes": len(rows),
                "reviewed": len(reviewed),
                "visual_review_required": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, default=Path("outputs/compact"))
    parser.add_argument("--discussion", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/pdf/vla-replanning-report.pdf"))
    parser.add_argument("--font", type=Path, help="Regular TTF font")
    parser.add_argument("--font-bold", type=Path, help="Matching bold TTF font")
    args = parser.parse_args(argv)
    build_report(args)


if __name__ == "__main__":
    main()
