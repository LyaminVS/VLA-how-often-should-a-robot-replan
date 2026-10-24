"""Load immutable historical fixed controls for per-episode first-action gates."""

import json
import math
from pathlib import Path

from vla_replan.config import Episode as HistoricalEpisode

from .integrity import file_sha256

MATCH_FIELDS = (
    "model_id", "model_revision", "backbone_id", "backbone_revision", "chunk_size",
    "num_denoising_steps", "control_hz", "settle_steps", "precision", "record_video", "warmup_calls",
)


def validate_reference_runtime(audit, metadata, hardware):
    """Permit only the explicitly recorded CPU-thread difference in historical controls."""
    differences = []
    for old in audit["runtime_comparison"]:
        if old["runtime"] != metadata:
            raise ValueError(f"Historical software/runtime differs: {old['root']}")
        allowed = "torch_cpu_threads"
        before = {key: value for key, value in old["hardware"].items() if key != allowed}
        after = {key: value for key, value in hardware.items() if key != allowed}
        if before != after:
            raise ValueError(f"Historical hardware or numerical settings differ: {old['root']}")
        differences.append({"root": old["root"], "historical_torch_cpu_threads": old["hardware"].get(allowed),
                            "current_torch_cpu_threads": hardware.get(allowed)})
    return {"runtime_equal": True, "hardware_equal_except": ["torch_cpu_threads"], "differences": differences}


def load_references(roots, cfg, episodes=None):
    selected = cfg.episodes() if episodes is None else episodes
    required = {episode.pair_id for episode in selected}
    selected_pairs = {episode.pair_id: episode for episode in selected}
    expected_k = set(cfg.base.intervals)
    groups = {}
    files = {}
    runtime_comparison = []
    resolved_roots = []
    base = cfg.base.asdict()
    for input_root in roots:
        root = Path(input_root).resolve()
        if str(root) in resolved_roots:
            raise ValueError("Duplicate historical reference root")
        resolved_roots.append(str(root))
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("data_kind") != "real_libero":
            raise ValueError("Historical controls must be real LIBERO data")
        for field in MATCH_FIELDS:
            if manifest["config"].get(field) != base[field]:
                raise ValueError(f"Historical inference conditions differ: {root}: {field}")
        historical_tasks = {(task["suite"], task["name"]): task["max_steps"] for task in manifest["config"]["tasks"]}
        for task in cfg.base.tasks:
            if historical_tasks.get((task.suite, task.name)) != task.max_steps:
                raise ValueError(f"Historical task identity or action limit differs: {task.name}")
        files[str(manifest_path)] = file_sha256(manifest_path)
        runtime_comparison.append({"root": str(root), "runtime": manifest["runtime"],
                                   "hardware": manifest["hardware"]})
        for summary_path in sorted(root.glob("episodes/*/summary.json")):
            summary = json.loads(summary_path.read_text())
            pair_id = summary.get("pair_id")
            if pair_id not in required:
                continue
            k = summary.get("k")
            if k not in expected_k or type(k) is not int:
                raise ValueError("Unexpected historical fixed interval")
            requested = selected_pairs[pair_id]
            expected = HistoricalEpisode(requested.suite, requested.task, requested.state,
                                         requested.policy_seed, k, requested.max_steps)
            if (any(summary.get(key) != value for key, value in expected.asdict().items())
                    or summary.get("episode_id") != expected.id or summary_path.parent.name != expected.id
                    or summary.get("pair_id") != expected.pair_id or summary.get("env_seed") != expected.env_seed
                    or summary.get("config_fingerprint") != manifest["config_fingerprint"]
                    or summary.get("data_kind") != "real_libero"
                    or requested.state not in manifest["config"]["initial_states"]
                    or requested.policy_seed not in manifest["config"]["policy_seeds"]
                    or k not in manifest["config"]["intervals"]):
                raise ValueError(f"Historical summary identity differs from its manifest: {summary_path}")
            group = groups.setdefault(pair_id, {})
            if k in group:
                raise ValueError(f"Duplicate historical control: {pair_id}, K={k}")
            chunk_path = summary_path.parent / "chunks.jsonl"
            with chunk_path.open() as stream:
                first = json.loads(next(stream))
            actions = first.get("actions")
            if (first.get("step") != 0 or not isinstance(actions, list) or len(actions) != cfg.base.chunk_size
                    or any(not isinstance(action, list) or len(action) != 7 for action in actions)
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for action in actions for value in action)):
                raise ValueError("Invalid historical first complete prediction")
            expected_seed = summary["policy_seed"] * 1_000_003 % (2**63 - 1)
            if first.get("sampling_seed") != expected_seed:
                raise ValueError("Invalid historical first-call seed")
            for field in ("initial_state_sha256", "initial_observation_sha256"):
                if not isinstance(summary.get(field), str) or len(summary[field]) != 64:
                    raise ValueError(f"Missing historical initial hash: {field}")
            group[k] = {"summary": summary, "chunk": first}
            files[str(summary_path)] = file_sha256(summary_path)
            files[str(chunk_path)] = file_sha256(chunk_path)
    if set(groups) != required:
        raise ValueError("Historical controls do not cover every requested task/state/seed pair")
    references = {}
    for pair_id, group in sorted(groups.items()):
        if set(group) != expected_k:
            raise ValueError(f"Missing historical fixed intervals for {pair_id}")
        first = group[min(group)]
        reference = {
            "pair_id": pair_id,
            "initial_state_sha256": first["summary"]["initial_state_sha256"],
            "initial_observation_sha256": first["summary"]["initial_observation_sha256"],
            "sampling_seed": first["chunk"]["sampling_seed"],
            "actions": first["chunk"]["actions"],
            "reference_episode_ids": [item["summary"]["episode_id"] for _, item in sorted(group.items())],
        }
        for item in group.values():
            for field in ("initial_state_sha256", "initial_observation_sha256"):
                if item["summary"][field] != reference[field]:
                    raise ValueError(f"Historical fixed initial states disagree: {pair_id}")
            if (item["chunk"]["sampling_seed"] != reference["sampling_seed"]
                    or item["chunk"]["actions"] != reference["actions"]):
                raise ValueError(f"Historical first complete predictions disagree: {pair_id}")
        references[pair_id] = reference
    audit = {"roots": resolved_roots, "files": dict(sorted(files.items())), "pairs": len(references),
             "episodes": sum(len(group) for group in groups.values()),
             "config_match_fields": [*MATCH_FIELDS, "tasks.suite/name/max_steps"], "runtime_comparison": runtime_comparison}
    return references, audit
