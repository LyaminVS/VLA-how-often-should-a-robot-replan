"""Bind every new start to retained C=50 observation and complete action hashes."""

import json
from pathlib import Path
from capped_replan.integrity import file_sha256, source_manifest as capped_sources
from capped_replan.reference import MATCH_FIELDS


def load_references(root, cfg):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    base = json.loads(json.dumps(cfg.base.asdict()))
    if manifest.get("data_kind") != "real_libero" or manifest["capped_source"] != capped_sources():
        raise ValueError("Historical model/rollout sources changed")
    for field in (*MATCH_FIELDS, "tasks"):
        if manifest["config"]["base"][field] != base[field]:
            raise ValueError(f"Historical inference setting differs: {field}")
    files = {name: file_sha256(root / name) for name in ("manifest.json", "invariance.json")}
    required = {episode.pair_id for episode in cfg.episodes()}
    references = {}
    for path in sorted(root.glob("episodes/*/summary.json")):
        row = json.loads(path.read_text())
        if row.get("method") != "dvac_cap_50" or row.get("pair_id") not in required:
            continue
        if (row["manifest_sha256"] != files["manifest.json"]
                or row["config_fingerprint"] != manifest["config_fingerprint"]
                or row["episode_id"] != path.parent.name or row["data_kind"] != "real_libero"):
            raise ValueError("Historical reference identity differs")
        if set(row.get("evidence_sha256", {})) != {"chunks.jsonl", "steps.jsonl", "video.mp4"}:
            raise ValueError("Missing historical reference evidence")
        files[str(path.relative_to(root))] = file_sha256(path)
        for filename, checksum in row["evidence_sha256"].items():
            artifact = path.parent / filename
            if file_sha256(artifact) != checksum:
                raise ValueError("Historical evidence checksum mismatch")
            files[str(artifact.relative_to(root))] = checksum
        with (path.parent / "chunks.jsonl").open() as stream:
            first = json.loads(next(stream))
        if row["pair_id"] in references or first["step"] != 0:
            raise ValueError("Duplicate or invalid reference")
        references[row["pair_id"]] = {
            "pair_id": row["pair_id"], "initial_state_sha256": row["initial_state_sha256"],
            "initial_observation_sha256": row["initial_observation_sha256"],
            "sampling_seed": first["sampling_seed"], "actions": first["actions"],
            "reference_episode_ids": [row["episode_id"]],
        }
    if set(references) != required:
        raise ValueError("References must cover every planned task/state/seed")
    return references, {"root": str(root), "files": dict(sorted(files.items()))}, manifest


def validate_runtime(old, metadata, policy):
    if (old["runtime"] != metadata or old["hardware"] != policy.runtime
            or old["upstream_sampler"] != policy.sampler_manifest()):
        raise ValueError("Historical adaptive runtime, hardware or sampler differs")
