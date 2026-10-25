"""Portable collection uses the same disturbance protocol without pooling GPUs."""

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from action_dropout.config import DropoutExperiment
from action_dropout.core import run_episode
from scripts import run_action_dropout_kaggle as portable
from scripts.build_action_dropout_kaggle_bundle import notebook
from vla_replan.core import Observation
from vla_replan.storage import atomic_json, write_jsonl


def configuration():
    data = json.loads((Path(__file__).resolve().parents[1] / "configs/action-dropout.json").read_text())
    data["methods"], data["probabilities"] = ["fixed_1"], [.1, .3]
    return DropoutExperiment.fromdict(data)


class Environment:
    def reset(self, episode):
        self.steps = 0
        return Observation(None, [0., 0., 0.], [0., 0.]), {
            "initial_state_sha256": "a" * 64, "initial_observation_sha256": "b" * 64}

    def step(self, action):
        self.steps += 1
        return Observation(None, [self.steps * .01, 0., 0.], [0., 0.]), self.steps == 3, False


class Policy:
    def reset(self):
        pass

    def predict(self, obs, seed):
        self.last_diagnostics = {"clean_estimates": [[[0.] * 7 for _ in range(50)] for _ in range(5)],
                                 "variance": [0.] * 50, "fm_times": [.5, .4, .3, .2, .1],
                                 "velocity_evaluations": 10, "action_dimensions": 7,
                                 "statistics_dtype": "float64", "diagnostics_finalize_s": 0.}
        return [[.2, -.3, .4, .1, -.2, .3, 1.] for _ in range(50)]


def write_fixture(tmp_path, monkeypatch, cfg=None, selected=None):
    cfg = configuration() if cfg is None else cfg
    selected = portable.plan(cfg)[:2] if selected is None else selected
    home, raw = {}, []
    for episode in selected:
        summary, rows, chunks = run_episode(Environment(), Policy(), episode,
            selector=portable.make_selector(cfg, episode))
        reference = portable.local_reference(summary, chunks[0])
        home[episode.pair_id] = reference
        summary.update(home_comparison=portable.compare_home(reference, summary, chunks[0]), episode_wall_s=1.)
        raw.append((episode, summary, rows, chunks))
    root, output = tmp_path / "source", tmp_path / "results"
    root.mkdir()
    output.mkdir()
    atomic_json(root / "references/home-first-predictions.json", {"references": home})
    monkeypatch.setattr(portable, "ROOT", root)
    bundle = {"manifest_sha256": "test", "files": {}}
    monkeypatch.setattr(portable, "bundle_evidence", lambda: bundle)
    invariance = {"passed": True, "kind": "kaggle_p0_passthrough", "bundle": bundle,
                  "runtime": {}, "hardware": {}, "checks": [
                      {"exact_action_equality": True, "exact_observation_equality": True}] * 3}
    atomic_json(output / "invariance.json", invariance)
    manifest = {"config": json.loads(json.dumps(cfg.asdict())), "config_fingerprint": cfg.fingerprint,
                "planned_episode_ids": [e.id for e in portable.plan(cfg)], "bundle": bundle,
                "home_reference_sha256": portable.sha(root / "references/home-first-predictions.json"),
                "invariance_sha256": portable.sha(output / "invariance.json"), "runtime": {}, "hardware": {},
                "experiment_kind": "kaggle_translation_dropout", "data_kind": "real_libero"}
    atomic_json(output / "manifest.json", manifest)
    for episode, summary, rows, chunks in raw:
        directory = output / "episodes" / episode.id
        directory.mkdir(parents=True)
        summary.update(config_fingerprint=cfg.fingerprint, manifest_sha256=portable.sha(output / "manifest.json"))
        write_jsonl(directory / "steps.jsonl", rows)
        write_jsonl(directory / "chunks.jsonl", chunks)
        (directory / "video.mp4").write_bytes(b"fixture only")
        summary["evidence_sha256"] = {name: portable.sha(directory / name)
                                       for name in ("steps.jsonl", "chunks.jsonl", "video.mp4")}
        atomic_json(directory / "summary.json", summary)
    return cfg, output, manifest


def test_plan_is_exactly_sixty_matched_episodes_without_p0():
    selected = portable.plan(configuration())
    assert len(selected) == 60
    assert {e.dropout_p for e in selected} == {.1, .3}
    for low, high in zip(selected[::2], selected[1::2], strict=True):
        assert low.pair_id == high.pair_id
        assert low.dropout_p != high.dropout_p


def test_wrong_cohort_size_is_rejected():
    cfg = configuration()
    with pytest.raises(ValueError, match="60 episodes"):
        portable.plan(replace(cfg, base=replace(cfg.base, initial_states=(0,))))


def test_small_gpu_numerical_differences_are_reported_not_declared_exact():
    summary = {"initial_state_sha256": "state", "initial_observation_sha256": "new"}
    reference = {**summary, "initial_observation_sha256": "old", "sampling_seed": 4, "actions": [[.1] * 7]}
    compared = portable.compare_home(reference, summary, {"sampling_seed": 4, "actions": [[.10001] * 7]})
    assert compared["same_initial_state"] and not compared["same_initial_observation"]
    assert not compared["first_chunk_exact_equal"]
    assert compared["first_chunk_allclose_atol_1e4_rtol_1e4"]


def test_physical_reset_mismatch_stops_before_any_action():
    episode = portable.plan(configuration())[0]
    gate = portable.HomeStateGate(Environment(), {episode.pair_id: {"initial_state_sha256": "wrong"}})
    with pytest.raises(ValueError, match="Physical LIBERO"):
        gate.reset(episode)


def test_bundle_source_mutation_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(portable, "ROOT", tmp_path)
    code = tmp_path / "source.py"
    code.write_text("original\n")
    atomic_json(tmp_path / "bundle-manifest.json", {"files": {"source.py": hashlib.sha256(code.read_bytes()).hexdigest()}})
    portable.bundle_evidence()
    code.write_text("changed\n")
    with pytest.raises(ValueError, match="Bundle source"):
        portable.bundle_evidence()


def test_paired_evidence_verification_and_export(tmp_path, monkeypatch):
    cfg, output, manifest = write_fixture(tmp_path, monkeypatch)
    records, refs = portable.verify_completed(output, cfg, manifest)
    assert len(records) == 2 and len(refs) == 1
    portable.summarize(output, cfg, records)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["complete_matched_starts"] == 1
    assert [r["episodes"] for r in progress["conditions"]] == [1, 1]
    target = tmp_path / "checkpoint.zip"
    portable.export_zip(output, target)
    assert target.stat().st_size > 0


@pytest.mark.parametrize("field", ["config_fingerprint", "planned_episode_ids", "invariance_sha256", "bundle"])
def test_resume_manifest_mutations_are_rejected(tmp_path, monkeypatch, field):
    cfg, output, manifest = write_fixture(tmp_path, monkeypatch)
    manifest[field] = "changed"
    with pytest.raises(ValueError):
        portable.verify_completed(output, cfg, manifest)


def test_even_rehashed_wrong_applied_action_is_rejected(tmp_path, monkeypatch):
    cfg, output, manifest = write_fixture(tmp_path, monkeypatch)
    path = next(output.glob("episodes/*/summary.json"))
    summary = json.loads(path.read_text())
    steps_path = path.parent / "steps.jsonl"
    rows = [json.loads(line) for line in steps_path.read_text().splitlines()]
    rows[0]["applied_action"][6] = -1.
    write_jsonl(steps_path, rows)
    summary["evidence_sha256"]["steps.jsonl"] = portable.sha(steps_path)
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="dropout mask"):
        portable.verify_completed(output, cfg, manifest)


def test_changed_home_diagnostic_is_rejected(tmp_path, monkeypatch):
    cfg, output, manifest = write_fixture(tmp_path, monkeypatch)
    path = next(output.glob("episodes/*/summary.json"))
    summary = json.loads(path.read_text())
    summary["home_comparison"]["first_chunk_exact_equal"] = False
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="historical comparison"):
        portable.verify_completed(output, cfg, manifest)


def test_generated_notebook_has_compilable_cells_and_no_extra_baseline():
    document = notebook("eA==")
    for item in document["cells"]:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), "notebook-cell", "exec")
    prose = "".join("".join(cell["source"]) for cell in document["cells"])
    assert "60 episodes" in prose
    assert "does not add 30 baseline episodes" in prose


def p02_configuration():
    return replace(configuration(), methods=portable.ALL_METHODS, probabilities=(.2,))


def test_p02_plan_has_every_method_and_matched_start_once_k1_last():
    cfg = p02_configuration()
    episodes = portable.plan(cfg)
    assert len(episodes) == len({episode.id for episode in episodes}) == 270
    assert all(episode.method != "fixed_1" for episode in episodes[:240])
    assert all(episode.method == "fixed_1" for episode in episodes[240:])
    assert {episode.dropout_p for episode in episodes} == {.2}
    for method in portable.ALL_METHODS:
        method_episodes = [episode for episode in episodes if episode.method == method]
        assert len(method_episodes) == 30
        assert len({episode.pair_id for episode in method_episodes}) == 30


@pytest.mark.parametrize("methods,probabilities", [(("fixed_1",), (.2,)),
    (portable.ALL_METHODS[:-1], (.2,)), (portable.ALL_METHODS, (.1, .2)),
    (portable.ALL_METHODS, (.2, .3))])
def test_unregistered_portable_profiles_rejected(methods, probabilities):
    with pytest.raises(ValueError, match="Registered portable cohorts"):
        portable.plan(replace(configuration(), methods=methods, probabilities=probabilities))


def test_p02_saved_fixed_and_dvac_evidence_is_independently_verified(tmp_path, monkeypatch):
    cfg = p02_configuration()
    episodes = portable.plan(cfg)
    selected = [episode for episode in episodes if episode.pair_id == episodes[0].pair_id]
    cfg, output, manifest = write_fixture(tmp_path, monkeypatch, cfg, selected)
    records, refs = portable.verify_completed(output, cfg, manifest)
    assert len(records) == 9 and len(refs) == 1
    for episode in selected:
        chunks = [json.loads(line) for line in
                  (output / "episodes" / episode.id / "chunks.jsonl").read_text().splitlines()]
        if episode.adaptive:
            assert len(chunks) == 2
            assert [chunk["decision"]["chosen_k"] for chunk in chunks] == [1, episode.execution_cap]
            assert chunks[-1]["decision"]["raw_chosen_k"] == 50
        elif episode.k == 1:
            assert len(chunks) == 3
        else:
            assert len(chunks) == 1


@pytest.mark.parametrize("mutation", ["variance", "chosen_k", "raw_chosen_k", "step_offset", "action_prefix"])
def test_p02_rehashed_invalid_policy_evidence_is_rejected(tmp_path, monkeypatch, mutation):
    cfg = p02_configuration()
    selected = [next(episode for episode in portable.plan(cfg) if episode.method == "dvac_cap_5")]
    cfg, output, manifest = write_fixture(tmp_path, monkeypatch, cfg, selected)
    path = next(output.glob("episodes/*/summary.json"))
    summary = json.loads(path.read_text())
    trace = "steps.jsonl" if mutation == "step_offset" else "chunks.jsonl"
    records = [json.loads(line) for line in (path.parent / trace).read_text().splitlines()]
    if mutation == "variance":
        records[1]["diagnostics"]["variance"][0] = .2
    elif mutation in ("chosen_k", "raw_chosen_k"):
        records[1]["decision"][mutation] = 4
    elif mutation == "step_offset":
        records[2]["chunk_index"] = 0
    else:
        records[1]["actions"][1][0] = .9
    write_jsonl(path.parent / trace, records)
    summary["evidence_sha256"][trace] = portable.sha(path.parent / trace)
    atomic_json(path, summary)
    with pytest.raises(ValueError):
        portable.verify_completed(output, cfg, manifest)


def test_p02_summaries_use_complete_method_panels_not_incomplete_rows(tmp_path):
    cfg = p02_configuration()
    episodes = portable.plan(cfg)
    selected = [episode for episode in episodes if episode.method == "fixed_5"]
    selected += [next(episode for episode in episodes if episode.method == "fixed_10")]
    records = {episode.id: {"pair_id": episode.pair_id, "method": episode.method,
        "success": True, "steps": 10, "model_calls": 2, "episode_wall_s": 1.,
        "translation_drop_count": 2, "dropout_p": .2} for episode in selected}
    portable.summarize(tmp_path, cfg, records)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["completed_episodes"] == 31 and progress["planned_episodes"] == 270
    assert progress["complete_matched_starts"] == 0
    assert progress["complete_methods"] == ["fixed_5"]
    assert [(row["method"], row["episodes"]) for row in progress["conditions"]] == [("fixed_5", 30)]
    assert len(progress["missing_episode_ids"]) == 239


def test_p02_zip_has_separate_restore_root(tmp_path):
    output = tmp_path / "action-dropout-kaggle-p020"
    output.mkdir()
    atomic_json(output / "progress.json", {"completed_episodes": 0})
    archive = tmp_path / "checkpoint.zip"
    portable.export_zip(output, archive)
    with zipfile.ZipFile(archive) as result:
        assert result.namelist() == ["action-dropout-kaggle-p020/progress.json"]
