"""Independent evidence replay and paired descriptive analysis safeguards."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_replan.config import AdaptiveExperiment, AdaptiveEpisode
from scripts.analyze_dvac import paired_statistics, wilson
from scripts.verify_dvac import recompute_variance, replay_decision, verify_episode, verify_inherited_sources


def test_population_variance_uses_solver_iterations_and_sums_valid_coordinates():
    estimates = [[[float(i), float(2 * i), 0, 0, 0, 0, 0]] for i in range(5)]
    assert recompute_variance(estimates, horizon=1) == [10.0]
    with pytest.raises(ValueError, match="vector length"):
        recompute_variance([[[0] * 8] for _ in range(5)], horizon=1)


def test_threshold_uses_only_preceding_values_and_strict_zero_based_crossing():
    previous = [[0, 0, 0, 0]]
    decision = replay_decision([0, 0, 1, 0], previous)
    assert decision["threshold"] == 0
    assert decision["first_crossing"] == 2
    assert decision["chosen_k"] == 2
    assert replay_decision([1, 0, 0, 0], previous)["chosen_k"] == 1
    assert replay_decision([0, 0, 0, 0], previous)["chosen_k"] == 4
    assert replay_decision([100, 100, 100, 100], [])["threshold"] is None
    assert previous == [[0, 0, 0, 0]]


def fixture_episode(method="dvac"):
    cfg = AdaptiveExperiment.load(Path(__file__).resolve().parents[1] / "configs/dvac.json")
    cfg = replace(cfg, base=replace(cfg.base, record_video=False))
    episode = AdaptiveEpisode("libero_object", "fixture_not_real_data", 15, 11, method, 3)
    clean = [[[0.0] * 7 for _ in range(50)] for _ in range(5)]
    chunks, rows, history = [], [], []
    step = 0
    while step < 3:
        if method == "dvac":
            decision = replay_decision([0.0] * 50, history)
            history.append([0.0] * 50)
        else:
            decision = {"chosen_k": 5, "threshold": None, "first_crossing": None,
                        "window_calls": 0, "window_mean": None, "window_std": None, "bootstrap": False}
        executed = min(decision["chosen_k"], 3 - step)
        chunk = {"step": step, "sampling_seed": (11 * 1_000_003 + step * 97_409) % (2**63 - 1),
                 "actions": [[0.0] * 7 for _ in range(50)], "executed_steps": executed,
                 "inference_s": 1.0, "selector_s": .01 if method == "dvac" else 0.0, "decision": decision,
                 "diagnostics": {"clean_estimates": copy.deepcopy(clean), "variance": [0.0] * 50,
                                 "fm_times": [.5, .4, .3, .2, .1], "velocity_evaluations": 10,
                                 "action_dimensions": 7, "statistics_dtype": "float64", "diagnostics_finalize_s": .001}}
        chunks.append(chunk)
        for offset in range(executed):
            rows.append({"step": step + offset, "sim_time_s": (step + offset) / 20,
                         "chunk_index": offset, "model_calls": len(chunks), "chosen_k": decision["chosen_k"],
                         "next_replan_step": step + decision["chosen_k"], "action": [0.0] * 7,
                         "replanned": offset == 0, "threshold": decision["threshold"],
                         "success_after_action": False, "inference_s": 1.0 if offset == 0 else 0.0,
                         "selector_s": chunk["selector_s"] if offset == 0 else 0.0})
        step += executed
    ks = [chunk["decision"]["chosen_k"] for chunk in chunks]
    summary = {**episode.asdict(), "episode_id": episode.id, "pair_id": episode.pair_id,
               "steps": 3, "model_calls": len(chunks), "inference_s": float(len(chunks)),
               "selector_s": .01 * len(chunks) if method == "dvac" else 0.0,
               "simulation_s": .1, "compute_s": sum(c["inference_s"] + c["selector_s"] for c in chunks) + .1,
               "episode_wall_s": 5.0, "sim_time_s": .15, "eef_path_m": 0.0,
               "mean_chosen_k": sum(ks) / len(ks), "min_chosen_k": min(ks), "max_chosen_k": max(ks),
               "mean_executed_prefix": 3 / len(chunks), "clipped_action_values": 0, "gripper_close_commands": 0,
               "bootstrap_calls": int(method == "dvac"), "success": False, "termination": "time_limit"}
    return cfg, summary, rows, chunks


@pytest.mark.parametrize("method", ["dvac", "fixed_5"])
def test_evidence_replay_accepts_short_final_prefix_without_extra_call(method):
    cfg, summary, rows, chunks = fixture_episode(method)
    compact, report = verify_episode(summary, rows, chunks, cfg)
    assert report["steps"] == 3
    assert report["max_variance_error"] == 0
    assert compact[-1]["executed_steps"] < compact[-1]["decision"]["chosen_k"]


@pytest.mark.parametrize("mutation,pattern", [
    (lambda s, r, c: c[1]["decision"].update(chosen_k=49), "Decision replay"),
    (lambda s, r, c: c[0]["diagnostics"]["variance"].__setitem__(0, .01), "denoising variance"),
    (lambda s, r, c: c[1].update(sampling_seed=0), "sampling seed"),
    (lambda s, r, c: r[1]["action"].__setitem__(0, .01), "Executed action"),
    (lambda s, r, c: r[1].update(replanned=False), "replanning marker"),
    (lambda s, r, c: r[0].update(success_after_action=True), "continued after"),
    (lambda s, r, c: c[1]["decision"].update(window_calls=2), "Decision replay"),
])
def test_evidence_replay_rejects_corrupted_decisions_or_actions(mutation, pattern):
    cfg, summary, rows, chunks = fixture_episode()
    mutation(summary, rows, chunks)
    with pytest.raises(ValueError, match=pattern):
        verify_episode(summary, rows, chunks, cfg)


def test_paired_statistics_keep_seed_pairs_and_physical_settings_distinct():
    records = []
    for state in (15, 16):
        for seed in (11, 22):
            for method in ("fixed_5", "dvac"):
                records.append({"summary": {
                    "pair_id": f"task/state{state}/seed{seed}", "suite": "suite", "task": "task", "state": state,
                    "policy_seed": seed, "method": method, "success": method == "dvac" and seed == 11,
                    "model_calls": 4 if method == "dvac" else 5, "steps": 10,
                    **{field: 1.0 for field in ("inference_s", "selector_s", "simulation_s", "compute_s", "episode_wall_s", "sim_time_s")},
                }})
    comparisons, outcomes, physical = paired_statistics(records, ("fixed_5", "dvac"))
    assert len(outcomes) == 4 and len(physical) == 2
    assert comparisons[0]["dvac_only_success"] == 2
    assert comparisons[0]["success_rate_difference"] == .5
    assert all(row["policy_seed_repetitions"] == 2 for row in physical)
    assert all(row["success_rate_difference"] == .5 for row in physical)
    with pytest.raises(ValueError, match="every method"):
        paired_statistics(records[:-1], ("fixed_5", "dvac"))


def test_descriptive_wilson_handles_zero_and_all_successes():
    low, high = wilson(0, 12)
    assert low == 0 and .24 < high < .25
    low, high = wilson(12, 12)
    assert .75 < low < .76 and high == 1


def test_inherited_backend_and_dependency_hashes_detect_later_changes(tmp_path):
    (tmp_path / "vla_replan").mkdir()
    backend = tmp_path / "vla_replan/backend.py"
    backend.write_bytes(b"original backend\n")
    (tmp_path / "requirements").mkdir()
    lock = tmp_path / "requirements/linux.lock.txt"
    lock.write_bytes(b"frozen==1.0\n")
    expected = {
        "code_sha256": hashlib.sha256(b"backend.pyoriginal backend\n").hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(b"frozen==1.0\n").hexdigest(),
    }
    assert verify_inherited_sources(expected, tmp_path) == expected
    backend.write_bytes(b"modified backend\n")
    with pytest.raises(ValueError, match="code_sha256"):
        verify_inherited_sources(expected, tmp_path)
    backend.write_bytes(b"original backend\n")
    lock.write_bytes(b"frozen==2.0\n")
    with pytest.raises(ValueError, match="dependency_lock_sha256"):
        verify_inherited_sources(expected, tmp_path)
