"""Media selection must be fixed or explicitly outcome-selected, never hand picked."""

from types import SimpleNamespace

import pytest

from scripts.build_capped_demo import METHODS, predetermined_sections
from scripts.build_capped_review import select_groups, select_steps


def example_records():
    cfg = SimpleNamespace(base=SimpleNamespace(tasks=[SimpleNamespace(suite="suite", name=name)
                                                    for name in ("alpha", "beta", "gamma")]))
    records = []
    for task in ("alpha", "beta", "gamma"):
        for state in (0, 1, 2):
            for method in METHODS:
                records.append({"summary": {
                    "suite": "suite", "task": task, "state": state, "policy_seed": 11, "method": method,
                    "episode_id": f"{task}_{state}_{method}", "pair_id": f"suite/{task}/state{state}/seed11",
                    "initial_state_sha256": f"state-{task}-{state}",
                    "initial_observation_sha256": f"observation-{task}-{state}", "instruction": task,
                    "success": False,
                }})
    return cfg, records


def test_demo_includes_all_three_state_zero_pairs_regardless_of_outcomes():
    cfg, records = example_records()
    records[1]["summary"]["success"] = True
    sections = predetermined_sections(cfg, records)
    assert len(sections) == 3
    assert [[record["summary"]["method"] for record in section] for section in sections] == [list(METHODS)] * 3
    assert all(record["summary"]["state"] == 0 for section in sections for record in section)
    assert [section[0]["summary"]["task"] for section in sections] == ["alpha", "beta", "gamma"]


def test_discordance_uses_first_pair_in_frozen_order_not_state_sort():
    cfg, records = example_records()
    for record in records:
        row = record["summary"]
        if row["task"] == "alpha" and row["state"] in (1, 2) and row["method"] == "dvac_cap_10":
            row["success"] = True
        if row["task"] == "beta" and row["state"] == 1 and row["method"] == "dvac_cap_50":
            row["success"] = True
    ordered = sorted(records, key=lambda record: (record["summary"]["state"] != 2, record["summary"]["episode_id"]))
    order = [record["summary"]["episode_id"] for record in ordered if record["summary"]["method"] != "fixed_10"]
    groups, selection, frozen_pairs = select_groups(cfg, records, order)
    assert selection[0]["qualifying_pairs"] == 2
    assert selection[0]["selected_pair_id"] == "suite/alpha/state2/seed11"
    assert selection[1]["selected_pair_id"] == "suite/beta/state1/seed11"
    assert len(groups) == 5
    assert len(frozen_pairs) == 9
    assert all(row["posthoc_outcome_selected"] for row in selection)


def test_missing_discordance_is_explicit_and_duplicate_example_is_not_repeated():
    cfg, records = example_records()
    for record in records:
        row = record["summary"]
        if row["task"] == "alpha" and row["state"] == 0 and row["method"] == "dvac_cap_10":
            row["success"] = True
    groups, selection, _ = select_groups(cfg, records, [record["summary"]["episode_id"] for record in records])
    assert len(groups) == 3
    assert len(groups[0]["selection_reasons"]) == 2
    assert selection[1]["selected_pair_id"] is None
    assert selection[1]["absence_reason"]


def test_review_selects_final_observations_and_gripper_transition_neighborhoods():
    def rows(length, transition):
        return [{"step": step, "action": [0] * 6 + [-1 if step < transition else 1]} for step in range(length)]
    selected, transitions = select_steps({"fixed_10": rows(10, 4), "dvac_cap_10": rows(7, 3)}, stride=5)
    assert selected == [0, 2, 3, 4, 5, 6, 7, 9, 10]
    assert transitions == {"fixed_10": [4], "dvac_cap_10": [3]}
    assert selected[-1] == 10
    with pytest.raises(ValueError, match="positive"):
        select_steps({"fixed_10": rows(10, 4)}, stride=0)
