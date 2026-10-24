import json

import pytest

from scripts.analyze_research import command_diagnostics, deferred_transitions, load_audited


def test_deferred_transition_requires_observed_suffix_and_actual_absence():
    actions = [[0, 0, 0, 0, 0, 0, -1] for _ in range(50)]
    actions[3][6] = 1
    chunks = [{"step": 10, "actions": actions}]
    steps = [{"step": i, "action": [0, 0, 0, 0, 0, 0, -1]} for i in range(10, 14)]
    assert deferred_transitions(steps, chunks) == {
        "planned_transitions": 1, "evaluable_transitions": 1, "deferred_transitions": 1}
    steps[2]["action"][6] = 1
    assert deferred_transitions(steps, chunks)["deferred_transitions"] == 0
    assert deferred_transitions(steps[:2], chunks)["evaluable_transitions"] == 0


def test_transition_after_nine_steps_is_outside_diagnostic_window():
    actions = [[0, 0, 0, 0, 0, 0, -1] for _ in range(50)]
    actions[10][6] = 1
    assert deferred_transitions([], [{"step": 0, "actions": actions}])["planned_transitions"] == 0


def test_incomplete_study_requires_explicit_preview(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"scheduled_jobs": 1}))
    (tmp_path / "status.json").write_text(json.dumps({"phase": "running"}))
    with pytest.raises(ValueError, match="incomplete"):
        load_audited(tmp_path)
    _, trials, skipped = load_audited(tmp_path, allow_incomplete=True)
    assert trials == skipped == []


def test_revision_compares_the_old_plan_at_the_new_absolute_step():
    old = [[0, 0, 0, 0, 0, 0, -1] for _ in range(50)]
    old[10] = [1, 0, 0, 0, 0, 0, 1]
    new = [[.5, 0, 0, 0, 0, 0, -1] for _ in range(50)]
    d = command_diagnostics([], [{"step": 80, "actions": old}, {"step": 90, "actions": new}])
    assert d["mean_plan_revision_translation"] == .5
    assert d["gripper_plan_disagreements"] == d["comparable_replans"] == 1
    assert command_diagnostics([], [{"step": 0, "actions": old}, {"step": 50, "actions": new}])["comparable_replans"] == 0
