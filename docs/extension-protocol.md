# Prospective extension to the replanning study

Specified on 17 October 2026, before launching or inspecting the new episodes.
The original 30-episode compact experiment is retained as an exploratory pilot.
Its observed outcomes motivated a larger sample, but no task, K value, model,
episode horizon, control setting, or policy seed is changed for the extension.

## Fixed sample and stopping rule

- Configuration: `configs/validation.json`.
- Configuration SHA-256 fingerprint: `16689960309feaa035e8722bda9c7bd074d9227766464c7123ab1c64c8f2ec3e`.
- Three tasks: soup can into basket; black bowl between the plate and ramekin onto
  the plate; opening the middle cabinet drawer.
- Initial-state indices 2 through 9, inclusive, for every task.
- K = 1, 5, 10, 25, 50; H = 50; policy seed 11.
- 3 tasks x 8 states x 1 seed x 5 K values = 120 new episodes.
- Complete every planned episode irrespective of intermediate success rates.
  Resume only interrupted episodes under the existing provenance checks.
- Use the same RTX 2060, software, frozen rollout code, camera/recording settings,
  precision, model revisions, assets, episode horizons and settling procedure.
  Verify both package/asset metadata and source hashes against the pilot before
  launch. Preserve the original manifests and episode records without rewriting.
- As in the pilot, shuffle all task/state blocks and the K values within each
  block using the fixed order seed 20261017.

These state indices were not part of the project's pilot; they are not claimed
to be unseen by the pretrained checkpoint. State indices and tasks are fixed
benchmark choices, not a random sample of all manipulation tasks.

## Analysis fixed before collection

The primary extension results use only the 120 new validation episodes. Report
the original 30 pilot episodes separately and the 150 unique episodes together
as a descriptive aggregate. Never count a repeated task/state/seed/K identity
twice, and never rewrite a source configuration fingerprint to pool records.

Retain the original metrics and equal task weighting in `docs/protocol.md`.
For each phase, resample initial states separately within each task, keeping the
same sampled states for every K (5000 bootstrap draws, seed 20261017). Keep K=10
as the reference for paired differences. Display success, model calls, inference
plus simulation time, episode wall time, and motion metrics. Report per-task
outcomes and paired differences with 95% percentile intervals. Do not declare
equivalence from an inconclusive difference or select a new reference after
seeing the results. Intervals remain exploratory, conditional on these tasks
and the one policy seed, without adjustment for multiple comparisons.

The combined estimate weights all ten states equally within each task; it is
not an unweighted average of the 30- and 120-episode phase percentages. The
pilot influenced the decision to expand, so validation and pooled results must
remain distinguishable. Timing comparisons across collection sessions require
matching hardware/software and should acknowledge possible system-load drift.

## Visual review and demonstrations

Before collection, select validation states 2 and 3 for all three tasks and all
five K values: 30 new videos. This balanced subset is used for grasp/error
annotation, alongside the existing 30 pilot annotations. Numerical success and
runtime analysis uses all 120 validation episodes. Do not substitute only failed
videos or favorable examples for the predefined annotation subset.

Use both recorded cameras, frames every five actions, neighborhoods of gripper
command sign changes, and the final outcome frame. Inspect ambiguous events in
denser windows when possible. Keep unknown event counts unknown; `reviewed=no`
does not mean zero errors. Report the number selected, inspected and fully
resolved separately, with per-event denominators for partially known labels.

Select illustrative paired demos after viewing this predefined subset. Identify
their phase, task/state/seed/K and selection rationale; they illustrate possible
mechanisms and are not evidence for event frequencies across the full study.
Show the complete validation outcome table alongside examples, including cases
that contradict any simple "more frequent is better" claim if they occur.

## Deliverables

Keep raw evidence under `outputs/compact` and `outputs/validation`. Save combined
analysis and provenance under `outputs/expanded`, with no fabricated merged run
manifest. Update the English README, report, charts, demo and submission archive
after all 120 episodes and the predefined review are complete. Preserve the
original pilot report for comparison. No additional policy seeds, perturbations
or adaptive-K policies are included in this extension.
