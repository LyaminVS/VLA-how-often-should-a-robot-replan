# Experimental method and metric definitions

The study compares five replanning intervals using 150 unique episodes:
**3 tasks × 10 initial states (0–9) × 1 policy seed (11) × 5 values of K**.
All results describe the complete study; technical smoke episodes are excluded.

## Research question and hypotheses

How does the number of executed actions K affect the success rate and computational cost
of a single model?

1. A small K may help with gripper approach, contact, and correction of deviations.
2. During predictable motion, a larger K may preserve success with fewer model calls.
3. K=1 is not necessarily best: stochastic replanning may change direction, and the model
   was trained to generate sequences. This is a hypothesis rather than a predetermined conclusion.

A null or non-monotonic result also answers the research question. All completed
episodes in the selected grid are included, rather than only favorable tasks or
outcomes. The three tasks are placing the alphabet soup can in the basket, placing
the black bowl between the plate and ramekin on the plate, and opening the middle
drawer. Each task has ten matched initial situations, evaluated at K=1/5/10/25/50.

## Controlled conditions

Only K varies. The following remain fixed: H=50, 10 flow-matching steps, 20 Hz control,
two 256×256 camera images before the standard resize to 512×512, checkpoint normalization,
relative end-effector control, dtype, one GPU, batch=1, and recording settings.
Asynchronous inference, real-time chunking (RTC), and chunk averaging are not used.

The policy receives a new observation at its next invocation. The environment produces
images at every step for logging, but these do not change the selected actions. After
K steps, the remaining plan is discarded. Repeating one action K times or varying H
alongside K would constitute different experiments.

Each K uses the same saved state and the same number of settling steps. The environment
seed depends only on the task and state. A policy seed identifies a separate repeat.
Generation noise is determined by `(policy_seed, simulation_step)`; replanning at a
shared simulation step uses the same noise, although observations and predictions may differ.

The order of K values is shuffled within paired blocks, and the blocks are also shuffled
using a fixed seed. Two warmup calls precede measurement. The environment, policy, and
processors are reset before each episode. Checksums verify identical initial states
and observations, including camera images.

## Metrics

| Metric | Definition | Limitation |
|---|---|---|
| Success | Standard `check_success` before the step limit | Uses the LIBERO criterion |
| Calls | Actual generations, `ceil(steps/K)` | Also depends on the completed episode length |
| Sim time | Actions / 20 Hz | Not elapsed real time |
| Inference time | Preprocessing + model + postprocessing, with GPU synchronization | Excludes downloads and warmup |
| Simulation cost | Wall time of `env.step`, including images | Depends on the CPU and renderer |
| Compute time | Inference + simulation cost | Excludes reset and video encoding |
| Episode wall time | Reset + rollout + video finalization | Excludes model loading and JSON writing |
| EEF path | Cumulative gripper path length in metres | An early failure can have a short path |
| Plan revision | New first translation vector versus the next vector of the previous plan | Command units; no remaining plan at K=50 |
| Action jump | Difference between consecutive translation commands | Does not establish that a correction is useful |
| Gripper close commands | Command transitions from ≤0 to >0 | Not a count of grasping errors |
| Grasp errors | Visual annotation of videos | Report coverage and ambiguities |

An episode ends immediately after the standard success signal. In the basket task,
this may occur while the can is still held by the fingers inside the target region.
Object release and stability after termination are not additional checks in this protocol.

The simulator waits for the model. The experiment examines the interval between feedback
updates, not the stability of a physical robot under latency. Success at K=1 does not
establish that an RTX 2060 can control a real robot at 20 Hz. That would require a model
call under 50 ms or a separate control architecture.

## Statistical analysis

The comparison unit is a task and initial state. Average across seeds within each state,
then across states within each task, and finally across tasks with equal weights.
Bootstrap: 5000 resamples of states separately within each task, using identical sampled
indices for every K. The primary difference is relative to K=10. Multiple seeds for one
state do not increase the number of independent initial situations.

The 95% percentile intervals are conditional on the selected tasks, seeds, and checkpoint.
They do not cover all possible tasks or randomness in model training. There is no
adjustment for multiple comparisons; the conclusions are exploratory. Overlap between
two separate intervals does not test the paired difference. An inconclusive difference
does not establish equivalence. Small samples and identical outcomes can produce narrow
or zero-width bootstrap intervals without establishing error-free performance.

Incomplete runs are not analyzed by default. `--allow-incomplete` includes only states
with every K and seed present, and marks the report as provisional. Source evidence
uses two disjoint directories, `outputs/compact` and `outputs/validation`; their raw
manifests remain unchanged. The unified analysis rejects overlapping episode identities
and incompatible conditions. Alternative run configurations and smoke results must
not be added as independent observations when their states overlap.

## Visual annotation

The visual review covers 60 episodes: states 0–3 for every task and every K, with
seed 11. It is balanced across conditions and includes successful and unsuccessful
episodes. All 150 episodes contribute to numerical success and timing results.

- `reviewed`: yes when both event counters are resolved; no indicates unresolved
  counters or an uninspected episode, not an absence of errors.
- `missed_grasps`: distinct attempts to close the fingers on the target object or handle
  without securing it. The absence of an attempt is not automatically a missed grasp.
- `dropped_objects`: loss of a held object before placement. Intentional release is not an error.
- `correction_after_replan`: yes/no/unclear. For yes, record the steps and visible correction
  in notes and cross-check `chunks.jsonl`. Temporal coincidence alone does not establish causality.
- `failure_category`: none, missed_grasp, dropped_object, wrong_object, placement, collision,
  timeout, other, unclear. Success after an intermediate miss has category=none and missed_grasps>0.

If an event count is ambiguous, retain reviewed=no and explain why in the notes.
`inspection.json` separately records which episodes were actually inspected. The
assistant reviewed the 60 selected episodes using both cameras and the logs; there
was no independent human review. Known individual counters are retained even if the
other counter is uncertain. The PDF reports known events with a separate denominator
for each counter; these are not error-frequency estimates for the entire experiment.
The original CLI statistical table uses only fully determined labels; the full-study
analysis additionally reports per-counter coverage.

Contact sheets include a frame every 5 steps, frames around changes in the sign of the
gripper command, and the final frame; ambiguous moments were additionally inspected frame
by frame. This review can miss brief or occluded contact, so uncertainty is not replaced
with zero. For the five object/state0 episodes, visual observations from the technical run
were reused after verifying identical actions and every decoded video frame against the
main episodes. Evidence is stored in `review-object-state0-equivalence.json`; numerical
measurements from the technical run were not added to the main statistics.

## Demonstration and future work

Show paired cases in which frequent updates help, the same outcome is achieved at lower
cost, and frequent updates do not help or make the outcome worse, if such cases occur.
State explicitly if a requested pattern is absent. Present the overall results before
the videos and retain the IDs of the selected episodes.

An additional experiment with perturbations or adaptive K may be useful after the main
series. Define the intervention or heuristic in advance, tune it on other states, and
evaluate it on held-out states. This requires a separate protocol; the current
implementation compares fixed values of K.
