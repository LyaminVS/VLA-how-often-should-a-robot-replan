# Report and oral presentation guide

Outline for the final report and presentation. The main experiment was completed on
17 October 2026: 30 real episodes, 3 tasks, 2 initial states, and 5 values of K.
Numerical sources are `outputs/compact/episodes/*/summary.json` and
`outputs/compact/report/statistics.json`; the technical smoke run is excluded.

## Observed results

| K | LIBERO successes | Mean model calls | Mean compute time, s | Mean episode wall time, s |
|---|---:|---:|---:|---:|
| 1 | 2/6 | 228.2 | 180.6 | 184.6 |
| 5 | 5/6 | 34.2 | 30.1 | 35.0 |
| 10 | 4/6 | 17.8 | 17.8 | 21.5 |
| 25 | 2/6 | 10.5 | 13.6 | 18.5 |
| 50 | 2/6 | 5.2 | 9.6 | 14.0 |

Totals: 15/30 successes, 1775 model calls, and 6409 steps; the sum of `episode_wall_s`
is 27.35 minutes. This is the sum of measured episode durations, not installation time
or the duration of the entire process. `compute_s` includes preprocessing, inference,
and postprocessing with GPU synchronization, plus `env.step` with rendering.
`episode_wall_s` additionally includes reset, frame recording, and video finalization,
but excludes video-object creation before the timer, JSON log writing, model loading,
and warmup. Simulated motion time, `steps / 20`, is a separate quantity.

| Task | K=1 | K=5 | K=10 | K=25 | K=50 |
|---|---:|---:|---:|---:|---:|
| Soup can into the basket | 0/2 | 2/2 | 2/2 | 1/2 | 1/2 |
| Black bowl onto the plate | 1/2 | 1/2 | 0/2 | 0/2 | 1/2 |
| Open the middle drawer | 1/2 | 2/2 | 2/2 | 1/2 | 0/2 |

Across these six initial situations, K=5 achieved the most successes, while K=10 required
fewer calls and less time than K=5. The most frequent updates, K=1, were not best.
Bowl outcomes illustrate task dependence: K=50 succeeded in one state, whereas K=10
succeeded in neither. Two states per task cannot establish a universal optimum or
reliably rank closely performing variants. Intervals in `statistics.json` are descriptive
and based on a very small sample; zero width does not establish certainty or equivalence.

Success is the standard LIBERO signal, which ends the episode. It does not guarantee
that the can has been released and rests stably in the basket; this is assessed separately
from the video. Do not replace the environment's binary result with a subjective judgment
or describe it as proof that object release has completed.

Two examples to present are the drawer task in state0 with K=1 versus K=10, and the same
task and state with K=25. In the first comparison, K=1 remains near the handle for a long
time and only partially opens the drawer near the end, timing out at step 300; K=10
succeeds at step 113. With K=25, the first pulling motion at steps 105–135 does not move
the drawer. After another approach at steps 140–200, opening succeeds at step 228.
Plan revision at steps 125/150 is 0.268070/0.486133 in command units. These are observations
from videos and logs; an explanation based on feedback is a hypothesis, not established causality.

## Report structure

1. Research question, hypotheses, and the role of feedback.
2. VLA inputs, VLM and action expert, flow matching, and the distinctions between H, K,
   control frequency, and generation iterations.
3. Model, revisions, hardware, versions, dtype, tasks, states and seeds, step limits,
   warmup, and run order.
4. Metrics, statistical unit of analysis, and definitions of grasping errors.
5. Overall table, figures, task-specific results, and paired differences from K=10 with intervals.
6. Annotated errors with review coverage, and two or three paired videos with exact step references.
7. Cases where frequent updates help and where additional calls are not justified.
   Separate observations from proposed explanations. Do not declare a universal best K from three tasks.
8. Limitations: number of states and tasks, one model, stochasticity, annotation, and synchronous simulation.
9. Reproduction commands, manifest, logs, videos, and sources.

## Coverage of the project requirements

| Requirement | Evidence |
|---|---|
| Pretrained VLA and compatible simulator | Pinned SmolVLA LIBERO, LeRobot, and hf-libero |
| Explanation of action sequence prediction | H/K diagram, flow-matching description, and chunk logs |
| Configurable number of executed actions | Intervals and explicit `run_episode` execution |
| Tasks and initial situations | Three tasks, state IDs 0/1, fixed policy seed 11 |
| Task success | Standard success signal, task-specific results, and macro average |
| Corrections | Motion traces, old and new plans, videos, and interpretation |
| Grasping errors | Visual annotation table, rules, and coverage |
| Calls and runtime | Actual calls, GPU synchronization, and separate timing metrics |
| Controlled comparison | Versions, revisions, hashes, paired states, and shuffled order |
| Demonstration | Instruction, cameras, REPLAN, outcome, and paired K videos |
| Cases without improvement | Specific paired episodes and limitations of the conclusions |

## A 7–10 minute presentation

- 1 minute: research question, H and K, and tasks.
- 2 minutes: experimental design and control of initial conditions.
- 2 minutes: success, computational cost, and paired differences.
- 2 minutes: videos, errors, and corrections.
- 1 minute: conclusions, trade-offs, and limitations.

Prepare to explain why K=1 can be worse, why one seed is insufficient for broad conclusions,
why intervals are reported, why training is unnecessary, why pausing during inference is
not real-time control, why finger closure does not imply a successful grasp, and how
changes in the initial scene are ruled out.

## Short answers for the presentation

**What does the model receive?** Two images, from the external and wrist cameras,
the robot state, and a text instruction. The checkpoint's processors and normalization
are used.

**What does it predict?** A sequence of H=50 actions. Each action contains three
translation components, three rotation components, and a gripper command. The complete
predicted chunk is saved in `chunks.jsonl`; executed actions are saved in `steps.jsonl`.

**What does flow matching do?** The action expert starts from a noisy sequence and
transforms it into actions conditioned on the observation and instruction over a fixed
10 computational iterations. These iterations are not simulation steps. No training
is performed in this project.

**How does K differ from H?** H is the number of predicted actions and is always 50.
K is the number of initial actions executed before a new observation. The remainder of
the old plan is replaced. K=50 also replans after 50 actions if the task has not ended.

**What makes the comparison controlled?** Identical environment states are restored for
all K values; both states and initial observations are hashed. The model, input processing,
precision, cameras, step limits, control frequency, generation settings, and hardware
remain fixed. Block order and K order are shuffled with a fixed seed.

**Why 30 episodes?** The design is 3 tasks × 2 states × 5 values of K. There are only six
independent initial situations, making this a compact behavioral study rather than a
precise estimate of performance across LIBERO. Five technical checks are not added to
the main statistics.

**Why might frequent replanning be detrimental?** It interrupts a coherent action
sequence more often. A new stochastic prediction may change motion or gripper commands.
This is a possible mechanism. For an individual failure, present the video and log
observations first, then state the proposed explanation as a hypothesis.

**How does a trajectory correction differ from a command jump?** A difference between
old and new plans measures action revision, not its usefulness. A correction is assessed
against a visible deviation, a new observation, and subsequent movement toward the goal;
temporal coincidence alone does not establish the cause.

**Why does video duration differ from computation time?** The video plays simulated
motion at 20 Hz. The simulator pauses while the GPU computes a new chunk. Thus, 10 seconds
of motion can require several minutes of elapsed computation at K=1.

**Can the system control a physical robot at 20 Hz?** This was not tested. The measured
call latency of approximately 0.8 s is much longer than 0.05 s per control step. A physical
robot would require a different execution and latency-handling architecture and separate validation.

**Can one K be declared best?** The best observed outcome in these six situations and
the cost of each K can be reported. Generalization to other states, tasks, seeds, and
models requires more data. Identical outcomes in two states do not imply an absence of risk.

**What is the project's original contribution?** The explicit K-step executor, a controlled
paired protocol, reproducible configurations and manifest, timing metrics, videos marked
with replanning events, comparisons, and visual error analysis. The model and simulator
are existing third-party components.
