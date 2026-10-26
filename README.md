# VLA: how often should a robot replan?

A controlled study of when a robot should obtain a new observation and regenerate
its actions. A pretrained **SmolVLA** policy, running through **LeRobot**, controls a
robot in **LIBERO / MuJoCo** from two camera images, robot state, and a language
instruction. No model training is performed.

**Research question:** How does replanning frequency change task success and
model-call cost, and does the preferred interval change when execution becomes
less predictable?

The project compares fixed execution intervals with denoising-variance adaptive
chunking (**DVAC**) under three conditions:

- **Normal execution:** 270 matched episodes across nine scheduling rules.
- **Moving target:** 150 episodes comparing five intervals under a static control
  and movement in four directions.
- **Random command dropout:** translation commands are suppressed with probability
  p, testing whether unreliable execution favors more frequent replanning.

The main finding is that **the useful interval depends on the task, execution
conditions, and the model-call budget**. Replanning after every action is expensive
and does not consistently produce the highest success rate.

**[Main comparison report](output/pdf/vla-capped-replanning-report.pdf)** ·
**[Comparison demo](outputs/capped-report/demo.mp4)** ·
**[Results and tables](outputs/capped-report/report.md)** ·
**[Protocol](docs/capped-protocol.md)** ·
**[Reproduction](docs/capped-reproduction.md)**

## How replanning works

```text
Two camera images + robot state + instruction
                        ↓
       SmolVLA generates H = 50 actions
       using 10 flow-matching iterations
                        ↓
        Execute the first K actions
                        ↓
        New observation → new prediction
```

An action controls translation, rotation, and the gripper. The generated horizon
**H=50 remains fixed**. Unexecuted actions are discarded when the robot replans.

- **Fixed intervals:** execute K = 1, 5, 10, 25, or 50 actions before replanning.
- **Capped DVAC:** select an interval from internal flow-matching variance, then
  apply `K = min(C, K_DVAC)`, with execution cap C = 5, 10, 25, or 50.

DVAC records the last five clean-action estimates `z_i = x_i - t_i v_i` within a
single model call. It measures their population variance for each future action,
summing over the seven valid normalized action coordinates. It selects the prefix
before the first variance exceeding `mean(W) + 2 * std(W)`, where W contains the
variance arrays from up to five preceding calls. The first call executes K=1 to
initialize this history; history resets every episode.

The cap limits execution only: every method still generates all 50 actions with
all ten flow-matching iterations. The robot executes the final sampled actions,
not the average intermediate estimate. The [protocol](docs/capped-protocol.md)
gives the equations, indexing, initialization, and implementation choices.

The simulator advances at 20 Hz and pauses during inference. K therefore determines
a simulated observation interval, not a wall-clock control rate. Changing K changes
both access to fresh observations and the frequency of stochastic action generation.

## Matched design

| Task | LIBERO suite | Initial states |
|---|---|---:|
| Pick up the alphabet soup can and place it in the basket | Object | 10 |
| Pick up the black bowl between the plate and ramekin and place it on the plate | Spatial | 10 |
| Open the middle drawer | Goal | 10 |

All nine methods use state indices 0–9 and policy seed 11: **30 task/state settings
per method**. The checkpoint, observation processing, simulator assets, action
normalization, precision, prediction horizon, solver, control frequency, and episode
limits are shared. Initial simulator states, observations, and complete first action
predictions match exactly across methods.

All 270 episodes passed trace and video verification. Computational comparisons
use **model calls**, averaged over successful and unsuccessful episodes. This
counts policy invocations, not the ten internal flow-matching steps per invocation.
It does not imply an equivalent wall-clock speedup.

## Normal execution: success versus model calls

Each method has ten episodes per task. Model calls are averaged over all 30 episodes;
failed episodes are included. Calls per 100 actions use the ratio of total calls to
total executed actions.

| Method | Soup | Bowl | Drawer | Completed | Mean model calls | Calls / 100 actions |
|---|---:|---:|---:|---:|---:|---:|
| Fixed K=1 | 6/10 | 8/10 | 9/10 | 23/30 | 162.0 | 100.00 |
| Fixed K=5 | 9/10 | 7/10 | 10/10 | 26/30 | 31.3 | 20.23 |
| Fixed K=10 | 10/10 | 4/10 | 10/10 | 24/30 | 15.5 | 10.22 |
| Fixed K=25 | 6/10 | 4/10 | 1/10 | 11/30 | 9.6 | 4.18 |
| Fixed K=50 | 3/10 | 3/10 | 3/10 | 9/30 | 5.1 | 2.13 |
| DVAC cap 5 | 10/10 | 7/10 | 10/10 | **27/30** | 40.2 | 29.24 |
| DVAC cap 10 | 10/10 | 7/10 | 8/10 | 25/30 | 26.4 | 17.53 |
| DVAC cap 25 | 10/10 | 8/10 | 7/10 | 25/30 | 14.1 | 8.97 |
| DVAC cap 50 | 9/10 | 7/10 | 9/10 | 25/30 | 9.5 | 6.86 |

![Completion and model-call trade-off](outputs/capped-report/success_calls_frontier.png)

**A smaller cap did not consistently improve completion.** Caps 10, 25, and 50 each
completed 25/30 episodes, while their mean call counts were 26.4, 14.1, and 9.5.
Cap 5 had the highest observed count, 27/30, at 40.2 calls per episode. Compared with
fixed K=5, it added one successful episode and 8.9 calls per episode; the paired
completion difference was +3.3 percentage points, with a descriptive 95% interval
of [0.0, +10.0]. This sample does not establish a universal best method.

The two designated comparisons show why both outcomes and cost matter:

- **DVAC cap 10 versus fixed K=10:** +3.3 percentage points in completion
  (95% interval [−10.0, +16.7]), with 10.9 additional calls per episode. DVAC alone
  succeeded in three settings; fixed K=10 alone succeeded in two.
- **DVAC cap 10 versus cap 50:** no aggregate completion difference
  (95% interval [−16.7, +16.7]), with 16.9 additional calls per episode. Each method
  succeeded in three settings where the other failed. Equal counts do not mean
  identical behavior or equivalent reliability.

Intervals use **10,000 paired bootstrap samples of initial states within each task**,
keeping all methods together and weighting tasks equally. They are descriptive,
not multiplicity-adjusted confirmatory tests. The [full results](outputs/capped-report/report.md)
are accompanied by [all 36 paired comparisons](outputs/capped-report/paired_contrasts.csv),
task-level tables, and every individual outcome.

## Behavior and demonstration

The [demo](outputs/capped-report/demo.mp4) shows **fixed K=10, DVAC cap 10, and DVAC
cap 50** side by side for state 0 / seed 11 on all three tasks. These examples were
selected independently of their outcomes. The instruction, current K, model-call
count, replanning moments, and outcome remain visible. Panels advance together in
simulation time; ended episodes hold their final observation.

Adaptive behavior is not described adequately by an average interval. The mean
selected K was 3.45, 5.87, 11.92, and 16.09 for caps 5, 10, 25, and 50. After
initialization, approximately 34–43% of calls still selected K=1. The actual executed
prefix can be shorter than selected K when an episode ends.

![Selected and executed adaptive intervals](outputs/capped-report/cap_horizon_behavior.png)

The [review materials](outputs/capped-report/review/review.json) also include the
first cap-10/cap-50 disagreement in each direction under the frozen collection order.
These are explicitly outcome-selected illustrations. The full report separates
observed behavior from hypotheses about failure causes. A gripper command, action
jump, or contact is not automatically a grasp error or a successful correction.
Qualitative annotations are assistant-generated without independent human review.

## What the comparison establishes

The experiment measures a completion/computation trade-off for this checkpoint and
these initial situations. It tests the **complete DVAC scheduling rule and its
execution cap**. It does not isolate the usefulness of denoising variance itself:
establishing that contribution would require a control schedule matched on model-call
budget or execution-interval distribution.

Only three tasks and one policy seed are represented in the main comparison. The
state outcomes were already examined while developing the experiment, so this is
an exploratory comparison, not a held-out confirmation. Internal FM stability does
not guarantee a correct action. The results do not isolate fresh observations from
stochastic regeneration, establish equivalence from an interval containing zero, or
demonstrate real-time control of a physical robot. LIBERO success follows its native
predicate and may occur before release; post-release stability is not evaluated.

## Moving target: does motion favor shorter intervals?

The presentation's moving-target experiment uses the soup-can task with **three
held-out robot states and two initial can positions**. Each of these six scenarios
is evaluated with K = 1, 5, 10, 25, 50 under five conditions: a free static control
and motion along +X, -X, +Y, or -Y. This gives **6 × 5 × 5 = 150 episodes**.

The commanded target speed is 1.5 cm/s; measured free-motion speed is approximately
0.91 cm/s. Motion starts smoothly, and the external drive stops when bilateral
finger contact occurs. Parameters are fixed before evaluation on the held-out
states. X and Y are perpendicular directions along the tabletop.

| Target condition | K=1 | K=5 | K=10 | K=25 | K=50 |
|---|---:|---:|---:|---:|---:|
| Free static control | 6/6 | 6/6 | 6/6 | 5/6 | 5/6 |
| +X | 3/6 | 5/6 | 3/6 | 3/6 | 3/6 |
| -X | 6/6 | 5/6 | 6/6 | 5/6 | 4/6 |
| +Y | 3/6 | 2/6 | 2/6 | 3/6 | 2/6 |
| -Y | 4/6 | 6/6 | 5/6 | 5/6 | 4/6 |

These counts reproduce the moving-target chart in the final defense presentation.
**Direction matters:** K=5 performs better than K=1 for +X, while K=1 and K=10 both
complete all six -X cases. Target motion does not create a universal advantage for
K=1. With only six scenarios per condition, these are descriptive findings.

The presentation demo aligns the five intervals in simulation time, shows model-call
counters, and holds the final frame after each episode ends. Video playback speed
must not be interpreted as inference speed.

## Random command dropout: does unreliable execution favor shorter intervals?

At each control step, with probability **p**, the commanded XYZ translation is set
to zero. Rotation and gripper commands remain unchanged, and simulation time still
advances. The policy receives no explicit dropout notification: it sees the resulting
state at its next observation.

The comparison uses the same three tasks and 30 starts per method, with shared
dropout masks at corresponding control steps. The presentation shows p=0.10 and
p=0.30 against the p=0 reference; p=0.20 is left pending. The tested rules are fixed
K=1, 5, 10, 25, 50 and DVAC caps C=5, 10, 25, 50. C=1 shares the K=1 baseline;
it is not a separate run. Mean calls include all episodes, including failures.

Measured findings highlighted in the defense are:

- At **p=0.30**, K=1 has the highest observed completion count: **23/30**, with
  **187.8 calls** per episode. K=10 completes **19/30** with **21.9 calls**.
  The higher observed success therefore costs about **8.6 times as many calls**.
- Averaged over limits 5, 10, 25, and 50, success decreases by **36.7 percentage
  points for DVAC**, versus **16.7 points for fixed intervals**, from p=0 to p=0.30.
  This describes the tested settings, not a general guarantee about either method.

Some dropout summaries used in the presentation contain manually adjusted
**illustrative data**, including the K=1 value at p=0.10. Those values are not
experimental evidence and are not used to support the measured findings above.

## Conclusions and limitations

- **Frequent replanning is not always better.** Under normal execution, fixed K=5
  completes 26/30 episodes with 31.3 calls, compared with 23/30 and 162.0 calls for K=1.
- **Adaptive scheduling can offer a useful trade-off.** DVAC C=50 completes 25/30
  with 9.5 calls. The experiment evaluates the complete scheduling rule; it does not
  prove that FM variance identifies the optimal moment to replan.
- **Execution conditions change the trade-off.** K=1 has the highest observed
  success at p=0.30, at a substantial call cost. DVAC loses more success on average
  under dropout, while moving-target results depend strongly on motion direction.

The study covers one pretrained policy in simulation, three main tasks, and a
limited set of intervals. Small differences in completion counts do not establish
superiority. FM disagreement is a heuristic, not a calibrated failure probability.
The disturbances model only selected execution problems, and no real-robot
validation is performed. The simulator pauses for inference, so these experiments
do not establish real-time deployment performance.

## Additional archived experiments

Earlier experiments are retained as separate evidence, not pooled into the
presentation's main comparisons:

- [Uncapped DVAC report](output/pdf/vla-dvac-report.pdf): 48 episodes using states
  15 and 16, seeds 11 and 22, and four scheduling rules.
- [Earlier moving-target demo](outputs/moving-target-report/demo.mp4) and
  [protocol](docs/moving-target-protocol.md): 135 episodes with oscillating force
  references. This is a different experiment from the 150-episode directional test.
- [Original fixed-interval report](output/pdf/vla-replanning-report.pdf) and
  [paired videos](outputs/expanded/demo.mp4).

## Evidence and reproduction

The [reproduction guide](docs/capped-reproduction.md) covers compatibility checks,
collection, tests, verification, and rebuilding the report and demo. The
[verification record](outputs/capped-report/verification.json) checks all predicted
and executed actions, FM variances, rolling thresholds, caps, sampling seeds, matching
starts, source hashes, and video alignment. The
[statistics](outputs/capped-report/analysis.json) and
[artifact hashes](outputs/capped-report/artifacts.json) bind the tables and figures to
the verified evidence. All 30 matched blocks are visible in the
[outcome matrix](outputs/capped-report/outcome_matrix.csv).

The implementation is in `vla_replan/`, `adaptive_replan/`, `capped_replan/`,
`moving_target/`, and `action_dropout/`;
configurations are in `configs/` and analysis tools in `scripts/`. Model weights and
virtual environments are excluded from the repository.

## Sources

- [SmolVLA paper](https://arxiv.org/abs/2506.01844).
- [DVAC: Denoising Tells When to Replan](https://arxiv.org/html/2606.03847v1).
- [Pinned SmolVLA LIBERO checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102).
- [SmolVLA implementation in LeRobot](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py).
- [LIBERO documentation](https://huggingface.co/docs/lerobot/en/libero).

The project's contribution is the explicit execution-prefix control, the DVAC transfer
and cap intervention, paired comparisons, independent evidence verification, and
behavioral analysis. It is not a reproduction of the DVAC paper's complete benchmark.
The pretrained policy and simulator are third-party components with their own licenses.
