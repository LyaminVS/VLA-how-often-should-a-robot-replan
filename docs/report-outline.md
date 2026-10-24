# Results and oral presentation guide

The project contains **150 standard-task episodes: 3 tasks × 10 initial states ×
5 replanning intervals**, and a **135-episode controlled moving-target experiment**.
The complete report is `output/pdf/vla-replanning-complete-report.pdf`. Standard-task
statistics are in `outputs/expanded/report/statistics.json`; moving-target statistics
are in `outputs/moving-target-report/statistics.json`. Their outcomes are analyzed
separately. Calibration and integration checks are excluded from both result sets.

## Standard-task result to explain

K=5 succeeded in 26 of the 30 task/state situations, K=10 in 24, and K=1 in 23.
K=5 used 4.7 times less mean compute time than K=1; K=10 used 44% less compute than
K=5. K=25 and K=50 had substantially fewer successes in this fixed sample, but some
individual states favored a longer interval. The results support a trade-off
between cost and task completion, not a universal optimum or a claim that K=1 is
intrinsically worse.

| K | Success | Mean calls | Mean compute, s | Mean episode wall time, s |
|---:|---:|---:|---:|---:|
| 1 | 23/30 (76.7%) | 162.0 | 128.8 | 132.5 |
| 5 | 26/30 (86.7%) | 31.3 | 27.7 | 31.7 |
| 10 | 24/30 (80.0%) | 15.5 | 15.5 | 19.0 |
| 25 | 11/30 (36.7%) | 9.6 | 12.5 | 16.9 |
| 50 | 9/30 (30.0%) | 5.1 | 9.3 | 14.0 |

| Task | K=1 | K=5 | K=10 | K=25 | K=50 |
|---|---:|---:|---:|---:|---:|
| Soup can into the basket | 6/10 | 9/10 | 10/10 | 6/10 | 3/10 |
| Black bowl onto the plate | 8/10 | 7/10 | 4/10 | 4/10 | 3/10 |
| Open the middle drawer | 9/10 | 10/10 | 10/10 | 1/10 | 3/10 |

The study totals 93/150 successes, 6,707 model calls and 28,140 steps. Summed
episode wall time is 107.1 minutes. This is measured episode duration, not
installation or total project work. Compute time includes preprocessing,
synchronized inference and postprocessing, plus environment stepping with rendering.
Episode wall time also includes reset, frame recording and video finalization; it
excludes model loading, warmup, JSON log writing and video-object creation before
the timer. Motion time is steps/20.

## Experimental method

Each K uses the same ten initial-state indices (0–9) per task and seed 11. State
and observation hashes verify identical initial conditions. All 150 episodes use
the same checkpoint, code, dependencies, action horizon, flow iterations, cameras,
precision, step limits and RTX 2060 hardware. Task/state blocks and K order are
shuffled with a fixed seed. All 150 unique episodes are included in the reported
analysis.

## Uncertainty and matched comparisons

The unit of comparison is a task/state situation evaluated at every K. There are
**30 paired situations**, not 150 independent measurements of each K. Bootstrap
analysis resamples states within each task, retaining the same selected states
across all K. It uses 5,000 draws, seed 20261017, equal task weighting and reference
K=10.

| K versus K=10 | Success difference, percentage points | Exploratory 95% interval |
|---:|---:|---:|
| 1 | −3.3 | [−20.0, +13.3] |
| 5 | +6.7 | [−10.0, +20.0] |
| 25 | −43.3 | [−60.0, −26.7] |
| 50 | −50.0 | [−70.0, −30.0] |

Intervals for the shorter intervals include zero, so a reliable ranking is not
established. This also does not demonstrate equivalence. Intervals are conditional
on three fixed tasks and one policy seed, with no multiple-comparison adjustment.
Ten states per task remain a small sample. A 10/10 result can produce a zero-width
within-task bootstrap interval without establishing error-free performance.

## Three evidence-based video cases

The demo uses state 3, seed 11, with complete paired recordings. Examples were
selected after inspecting the balanced visual subset to illustrate behavior,
not to estimate how often each error occurs.

1. **Can: K=10 versus K=25.** K=10 reaches success at step 139. K=25 visibly holds
   the can through step 81, loses it at about 82–86 and continues toward the basket
   after replans at 100 and 125. The can is not recovered before timeout at 280.
   Positive gripper commands throughout dense frames 76–94 distinguish the loss
   from an intentional release. Slippage/contact geometry is a possible explanation;
   these recordings do not identify the mechanical cause or prove K caused it.
2. **Bowl: K=1 versus K=50.** K=50 withdraws at 80–100, returns at 105–125 after
   replan 100, transports the bowl at 130–160, then opens near the plate rim around
   160. Placement remains incomplete at 280. K=1 succeeds at 80 using 80 calls;
   K=10 also succeeds at 75 with eight calls. A visible redirection is not proof
   of an effective correction or evidence that the extra K=1 calls were necessary.
3. **Drawer: K=5 versus K=50.** Both succeed. K=5 remains near the handle around
   90–215 with little opening, finishing at 252 after 51 calls. K=50 pulls around
   100–127, finishing at 127 after three calls. Compute time is 44.7 versus 4.9
   seconds. Sustained pulling is a possible mechanism; contact forces were not
   recorded, so the difference cannot be attributed uniquely to replanning.

In state 2, drawer K=25 and K=50 instead open the upper drawer. A wrong target
must not be relabeled as a missed closing grasp. Open-finger hooking can open a
drawer without a conventional closed-finger grasp.

## Visual annotation scope

The inspected subset contains **60 videos**: states 0–3 for all tasks and all K
values, giving 12 inspected episodes per K. Both cameras, every fifth action frame,
gripper sign-change neighborhoods, the last action and final outcome are examined,
with dense contact windows where necessary. Numerical success and runtime results
use all 150 episodes.

Inspection and fully resolved event labels are separate. Unknown miss/drop counts
remain blank, never zero. Both counters are resolved in 41 of the 60 inspected
episodes. A missed-grasp count is known in 41 episodes (one known event), and a
dropped-object count in 42 episodes (two known events). These selected, partially
resolved counts are not error rates for the full study. Use the per-K coverage
table in the final PDF and the structured annotation statistics for denominators.

A command to close the fingers is neither a successful grasp nor an error by
itself. An intentional release into the target is not a drop. A change between
action plans is a command metric, not proof of useful motion correction. Annotation
was performed by the assistant without independent human raters.

## Moving-target results

The controlled experiment contains 135 episodes: five new initial states × three
policy seeds × three K values × three conditions. There are 15 paired state/seed
repeats per condition/K cell. Static controls use the same instrumentation with zero
external force; the checkpoint, instruction, initial scene and execution rule match.

| Condition | K=5 | K=10 | K=50 |
|---|---:|---:|---:|
| Static | 12/15 | 13/15 | 6/15 |
| Slow reference | 8/15 | 13/15 | 5/15 |
| Fast reference | 13/15 | 15/15 | 6/15 |

The study uses 2,622 model calls and 25,703 executed actions. Summed episode wall
time is 54.4 minutes; the collection service ran for 55.0 minutes including setup.
At fast K=10, mean model calls are 14.6 and mean compute is 14.7 seconds per episode.
The 15/15 success count is a result in these selected scenes and seeds, not a claim
of error-free deployment.

Each moving condition is compared with its static control at the same K, retaining
state/seed pairing. The condition-by-K contrast then subtracts the motion-related
change at K=10. The 5,000-draw bootstrap resamples five complete initial-state
clusters, retaining every seed and treatment. It does not treat 15 seed repeats as
15 independent scenes or estimate uncertainty over untested policy seeds.

Paired success differences are in percentage points (pp); positive values favor the
moving condition. Each cell contains 15 outcomes from the same five states and three
seeds. The intervals below are exploratory 95% state-cluster bootstrap CIs
(bootstrap seed 20261024).

| Moving condition | K | Moving − static (pp) | 95% bootstrap CI (pp) |
|---|---:|---:|---:|
| Slow | 5 | −26.7 | [−46.7, −6.7] |
| Slow | 10 | 0.0 | [−20.0, +20.0] |
| Slow | 50 | −6.7 | [−20.0, 0.0] |
| Fast | 5 | +6.7 | [0.0, +20.0] |
| Fast | 10 | +13.3 | [0.0, +26.7] |
| Fast | 50 | 0.0 | [0.0, 0.0] |

The interaction is `(moving − static at K) − (moving − static at K=10)`, using
the same paired state draws. A negative value means that the motion-related change
is less favorable at that K than at K=10; it is not a direct comparison of their
absolute success rates.

| Moving condition | K versus K=10 | Interaction (pp) | 95% bootstrap CI (pp) |
|---|---:|---:|---:|
| Slow | 5 | −26.7 | [−46.7, −6.7] |
| Slow | 50 | −6.7 | [−26.7, +13.3] |
| Fast | 5 | −6.7 | [−26.7, +13.3] |
| Fast | 50 | −13.3 | [−26.7, 0.0] |

The largest observed penalty is slow K=5, both against its own static control and
in the interaction relative to K=10. Fast K=10 has a positive paired estimate,
but its interval includes zero. With only five state clusters, these contrasts
provide exploratory evidence about this intervention, not a definitive general
ranking. They remain conditional on the selected task and three seeds, and no
multiple-comparison adjustment is applied. The zero-width fast K=50 interval
reflects identical paired outcomes in this sample; it does not establish
equivalence. Adjusting for static performance does not remove floor/ceiling effects
or differences in exposure caused by contact stopping the intervention.

| Static control K | Seed 11 | Seed 22 | Seed 33 |
|---:|---:|---:|---:|
| 5 | 5/5 | 5/5 | 2/5 |
| 10 | 5/5 | 4/5 | 4/5 |
| 50 | 1/5 | 0/5 | 5/5 |

These seed-specific counts are descriptive. K=50 changes from no successes for seed
22 to five for seed 33, while K=5 changes in the opposite direction. This illustrates
dependence on policy randomness; three selected seeds are still too few for broad
claims about its distribution.

## Moving-target demonstration

The illustration was fixed to state 10 and policy seed 11 before those outcomes were
collected. Five distinct recordings cover static/slow/fast at K=10 and fast at K=5/50.

| Condition | K | Native success | Actions | Model calls |
|---|---:|---|---:|---:|
| Static | 10 | Yes | 239 | 24 |
| Slow reference | 10 | Yes | 168 | 17 |
| Fast reference | 10 | Yes | 153 | 16 |
| Fast reference | 5 | Yes | 141 | 29 |
| Fast reference | 50 | No | 280 | 6 |

In fast K=50, finger contact is logged at 2.112 simulated seconds and the gripper
command closes at step 45. Frames 41–60 show the subsequent withdrawal leaving the
can on the table. At frames 80–120, the hand travels toward the basket without the
can and opens near it. The can remains on the table through timeout; subsequent
replans do not produce a successful retrieval. This does not establish why the
model continues the empty transport sequence.

Fast K=10 also initially withdraws without the can, then visibly returns and later
transports it. Fast K=5 succeeds with 13 more model calls than K=10, saving only 12
simulation actions in this example. All three K=10 conditions succeed, while the
static case takes more approaches and actions. These comparisons describe one fixed
illustration; use the full 135-episode analysis for aggregate conclusions.

Both cameras were inspected at sampled whole-trajectory frames and denser contact
and retry windows. The review is assistant-generated, without independent human
annotation. Missed-grasp and drop counts remain unknown because sampled images and
occlusion do not support consistent event counts. Contact is not a verified grasp,
and a redirection after a replan does not prove that fresh feedback caused recovery.
`outputs/moving-target-report/demo-review.json` records inspected frames, observations,
uncertainties and raw-file hashes.

## Coverage of the assignment

| Requirement | Evidence |
|---|---|
| Ready-to-use VLA and simulator | Pinned SmolVLA LIBERO, LeRobot and hf-libero |
| Explain action prediction | H/K diagram, fixed flow iterations and saved action chunks |
| Control actions before replanning | Explicit K-step execution and configurable interval list |
| Tasks and initial situations | Three tasks; ten matched states per task |
| Task completion | Native success, task tables, all-state outcome chart and paired intervals |
| Motion correction | Command revisions, REPLAN markers and visually checked return movements |
| Grasping errors | Defined events, known/unknown counts, dense contact review and coverage |
| Calls and runtime | Logged calls, GPU synchronization and distinct compute/wall/simulated time |
| Controlled comparisons | Same model and runtime; paired state/image hashes; fixed randomized order |
| Demo | Instructions, both cameras, replans, paired motion and outcomes |
| Cases without benefit | More K=1 calls without a higher total and successful low-call examples |
| Reproducibility | Source manifests, raw records, strict evidence checks and checksummed archive |
| Scientific extension | Calibrated target-motion intervention, matched static controls and condition-by-K contrasts |

## A 7–10 minute presentation

- 1 minute: question, two images plus instruction, H versus K.
- 1 minute: three tasks, ten paired initial states and five K values.
- 2 minutes: success/cost table, task dependence and paired intervals.
- 1–2 minutes: video cases and what is observed versus hypothesized.
- 2 minutes: moving-target intervention, matched static controls and condition-by-K results.
- 1 minute: limitations, contribution and next experiments.

## Short answers for the presentation

**What is predicted?** A sequence of 50 seven-dimensional actions: three translation,
three rotation and one gripper component. The VLM encodes visual/text context; an
action expert transforms a noisy sequence using flow matching conditioned on the
observation and instruction. Ten generation iterations are computation, not robot steps.

**How is K different from H?** H is the fixed prediction horizon. K is how much of
that prediction is executed before the next observation; the rest is replaced.
K=50 continues to replan after each full chunk until success or timeout.

**Why 150 episodes?** Three tasks × ten initial situations × five K values. This
gives 30 matched situations for comparing every K, tests behavior across different
starting positions, and keeps the experiment feasible on the available GPU.

**Why another 135 episodes?** The moving-target question needs its own static controls:
five new initial states × three policy seeds × three K values × three conditions.
There are 15 episodes per condition/K cell, but only five initial-state clusters for
the uncertainty calculation. Repeating a scene with another policy seed does not
create a new physical starting situation.

**Does the can move while the robot carries it?** The controlled force starts after
0.5 simulated seconds and stops permanently at first finger contact. A brush can
activate this stop without securing a grasp. Afterward, only ordinary contact physics
governs the transfer; the controller does not pull the can out of the hand.

**Must the fast condition be harder?** No. Fast names a four-second reference period,
not a guaranteed object speed. At 2.5 seconds, the fast reference returns to its
initial x position while the slow reference is near its maximum offset. This is also
the first post-onset replan for K=50. Actual motion can lag and contact can end the
intervention earlier, so this is a possible timing mechanism rather than an identified
cause of any particular success.

**What stayed fixed?** Model and asset revisions, code, dependencies, FP32/batch1,
cameras, normalization, action horizon, flow iterations, control rate, step limits,
settling/warmup, seed, and hardware. Initial physical state and observation hashes
match across K. Random draws are matched by policy seed and simulation step; that
does not force identical trajectories or contact conditions after actions diverge.

**Why might replanning change behavior?** New observations can expose deviations,
but new stochastic predictions can also change a previously coherent sequence.
Both mechanisms vary with K. The moving-target test introduces a controlled physical
intervention, but isolating visual feedback from stochastic regeneration still
requires a separate ablation.

**Why not declare K=5 best?** It has the highest observed success count, but K=10
is cheaper, the shorter-interval success differences are uncertain, and tasks
favor different K values. The choice depends on the cost of failure and computation
in the intended application.

**Does success mean stable placement?** It means the native LIBERO predicate became
true and the episode ended. It can happen before release; post-release stability
was not measured. Do not silently substitute a different success definition.

**Why do short videos take minutes to compute?** Videos use simulated time at 20 Hz.
The simulator pauses during model calls. The measured call latency is much longer
than the 0.05-second simulation step, so physical real-time control was not tested.

**What would improve the study next?** More initial states and tasks for the moving
target, randomized oscillation phase, broader seed replication, independent human
annotation and separate feedback/noise ablations. An adaptive-K policy would require
its own controlled evaluation.

**What is original here?** The K-step executor, matched design, strict provenance
checks, saved full predictions/actions, controlled target-motion intervention, paired
condition-by-K comparisons, timing and motion analysis, annotated demos and
interpretation. The pretrained model and simulator are existing third-party
components; training them was not part of the assignment.

**How does this relate to existing methods?**
[ACT](https://www.roboticsproceedings.org/rss19/p016.pdf) predicts action chunks and
uses temporal ensembling of overlapping predictions. Here, one checkpoint and H=50
remain fixed while K changes. [RTC](https://arxiv.org/abs/2506.07339) addresses latency
by generating chunks during action execution. Our simulator pauses for inference,
so the project does not test asynchronous control or compare performance with ACT
or RTC. Its contribution is the controlled empirical evaluation and evidence.
