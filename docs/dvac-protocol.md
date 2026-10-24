# Denoising-variance adaptive replanning with SmolVLA

## Research question and scope

Does choosing the number of executed actions from the internal flow-matching
trajectory improve the completion/computation trade-off compared with fixed
execution intervals? This is a transfer of Denoising-Variance Adaptive Chunking
(DVAC) to the project's frozen SmolVLA checkpoint. It is not a reproduction of
all experiments or an official benchmark score from the DVAC paper.

The reference is [Denoising Tells When to Replan: Denoising-Variance Adaptive
Chunking for Flow-Based Robot Policies](https://arxiv.org/html/2606.03847v1),
especially the clean-action estimate, denoising variance, adaptive threshold,
and prefix selection in Section 3 and Algorithm 1. The model is described in
the [SmolVLA paper](https://arxiv.org/abs/2506.01844).

The design and parameters below are fixed before collecting comparison results.
No parameter is selected by task-completion performance in these episodes. The
existing fixed-interval and moving-target data remain preserved. Neither is
pooled into this comparison; every method gets newly collected matched controls
with the same observation processing and diagnostic instrumentation.

## Method and indexing

Let `r` index a policy call, `i` a numerical solver iteration within that call,
`j=0,...,H-1` a future action, and `d=0,...,6` a valid action coordinate.
The observation and instruction remain fixed during a policy call. SmolVLA
generates a complete action chunk from Gaussian noise with `M=10` flow-matching
Euler steps, from flow time 1 toward 0. With the installed sampler's actual
time `t_i`, latent chunk `x_i`, and predicted velocity `v_i`, record

```text
z_i[j,d] = x_i[j,d] - t_i * v_i[j,d]
```

before applying the Euler update. The original sampler is responsible for all
updates. Diagnostic capture does not replace its integrator or consume another
random sample. Keep only the final `L=5` estimates. For each future action,

```text
z_mean[j,d] = mean_i(z_i[j,d])
V_r[j] = sum_d mean_i((z_i[j,d] - z_mean[j,d])**2)
```

The variance is the population variance, with denominator `L`. It measures
disagreement between clean-action estimates inside one generation. It is not
variance between separate replanning calls, adjacent actions, or independent
samples. Values are calculated for the seven valid normalized action channels;
padding channels are excluded. Logged float32 clean estimates are converted to
float64 for diagnostic reductions. Model inference and integration remain
float32. Variance is calculated before action unnormalization or clipping.

Keep the previous `m=5` calls' entire 50-element variance arrays in a queue `W`.
Flatten these arrays with every value retained; repeated values are not deduplicated.
Before adding the current call's variance, compute

```text
tau_r = mean(W) + alpha * population_std(W),  alpha = 2
j_first = first j for which V_r[j] > tau_r
K_r = H                                  if there is no crossing
K_r = max(N_min, j_first)                 otherwise
```

Here `H=N_max=50` and `N_min=1`. Because indices start at zero, the crossing
action is excluded from the prefix unless the minimum of one action forces it
to be executed. The strict `>` comparison matters. Execute the first `K_r`
actions from the final generated chunk, after the usual postprocessing and
clipping. The mean clean estimate is never used to control the robot. Add
`V_r` to the queue only after the decision, dropping its oldest array when
the queue exceeds five calls.

The paper's algorithm does not specify an empty-history initialization. This
implementation explicitly uses `K=1`, `tau=null`, and no crossing on the first
call of each episode, then adds that call's variance to the history. The history
is reset at every episode. This bootstrap rule and the use of seven valid
SmolVLA channels are implementation choices; they must not be attributed to
the paper as prescribed settings.

## Paired comparison

| Factor | Fixed values |
|---|---|
| Checkpoint | `HuggingFaceVLA/smolvla_libero` |
| Checkpoint revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| Backbone revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Methods | Fixed K=5, fixed K=10, fixed K=25, DVAC |
| Tasks | Soup into basket; bowl onto plate; open middle drawer |
| Initial states per task | 15, 16 |
| Policy seeds | 11, 22 |
| Episode count | 3 tasks × 2 states × 2 seeds × 4 methods = 48 |
| Generation horizon / solver steps | 50 actions / 10 steps |
| Control frequency | 20 Hz simulated time |
| Settling | 10 steps |
| Limits | 280 actions for soup/bowl; 300 for drawer |
| Precision / warmup | FP32 / 2 warmup calls |
| Randomized block order seed | 20261024 |

The exact task identifiers are:

- `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- `libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `libero_goal/open_the_middle_drawer_of_the_cabinet`

Each task/state/policy-seed block contains all four methods. Shuffle block order
and method order within blocks using the recorded order seed. Verify initial
simulator-state and observation hashes across methods and seeds for each task
and initial state. Sampling noise is indexed by policy seed and the simulation
step of the call, as in the existing project. Once replanning schedules differ,
shared seeds do not imply identical future observations or noise at every call.

Run the methods sequentially on the same RTX 2060. All methods capture the same
clean-action estimates and calculate the same variance signal. Fixed methods
ignore the signal for their execution decision. This makes diagnostic overhead
comparable. Log policy inference and diagnostics, selector computation,
simulator stepping, and episode wall time separately. The synchronous simulator
pauses during inference; this does not test asynchronous control latency.

## Integrity checks and retained evidence

Before collection, compare the original and instrumented sampler's final
predictions with the same observations and noise. Use an excluded initial state
for each task. Require exactly equal final action values; record maximum error,
the resolved upstream sampler source hashes, checkpoint/runtime identity, and
the configuration/source hashes. These checks establish instrumentation integrity,
not task-performance calibration.

Retain a run manifest and per-episode summaries, action logs, complete generated
chunks, final five clean estimates, their flow times, variance arrays, threshold,
history size, first crossing, chosen prefix, and timings. Hash published evidence.
An independent verifier recomputes variance, the past-only threshold, bootstrap,
crossing and prefix decisions from saved estimates. It verifies every clipped
executed action against its generated chunk, call seeds, replanning boundaries,
termination, complete planned episode identities, and matched initial hashes.

Videos display the instruction, motion, replanning moments, selected interval,
and outcome. A failed episode is retained just like a successful episode.
The comparison demo uses the predetermined state 15 and policy seed 11 for
each of the three tasks, with fixed K=10 on the left and DVAC on the right.
Include all three task pairs regardless of their outcomes.

## Analysis and limits

Report successes/episodes per method and task, policy calls per episode and per
executed action, inference/selection/simulation/wall times, chosen interval
distributions, and paired outcomes. Display the full task/state/seed outcome
table so that aggregate differences can be checked directly. Report discordant
pairs: DVAC succeeds when the fixed method fails, and the converse.

Wilson 95% intervals are descriptive binomial intervals, not cluster-adjusted
uncertainty estimates. There are only two initial states per task and two policy
seeds per state; these are correlated repetitions, not twelve independent task
settings. Do not use these intervals as a significance test or report a paired
test that treats the two seeds as independent physical situations. Task/state
means over the two seeds are also reported to show the six physical settings
explicitly. Paired outcome discordance is reported without a p-value.

The primary evidence is the success/computation trade-off, not whether DVAC wins.
More model calls can fail to fix a poor grasp or a systematically wrong prediction.
A stable internal flow trajectory can still generate an incorrect action. A
large variance is an empirical stability signal, not a calibrated probability
of task failure. Its scale depends on the checkpoint, solver, integration grid,
tail length, and normalization. This small transfer study does not establish
universal parameter choices or isolate fresh observations from stochastic
regeneration of action chunks.
