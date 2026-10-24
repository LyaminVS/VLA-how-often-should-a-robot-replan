# Does an execution cap improve adaptive replanning?

## Question and comparisons

The generation horizon H is the number of actions returned by SmolVLA. The
execution cap C is the largest prefix allowed before another observation.
This experiment keeps H=50 and tests C in {5,10,25,50}. It asks whether limiting
long open-loop intervals improves the success/computation trade-off of DVAC.

The complete comparison has nine distinct execution rules: fixed K in
{1,5,10,25,50} and DVAC with C in {5,10,25,50}. DVAC with C=1 always executes
one action, so its execution schedule is identical to fixed K=1. It is not
collected as a duplicate treatment. Exact trajectory identity also requires
the same sampler, observation processing, starting state and noise schedule.

Two comparisons receive particular attention:

1. DVAC C=10 versus fixed K=10: do occasional earlier updates help under the
   same maximum interval?
2. DVAC C=10 versus DVAC C=50: does preventing long intervals help?

Also report all caps against the corresponding fixed K, and the complete
success-versus-call-cost plot including fixed K=1 and K=5. A cap may improve
success by spending more calls. DVAC C=10 cannot use fewer calls per executed
action than the fixed K=10 schedule apart from differing episode-end effects;
per-episode totals also depend on how early success occurs. A higher success
count alone is not evidence of greater computational efficiency.

## Decision rule

The frozen sampler and instrumentation in `adaptive_replan/` are reused.
There are ten Euler updates, with the observation and instruction fixed.
For the last five updates, record the clean-action estimates z_i=x_i-t_i*v_i.
For each j=0,...,49, sum the population variance across the seven valid
normalized action coordinates. Padding is excluded. Reductions use float64
on the saved float32 estimates.

History W contains every variance value from each of up to five preceding
calls, retaining duplicates and all 50 future action positions. With alpha=2:

```text
tau = mean(W) + 2 * population_std(W)
j_first = first j such that V[j] > tau
K_raw = 50                    if no crossing exists
K_raw = max(1, j_first)        otherwise
K = min(C, K_raw)
```

The first call of every episode executes K=1 without a threshold and then
initializes W. The current variance array enters W only after choosing K.
Every episode resets this history. The cap applies both when no crossing
exists and when the first crossing is beyond C. Log K_raw, C, whether the cap
changed the choice, K, the first crossing, threshold, and history statistics.

The robot executes the original final action chunk's prefix, after ordinary
postprocessing and clipping. It never executes the average intermediate
estimate. The cap changes neither the number of predicted actions nor the
number of FM updates. In particular, it does not lower the cost of one call.

The underlying method is [DVAC](https://arxiv.org/html/2606.03847v1), Section 3.
Applying a hard cap to every branch while retaining H=50 is the intervention
studied here. We do not claim to reproduce the paper's full benchmark.

## Matched evidence and reuse

All methods use three tasks: soup can into basket, black bowl onto plate,
and opening the middle drawer. Each uses initial state indices 0 through 9
and policy seed 11. This gives 30 task/state/seed blocks per method.

The five fixed methods already have 150 recorded episodes in `outputs/compact/`
and `outputs/validation/`. Their complete manifests and raw data are retained.
The new collection contains only the four adaptive caps: 4 x 3 x 10 = 120
episodes. The comparison therefore contains 270 episodes, not 270 new runs.
The separate DVAC records at states 15 and 16 are not pooled into these blocks.

Before reuse, verify the pinned model/backbone, simulator assets, software,
precision, H=50, ten FM steps, 20 Hz control, ten settling steps, and limits
of 280 actions for soup/bowl and 300 for drawer. The baseline code and
dependency hashes must match. Every physical setting must have exactly one
episode for each fixed K, with identical reset hashes and first predictions.

A historical-reference check resets each of the 30 task/state situations
and generates one complete chunk with the same sampling seed. Require exact
agreement with the recorded initial state, observation, and 50-by-7 first
action chunk. This is a short inference check, not a repeated fixed rollout.
During collection, the same checks are applied before the first robot action
of each adaptive episode. A mismatch stops collection. These checks support
compatibility on observed inputs; they are not a proof of identical behavior
under every possible later observation.

The new methods use the same CPU/GPU resource settings and diagnostics.
Existing fixed runs used six Torch CPU threads; the new collection uses one
and records intermediate FM estimates. Thus fixed-versus-adaptive completion
and model-call counts are the primary comparison. Do not infer wall-time
speedup by mixing the old fixed timings with the new adaptive timings.
Adaptive caps may be compared with one another on their synchronized timing.

## Collection and integrity

Use `configs/capped.json`, with policy seed 11 and randomized order seed
20261024. Shuffle the 30 task/state blocks and the four adaptive methods
within each block before collection. All adaptive variants use alpha=2,
tail length 5, history length 5, minimum K=1, and the same first-call rule.
Freeze these choices and collection-source hashes before observing results.

Run sequentially on the RTX 2060 through `ssh home`, with one policy and one
simulator. Limit CPU allocation to two cores, memory to 8 GiB, swap to zero,
and numerical-library and video-encoder thread counts to one. Keep completed
episodes immutable and preserve incomplete attempts if interrupted.

Retain the passing plain/instrumented action-equality check, historical
reference audit, manifest, complete final action chunks, intermediate clean
estimates, variance arrays, decisions, executed actions, timings and videos.
Source manifests include all new collection code and frozen dependencies.
Historical references are bound to hashes of their manifests and evidence.
Independent verification replays variances and decisions, checks every
executed action and episode identity, and confirms all paired reset hashes.

## Analysis fixed before new results

Require exactly nine methods in each of the 30 matched blocks before a
complete report is published. Report every outcome, including failures.
For each method and task, report successes, calls per episode, calls per 100
executed actions, episode length, and motion diagnostics. For adaptive methods,
also report chosen/executed K, how often the cap binds, no-crossing frequency,
and the proportion of K=1 decisions after initialization.

Estimate paired differences using 10,000 bootstrap samples with seed
20261024. Within each of the three tasks, resample ten initial states with
replacement, keeping all nine methods together. Average tasks equally.
For a ratio such as calls per action, recompute the ratio of sums within each
resample. Report descriptive 95% percentile intervals and discordant outcomes.
No significance or equivalence claim follows from an interval containing
zero. These are three fixed tasks with one policy seed, not a representative
LIBERO benchmark or an evaluation of robustness across sampling seeds.

The fixed outcomes have already been examined, and they informed the choice
to study caps. Accordingly this is an exploratory comparison, not a held-out
confirmation. Report all four caps regardless of which looks best. Do not
select a winner and treat the same data as independent validation.

## Demonstrations and interpretation

The predetermined demo uses initial state 0, seed 11, for all three tasks;
show fixed K=10 beside DVAC C=10 and DVAC C=50. Include the instruction,
simulation step, chosen interval, cap, call count, replanning moments and
outcome. Video playback follows simulation time; inference pauses are not
portrayed as physical execution time.

For each direction of disagreement between DVAC C=10 and C=50, inspect the
first matching block in the frozen order when one exists. Label these as
outcome-selected illustrations. Sampled frames can suggest a missed grasp,
but do not establish its contact-mechanical cause or an error frequency.
Internal FM stability does not guarantee a correct action; frequent replanning
can also fail to recover. The intended result is a measured trade-off, not a
requirement that the adaptive method outperform every fixed interval.
