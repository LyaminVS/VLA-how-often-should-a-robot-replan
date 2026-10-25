# AutoHorizon with the frozen SmolVLA policy

## Research question

Does an attention-based execution horizon produce a useful task-completion and
model-call trade-off compared with fixed intervals and denoising-variance
adaptive chunking (DVAC), when the policy, tasks, initial states and generation
horizon remain unchanged?

This experiment transfers the published AutoHorizon implementation to SmolVLA.
It does not retrain the policy or reproduce the original authors' complete
benchmark. It compares scheduling methods as implemented; it is not a controlled
ablation of the information in FM variance alone. AutoHorizon and DVAC use
different signals and different selection rules.

The implementation reference is the AutoHorizon repository at commit
[`c7504f1756109103f2cfcc2e23f1b1a23841c885`](https://github.com/hatchetProject/AutoHorizon/blob/c7504f1756109103f2cfcc2e23f1b1a23841c885/src/openpi/models_pytorch/pi0_pytorch.py),
specifically `_soft_pointer_prefix` and `bidir_soft_pointer`. The pinned code
defines the evaluated rule. Its mapped backward index and additional pointer
join test are retained explicitly; substituting a reversed-prefix length or a
coverage-only explanation would evaluate a different rule.

## Frozen experimental design

| Factor | Value |
|---|---|
| Checkpoint | `HuggingFaceVLA/smolvla_libero` |
| Checkpoint revision | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| Backbone revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| New methods | AutoHorizon with execution caps C=5, 10, 25, 50 |
| Tasks | Soup into basket; bowl onto plate; open middle drawer |
| Initial states | 0 through 9 for every task |
| Policy seed | 11 |
| New episodes | 3 tasks x 10 initial states x 4 caps = 120 |
| Generation horizon H | 50 actions |
| Flow-matching evaluations | 10 per model call |
| Selected FM evaluation | Third, counted from one; flow time approximately 0.8 |
| AutoHorizon settings | Hold threshold 0.3; entropy quantile 0.9; run length 1 |
| Selection arithmetic | CPU, PyTorch float32 |
| Execution frequency | 20 Hz simulated time |
| Settling | 10 actions before policy control |
| Episode limits | 280 actions for soup/bowl; 300 for drawer |
| Precision | FP32 |
| Warmup | Two model calls, excluded from episode metrics |
| Order seed | 20261024; shuffled blocks and methods within blocks |

The task identifiers are:

- `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- `libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `libero_goal/open_the_middle_drawer_of_the_cabinet`

The existing 270 episodes provide five fixed-interval methods and four DVAC caps
on the same 30 task/state/seed settings. Those outcomes remain unchanged. The
new comparison can therefore contain 390 episodes across 13 methods, with 30
episodes per method. No additional policy seeds or tasks are implied by that
total. Cap values and selector settings are fixed before collecting AutoHorizon
outcomes; all four caps are reported, regardless of their performance.

The existing fixed and DVAC results have already been examined. This is an
exploratory extension on known evaluation settings, not a held-out confirmation
or a preregistered study of previously unseen tasks.

## Attention captured from SmolVLA

Let i index an FM evaluation within one policy call and let j,k index the 50
future action tokens. The current observation and instruction remain fixed
through all ten FM evaluations. Capture attention at i=3 only; do not average
across FM evaluations or across separate replanning calls.

For each action-expert self-attention layer, use the unchanged query and key
tensors after positional rotation. Reconstruct masked scaled dot-product
attention in float32, including the same key-head expansion and attention mask
as the model. Apply softmax over the complete key sequence first, then extract
the 50-by-50 action-query/action-key block. Average this block over attention
heads, and then average the resulting matrices over the captured self-attention
layers:

```text
A[j,k] = mean_layer(mean_head(attention[layer, head, j, k]))
```

Cross-attention layers are excluded because their keys do not represent the
future action sequence. The saved matrix A contains the averaged raw action
attention weights before the selector's row normalization. The per-layer head
means and the captured layer indices are also saved, so layer averaging can be
independently checked.

The original attention computation still produces the hidden states used by
the policy. The diagnostic reconstruction does not replace attention outputs,
velocity predictions, FM integration, or final actions, and it consumes no
sampling noise. The frozen model still generates all 50 actions on every call.
An excluded-state equality check must confirm unchanged final action chunks
before collection begins.

## Exact selection rule

All indices below start at zero. The selection implementation uses float32 on
the CPU and epsilon=1e-12. Its initial normalization is

```text
P[j,k] = A[j,k] / (sum_k A[j,k] + epsilon).
```

Each directional helper normalizes its input rows once more, as the pinned
implementation does. This second pass is retained for numerical parity.

For the forward direction, calculate the expected attended position and its
cumulative maximum:

```text
e_f[j]  = sum_k k * P_direction[j,k]
mu_f[j] = max(e_f[0], ..., e_f[j])
dmu_f[0] = mu_f[0]
dmu_f[j] = mu_f[j] - mu_f[j-1]                 for j > 0
```

The normalized entropy uses probabilities clamped from below by epsilon:

```text
Q[j,k] = max(P_direction[j,k], epsilon)
h[j] = -sum_k Q[j,k] * log(Q[j,k]) / log(H)
tau_h = quantile(h, 0.9)
reliable[j] = (h[j] <= tau_h)
hold[j] = (dmu_f[j] < 0.3) and reliable[j]
```

The first eligible hold selects the forward stopping row r_f. If no hold
exists, r_f=H-1. With run length 1, only one eligible row is required. A hold
means a small pointer increment; it is not a large jump. Convert its pointer
to a count of actions:

```text
N_f = clamp(floor(mu_f[r_f]) + 1, 1, H).
```

Apply the same directional computation to P with both action axes reversed,
obtaining the reverse pointer mu_rev and stopping row r_rev. Map this pointer
and row back to original coordinates:

```text
mu_b[j] = H - 1 - mu_rev[H - 1 - j]
r_b = H - 1 - r_rev
N_b = clamp(floor(mu_b[r_b]) + 1, 1, H)
join_exists = any_j(mu_b[j] - mu_f[j] <= 1)
```

The pinned implementation defines N_b in these mapped original coordinates.
It is not `floor(mu_rev[r_rev])+1`. Its full-horizon branch additionally
requires a pointer join:

```text
K_raw = H       if N_f + N_b >= H and join_exists
K_raw = N_f     otherwise
K = min(C, K_raw).
```

This raw rule operates on the full H-by-H attention matrix before the execution
cap is applied. The cap changes neither the predicted horizon nor the number
of FM evaluations. Each episode resets the selector. AutoHorizon is stateless
across calls and uses the same attention rule on its first call; it has no
forced first-call K=1. DVAC's retained first-call initialization is therefore
one of the differences between the scheduling methods.

The robot executes the first K actions from the original final prediction,
after the existing unnormalization and clipping. It does not execute attention
weights, intermediate estimates, or an average action trajectory. Success or
termination may stop an episode inside a selected prefix, so selected K and
executed prefix length are logged separately.

CPU float32 selection is a deliberate reproducibility choice. It allows the
saved matrix to reproduce decisions without a GPU, but a reduction exactly at
a threshold may differ from an original GPU implementation. The method is
identified as `official_code_mapped_backward_join_v1`. No attention entropy,
pointer plateau, or selected horizon is interpreted as a calibrated probability
of task failure.

## Collection, reference gates and verification

Run the new episodes sequentially on the RTX 2060 through `ssh home`, with one
policy and one simulator. Limit numerical-library and encoder threads, CPU
allocation, memory and swap using the same bounded collection setup as the
capped comparison. Save the manifest, configuration fingerprint, source hashes,
upstream sampler hashes and machine/runtime metadata before the first episode.
Completed evidence is immutable; preserve failed attempts separately.

Before collection, compare ordinary SmolVLA inference with attention-instrumented
inference using the same observation and sampling seed 101 on excluded initial
state 21 of each task. Require exactly equal final 50-by-7 action predictions.
Retain attention shape, capture metadata, the uncapped selector result and the
resolved sampler source hashes in the invariance record.

Bind all 30 retained DVAC C=50 reference episodes to hashes of their manifest,
invariance evidence, summaries and raw evidence files. Before each new rollout
moves the robot, require the same simulator reset hash, initial observation
hash and complete first prediction as its retained reference. The policy noise
seed remains indexed by policy seed and simulation step:

```text
sampling_seed = (policy_seed * 1000003 + simulation_step * 97409) mod (2^63 - 1).
```

Once schedules differ, methods visit different observations and may make calls
at different simulation steps. A common policy seed does not imply identical
noise at every call. No additional RNG stream is needed by AutoHorizon.

Each new call retains the final action chunk, standard FM clean estimates and
variances, selected-step attention matrix, per-layer attention matrices,
capture metadata, every selector pointer/entropy/mask/intermediate, raw and
capped K, and timings. Each executed action retains its source chunk offset,
replanning marker, cap, selected horizon, post-action success and simulation
timestamp. Videos show the instruction, robot motion, replanning moments,
selected interval and outcome.

The independent verifier must:

- Check the complete planned episode inventory, configuration and source hashes,
  episode identities, file hashes and the recorded invariance gate.
- Validate and, if relocated, safely resolve the retained reference paths;
  replay the original DVAC C=50 reference traces with their existing verifier.
- Reconstruct the attention layer mean, verify its FM-step and layer metadata,
  and independently replay the complete pinned float32 selector. Reusing the
  collection selector as the sole verifier is insufficient.
- Recompute standard FM variances, verify all sampling seeds and replanning
  boundaries, and match every clipped executed action to its predicted prefix.
- Verify first-chunk equality with retained references and first attention/FM
  diagnostics across caps, and reconcile all calls, steps, caps and outcomes.
- Reject a forced bootstrap, a different backward-index rule, a missing join
  condition, or incorrect raw decisions even when a small cap hides the error.

The retained per-layer matrices verify averaging and selector replay, not the
entire neural attention reconstruction from first principles: Q/K tensors are
not archived. Extraction source hashes, inspected hook logic and the equality
checks complement the trace-level evidence.

## Analysis and interpretation fixed before new outcomes

Report every method and every task, not only the best-looking cap. Primary
quantities are task success and model calls per episode, shown jointly on a
success-rate-versus-calls plot. Also report calls per 100 executed actions,
episode length, selected and executed K distributions, cap-binding frequency,
first-call K and measured inference/selection/simulation/wall time.

For each cap C, compare AutoHorizon C with DVAC C and fixed K=C using matched
task/state/seed blocks. Also show all 13 methods together. A matching cap is
only a matching maximum interval; it is not an equal computational budget or
an equal realized interval distribution. Early success changes episode length
and call count. A method that succeeds more often while making more calls is
not automatically more efficient.

Use 10,000 task-stratified paired bootstrap draws with seed 20261024. In each
task, resample its ten initial states with replacement, retaining every method
together. Give the three tasks equal weight. Recompute ratio metrics from sums
within each draw. Report descriptive 95% percentile intervals, individual
paired outcomes and both directions of discordance. The unit is the physical
task/state setting, not each action, call, or diagnostic value.

Attention extraction adds diagnostic work to an AutoHorizon call. Retained
DVAC calls have FM diagnostics but no attention extraction, and the historical
fixed methods also differ in instrumentation and CPU thread count. Therefore
equal calls do not guarantee equal latency. Model-call counts are the common
cost measure; observed timing is reported with its instrumentation conditions.
The synchronous simulator pauses during inference, so these rollouts do not
measure asynchronous control latency or delayed-observation robustness.

A favorable result would support this complete attention-based scheduler in
the tested settings. It would not establish that attention is calibrated
confidence, identify a causal advantage of one internal signal alone, or prove
generalization to arbitrary LIBERO tasks. The comparison includes three fixed
tasks, ten initial states per task and one policy seed. Small success-count
differences and overlapping descriptive intervals must not be presented as
proof of superiority or equivalence. Failed episodes and unfavorable caps are
part of the result.
