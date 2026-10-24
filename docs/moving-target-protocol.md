# Moving-target replanning experiment

## Status and purpose

This document specifies an additional controlled stress test of the frozen SmolVLA
checkpoint on the LIBERO soup-can task. **The motion parameters below are fixed after
physical calibration on initial state 20.** Calibration evidence is saved in
`outputs/moving-calibration/physics-b/calibration.json`; the experimental configuration
is `configs/moving-target.json`. A separate nine-episode policy check on state 20
checks execution and contact-release integrity using these fixed parameters. It is
excluded from the main results and is not used to tune parameters for task success.
This document reports physical calibration, not main-series task-completion results.

The question is whether the effect of replanning interval K changes when the target
object moves before the robot contacts it. A successful result would be informative
even if frequent replanning does not improve performance. This is a modified task
environment, not an official LIBERO benchmark score.

The existing 150-episode study, its raw records, and its frozen implementation remain
unchanged. Results from the moving-target experiment have distinct episode identities,
configuration records, and output files. The new static condition supplies its own
matched control; historical static episodes are not interchangeable controls because
the initial states and instrumentation differ.

## Hypotheses

1. Moving-target conditions may reduce completion relative to the matched static
   condition, with a larger reduction for K=50 than for K=5 or K=10.
2. A faster reference trajectory may increase the penalty from infrequent updates.
3. Extra model calls may fail to improve success when the policy cannot acquire the
   moving object, when the object stops early after contact, or when placement is the
   main failure mechanism.

These are testable expectations, not conclusions. K still changes both the arrival
of fresh observations and stochastic chunk regeneration. This experiment does not
separate those mechanisms.

## Planned design

| Factor | Planned values |
|---|---|
| Task | `libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket` |
| Main initial states | 10, 11, 12, 13, 14 |
| Policy seeds | 11, 22, 33 |
| Executed actions before replanning, K | 5, 10, 50 |
| Conditions | Static, slow reference, fast reference |
| Calibration state | 20, excluded from the main results |
| Predicted horizon | H=50 actions |
| Control frequency | 20 Hz simulated time |
| Episode limit | 280 actions |
| Main episode count | 5 states × 3 seeds × 3 K values × 3 conditions = 135 |

For each state and policy seed, run all nine combinations of K and condition. Shuffle
their order within a block, and shuffle block order using a saved order seed. Keep the
checkpoint and asset revisions, processing, precision, model iterations, settling,
cameras, simulator, step limit, warmup, and recording settings fixed. Reset the physics,
policy, processors, disturbance clock, contact latch, and controller state every episode.

The policy seed controls generation noise and does not change the initial physical
situation. Match noise by policy seed and simulation step using the existing sampling
rule. Shared seeds do not force identical trajectories after actions diverge.
Verify initial physical-state and observation hashes across all matched conditions.

## Physical intervention

Use a bounded horizontal external force on the soup-can body. Do not repeatedly
overwrite its position. The force controller follows a sinusoidal reference along the
world x axis around the can's position after normal reset and settling. Introduce the
reference with a one-second smoothstep envelope to avoid a sudden velocity request:

```text
tau = simulated_time - onset_time
u = clip(tau / ramp_duration, 0, 1)
ramp = 3*u*u - 2*u*u*u
ramp_derivative = 6*u*(1-u) / ramp_duration  # zero outside 0 < u < 1
omega = 2*pi / period
x_reference = initial_x + amplitude * ramp * sin(omega*tau)
v_reference = amplitude * (ramp_derivative * sin(omega*tau)
                           + ramp * omega * cos(omega*tau))
force_x = clip(kp * (x_reference - actual_x)
               + kd * (v_reference - actual_vx), -force_cap, +force_cap)
torque_y = force_application_offset_z * force_x
```

Before onset, and after the contact latch is set, the intervention applies zero force
and zero moment. The application point is 0.03 m below the centre of mass
along world z. Applying horizontal force Fx at this point is equivalent to a force Fx
at the centre of mass plus the moment `(0, offset_z*Fx, 0)`. This represents **one
external point force**, not a separate orientation servo. The lower application point
is intended to reduce tipping from the interaction of horizontal forcing and table
friction; the physical calibration and remaining deviations are reported below.

The intervention supplies no y or z force and no other moment. Normal gravity, friction,
contacts, and robot actions remain active. Use the same gains, force cap, application
offset, amplitude, axis, onset, ramp, and initial phase for the two moving conditions;
their reference period is the only controller parameter that differs.

| Parameter | Fixed value | Interpretation |
|---|---:|---|
| Amplitude A | 0.02 m | Desired excursion from the initial x position |
| Slow period T | 8.0 s | Steady sinusoidal peak reference speed about 0.0157 m/s |
| Fast period T | 4.0 s | Steady sinusoidal peak reference speed about 0.0314 m/s |
| Onset | 0.5 simulated s after rollout begins | Excludes reset and settling |
| Smoothstep ramp | 1.0 simulated s | Multiplies the sinusoid; velocity includes the envelope derivative |
| Initial phase | 0 radians | Same trajectory phase across K and seeds |
| Proportional gain kp | 100 N/m | Same value for both moving conditions |
| Derivative gain kd | 0.5 N s/m | Same value for both moving conditions |
| Force cap | 1.0 N | Cap on the horizontal force magnitude |
| Application offset along world z | −0.03 m relative to the centre of mass | `force_application_offset_z_m`; implies a moment magnitude no greater than 0.03 N m |

The listed steady sinusoidal speeds do not bound the velocity during the smoothstep
ramp, whose derivative contributes to the reference velocity. The reference amplitude
and speed are not measurements of the object's motion.
Friction, tracking error, contact, and force saturation can make its actual motion
smaller, asymmetric, or intermittent. Report actual position and velocity traces,
tracking error, force saturation, and the duration of exposure before contact.
Do not describe the conditions as two measured object speeds without that evidence.

Both motion generation and contact detection use simulator time. Inference pauses the
simulator, so slower wall-clock model calls do not make the target move farther during
computation. This experiment does not test control under asynchronous inference latency.

### Contact release and the static control

Stop the external force permanently at the first detected contact between a geometry
of the target can and any finger or finger-pad geometry of the robot gripper. Record
the resolved geometry names and IDs. An arm contact, a table contact, or a contact with
another object does not satisfy this rule. Detect contact during physics integration
where supported; otherwise record the actual detection resolution and acknowledge
possible within-step delay or missed brief contacts.

The latch applies even if contact occurs before the scheduled onset. Once set, it
remains set for the rest of the episode, including if the fingers separate or the can
is dropped. Remove the intervention's outstanding force and moment when setting the
latch. Do not resume target motion after a missed grasp, drop, or attempted placement. This avoids
an external controller fighting the robot while it is holding the object.

**First contact is not a verified grasp.** A single finger can brush the can without
securing it. The release rule defines the end of the intervention, not grasp success.
The force turns off when contact is detected; motion from existing momentum can remain.

Run the static control through the same environment wrapper, contact detector, clock,
telemetry, and logging path. Its intervention force and moment are identically zero. Do not use a
position-holding PD controller for this condition: that would oppose ordinary robot
manipulation and would not be a static unforced control.

The intervention must not reset the policy, discard its current action chunk, inject
an extra model call, or change K. Continue the ordinary replanning schedule. Observation
frames and object telemetry must describe the same post-step physical state.

## Calibration and freezing

State 20 is reserved for physical calibration and implementation checks. Check that
the target body and finger contacts resolve correctly, positions
and velocities remain finite, reference motion produces visible target displacement,
and the cap prevents unstable or excessive forcing. Inspect friction-induced sticking,
force saturation, tracking lag, and unintended tipping or collision. Verify that the
force and moment stay zero after first contact and in every static episode. Inspect
the target's vertical excursion and tilt as well as x tracking: close x tracking alone
does not show that the can remains stable on the table.

An initial calibration candidate applied the horizontal force at the centre of mass.
It showed rocking despite close x tracking. The selected controller moves the equivalent
application point below the centre of mass while retaining the gains and force cap.
The choice addresses observed physical instability, not task-success differences among
K values. The earlier evidence remains in `outputs/moving-calibration/physics-a/`.

The fixed controller was checked for 240 actions (12 simulated seconds) per condition
on state 20 without loading the VLA. The robot received zero motion and an open-gripper
command. Initial-state and observation hashes matched across all three conditions.
No task-success comparison was used to select this controller.

| Physical diagnostic | Slow reference, T=8 s | Fast reference, T=4 s |
|---|---:|---:|
| Actual x range | 35.056 mm | 35.136 mm |
| Actual y range | 4.368 mm | 2.790 mm |
| Actual z range | 0.246 mm | 0.369 mm |
| Maximum tilt from initial orientation | 8.58° | 11.49° |
| Mean absolute x tracking error during forcing | 2.209 mm | 2.227 mm |
| Peak applied force | 0.275 N | 0.291 N |
| Fraction of enabled samples at the force cap | 0 | 0 |

The static control had zero applied force and target drift approximately
`1.55e-13 m` in this calibration. The previous centre-of-mass forcing produced about
9.7 mm of vertical excursion and visible rocking. The lower application point reduced
vertical excursion, but the final controller still produced measurable tilt and y
drift. This is therefore a comparison of **two reference periods under the same force
rule**, not an isolated effect of object speed or perfectly straight translation.
Main-series motion diagnostics must be reported separately: this one-state calibration
does not guarantee identical motion at other initial states or during robot interaction.

Controller parameters were chosen from physical tracking and stability. Preserve all
calibration configurations and outcomes, including the rejected physical controller.
Do not retune the controller after examining main-series success rates. If an integrity
check reveals a physical implementation error, document and fix the error, repeat the
necessary checks, and freeze the corrected configuration before main evaluation.

Store a calibration summary with the final gains, cap, application offset, axis, onset,
ramp, amplitude, periods, contact-geometry set, sampling resolution, configuration hash,
and implementation hash.
Calibration episodes are not counted among the 135 main episodes. If a later bug requires
an implementation change, preserve the old evidence and clearly identify which results
were rerun under the corrected configuration.

## Endpoints and logs

The primary endpoint is native LIBERO task success by 280 actions. Report success for
every condition and K. The primary treatment contrasts are the paired changes from
static to each moving condition within K. To assess whether K modifies sensitivity to
motion, compare these changes across K, rather than relying only on the rankings of
raw moving-condition success rates.

All planned episodes contribute to completion results, including episodes with no
finger contact, contact before onset, or little actual displacement. A failed task is
an outcome, not a technical error. Report implementation failures and incomplete blocks
separately; do not treat missing output as a task failure or silently drop an unfavourable
episode. Repeated technical attempts keep their attempt identity and do not count as
additional independent observations.

| Measurement | Definition and interpretation |
|---|---|
| Native success | LIBERO success predicate before the step limit; may precede release |
| Model calls and time | Actual calls and synchronized inference, simulation, compute, and episode times |
| First finger contact | First detected target–finger/pad contact; missing if none occurs |
| Contact before onset | Whether contact disabled forcing before it could begin |
| Intervention exposure | Simulated time during which the intervention was enabled before contact or termination |
| Reference trajectory | Desired x position and x velocity at each recorded step |
| Actual target motion | Target position and velocity; record frame and units |
| Applied force and moment | Actual intervention force, equivalent y moment, and force-cap saturation indicator |
| Vertical excursion and tilt | Target height range and maximum orientation change before first contact; diagnostics of unintended rocking |
| Tracking error | Reference minus actual x during the enabled interval; exposure length must be shown |
| Distance to target | A defined geometric distance from the logged end-effector point to the target-body reference point; not finger clearance or grasp quality |
| Next scheduled replan | First ordinary model call at or after onset, with its simulation step |

Time to the next replan measures an opportunity to use a new observation. It does not
measure detection, correction, or recovery. Distances and action revisions alone do not
establish that the model noticed the motion. Recovery and grasp errors require an explicit
event definition and direct evidence; leave ambiguous events unresolved.

Episodes without contact have an unobserved contact time up to termination. Do not set
their time to zero or average contact times only among contacted episodes without also
showing the denominator and the resulting selection. Plot contact occurrence and exposure
alongside any conditional timing summaries. If a time-to-contact analysis is used, specify
how early episode termination and the administrative time limit are handled.

Persist raw actions, full predicted chunks, sampling seeds, reference and measured
motion, intervention status, first-contact event, success, and timing. Video overlays
should show the instruction, condition, K, simulated time, replan markers, and whether
forcing is active or permanently released after contact. Use the same cameras and
recording settings in every condition.

## Statistical analysis

There are five independent initial-state units in this design, with three policy-seed
repeats per state. Average the three seed outcomes within each state before averaging
across the five states. Report all per-state and per-seed outcomes so that sensitivity
to a particular seed or starting arrangement remains visible.

Use paired resampling of initial states: each resampled state carries its complete
three-seed block and all nine K/condition combinations. For 5,000 bootstrap draws with
a saved analysis seed, compute moving-minus-static success differences for each K and
differences of those differences relative to K=10. This preserves pairing and seed
nesting. Never resample all 135 episodes as independent observations. This bootstrap
is conditional on the selected seed set; it does not fully estimate variability over
all possible model seeds or tasks.

Report exploratory 95% percentile intervals with their limitations: five state units
give weak uncertainty estimation, identical outcomes may produce degenerate intervals,
and multiple comparisons are not automatically corrected. A wide or overlapping interval
does not prove equivalence. Avoid a universal ranking of K, a guaranteed adaptive-policy
benefit, or claims that the result generalizes beyond this model, task, controller, and
tested states.

## Interpretation limits

The intended comparison is the effect of a fixed external-force rule, not identical
realized trajectories in every rollout. Robot interaction can change the target's path.
Different K values can also produce first contact at different times, thereby changing
the intervention duration. That dependence is part of this rule's effect. Filtering to
only episodes that reached contact would change the groups being compared and cannot
isolate a causal grasping mechanism.

State matching and seeds control the start, not the entire trajectory. A shorter K can
change actions before the first disturbance, and common noise at shared simulation
steps does not erase the resulting physical differences. A separate experiment that
branches from an identical saved simulator and controller state would be needed to
isolate reactions from a common immediate pre-intervention situation.

The standard success predicate remains distinct from stable placement after release.
An unmodified pretrained policy may be unfamiliar with a moving object or externally
applied force. Findings should therefore be described as robustness to this specified
modified environment, alongside the ordinary static results.
