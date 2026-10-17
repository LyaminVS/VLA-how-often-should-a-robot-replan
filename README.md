# VLA: how often should a robot replan?

A study of how replanning frequency affects the behavior of a robot controlled by a
pretrained Vision-Language-Action model. The project uses **SmolVLA**, **LeRobot**, and
**LIBERO / MuJoCo**. The model controls the robot from images, robot state, and a natural
language instruction. No model training is performed in this project.

## Research question

How often should a robot receive a new observation and recompute its actions? Frequent
updates offer opportunities to correct motion, but require additional computation and
may interrupt a coherent action sequence. Executing a longer part of a plan reduces the
number of model calls while extending the interval without updated visual feedback.

This project examines the resulting trade-off in task success, observed motion
corrections, grasping errors, model calls, and execution time.

## Method

At each replanning step, SmolVLA receives two images, the robot state, and the instruction,
then predicts a sequence of **H = 50 actions**. Each action specifies translation,
rotation, and a gripper command.

```text
Images + robot state + instruction
                 ↓
     SmolVLA predicts 50 actions
                 ↓
     The robot executes the first K actions
                 ↓
     New observation → new plan
```

The parameter **K** determines the number of actions executed before the next observation.
The tested values are **1, 5, 10, 25, and 50**. The unexecuted remainder of the previous
plan is replaced. The prediction horizon H and generation settings remain fixed, so the
comparison isolates the execution interval between replanning calls. K=50 also replans
if the task has not finished after the complete action sequence.

## Experiment

The main experiment comprises **30 episodes: 3 tasks × 2 initial states × 5 values of K**.

| Task | LIBERO suite | Relevant behavior |
|---|---|---|
| Place the alphabet soup can in the basket | Object | Grasping, retention during transport, placement |
| Place the black bowl on the plate | Spatial | Selecting an object among nearby objects, placement accuracy |
| Open the middle drawer | Goal | Approaching the handle, making contact, pulling |

Every value of K uses the same initial states and policy seed. Checksums verify identical
initial states and images. The model, cameras, normalization, step limits, and execution
conditions are held constant; run order is shuffled. Five preliminary technical episodes
are excluded from the reported results.

## Results

| K | Soup can | Bowl | Drawer | Total successes | Mean model calls | Model + environment, s/episode |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0/2 | 1/2 | 1/2 | 2/6 | 228.2 | 180.6 |
| 5 | 2/2 | 1/2 | 2/2 | 5/6 | 34.2 | 30.1 |
| 10 | 2/2 | 0/2 | 2/2 | 4/6 | 17.8 | 17.8 |
| 25 | 1/2 | 0/2 | 1/2 | 2/6 | 10.5 | 13.6 |
| 50 | 1/2 | 1/2 | 0/2 | 2/6 | 5.2 | 9.6 |

The relationship was **non-monotonic** in the selected initial situations. K=5 produced
the most successful episodes; K=10 required less computation than K=5. The most frequent
replanning, K=1, did not achieve the highest success rate. Bowl outcomes depended on the
initial state: K=1 and K=5 succeeded in the first state, whereas only K=50 succeeded in
the second.

The videos also reveal different failure patterns. For example, the can may slip during
transport, and a subsequent observation does not necessarily lead to recovery of the
grasp. The drawer task includes repeated approaches after unsuccessful contact.
Observed behavior is distinguished from hypotheses about its causes.

**Scope of the conclusions.** Six initial situations and one policy seed support a compact
behavioral study, not a universal optimum for K. Success follows the standard LIBERO
criterion and may occur before object release. Ambiguous grasping events remain unknown;
gripper closing commands are not automatically counted as errors. The simulator pauses
while the model computes, so this experiment does not evaluate real-time robot control.

## Project materials

- [Full report](output/pdf/vla-replanning-report.pdf): methods, results, figures, analysis
  of errors and corrections, conclusions, and limitations.
- [Demonstration video](outputs/compact/demo.mp4): three paired comparisons showing the
  instruction, robot motion, REPLAN events, and outcomes.
- [Experimental protocol and metric definitions](docs/protocol.md).
- [Results and oral presentation guide](docs/report-outline.md).
- [Installation, reproduction, and artifact assembly](docs/linux-checklist.md).
- `outputs/compact/`: original logs and videos for all 30 episodes, tables, figures, and annotations.

The implementation is in `vla_replan/`, experiment configurations in `configs/`, tests in
`tests/`, and analysis and presentation tools in `scripts/`. The project pins dependencies
and model revisions, checks initial conditions, and supports resuming interrupted runs.
Execution details are provided in the documentation.

## Sources

- [SmolVLA paper](https://arxiv.org/abs/2506.01844).
- [SmolVLA LIBERO checkpoint used in this project](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102).
- [SmolVLA implementation in LeRobot](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/modeling_smolvla.py).
- [LIBERO documentation](https://huggingface.co/docs/lerobot/en/libero).

The model and simulator are existing third-party components. The project's contribution
is the configurable execution of the first K actions, controlled comparisons, metric
collection, visualization, and behavioral analysis. Dependencies and model weights retain
their respective licenses.
