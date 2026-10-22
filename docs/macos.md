# Native Apple Silicon macOS / MPS

This is an execution backend for the same K-step experiment, using native arm64
Python 3.12, PyTorch MPS and MuJoCo GLFW rendering. It does not retrain the model or
change the H=50 prediction horizon, task definitions, camera processing, or K-step
executor. Intel Macs and CPU-only measured rollouts are not supported by this path.

## Install and run

From a dedicated checkout in a logged-in macOS desktop session:

```bash
bash scripts/macos_setup.sh
bash scripts/run_macos.sh configs/smoke.json outputs/mac-smoke --limit 1
```

Setup installs the pinned `requirements/macos.lock.txt` into `.venv`. It does not
download model weights or execute the experiment. The run script prepares pinned
LIBERO assets, then starts the normal CLI with `--device mps`. Initial use downloads
the model and assets from Hugging Face. No CUDA installation is needed.

Resume the smoke experiment by repeating the run command without `--limit 1`.
For a larger experiment, choose another output folder:

```bash
bash scripts/run_macos.sh configs/compact.json outputs/mac-compact
.venv/bin/python -m vla_replan doctor
.venv/bin/python -m pytest -q
```

GLFW needs an available macOS OpenGL/display session; this path does not claim to
support a headless Linux-style EGL setup. Scripts set `MUJOCO_GL=glfw` and
`PYTORCH_ENABLE_MPS_FALLBACK=1` before torch is imported. Explicit existing values
are respected. FP32 is the validated precision; an FP16 configuration is rejected
before downloading the model.

## Linux-only transitive dependencies

`hf-libero` and `robomimic` require EGL device-discovery packages unconditionally.
Those packages are unnecessary for GLFW and can fail to build on macOS.
`macos-overrides.txt` changes only those two requirements to Linux-only markers.
The rest of the dependency graph is resolved and pinned normally. Rebuild the lock:

```bash
.tools/bin/uv pip compile requirements/macos.in \
  --override requirements/macos-overrides.txt \
  --python-version 3.12 --python-platform aarch64-apple-darwin \
  --output-file requirements/macos.lock.txt
```

Install it with `uv pip sync`, as the setup script does. A subsequent generic
dependency resolver may try to restore the unwanted EGL packages because their
upstream metadata is unchanged. The override is a packaging compatibility measure,
not a replacement EGL implementation.

## Randomness, timing and memory

- CUDA retains the original CUDA-local generator and sampling stream.
- MPS uses a private, seeded CPU generator, then transfers noise to MPS. This avoids
  dependence on MPS generator availability and does not reset global RNG state.
  `(policy_seed, simulation_step)` still selects the same noise at a shared step.
- CPU-generated MPS noise differs from the CUDA noise stream. Do not expect identical
  trajectories across devices, even with equal numerical seeds. Deterministic
  algorithms are requested, but repeatability must be checked on the actual stack.
- `torch.mps.synchronize()` bounds the timed prediction, including the noise transfer
  and preprocessing/postprocessing. CUDA uses its original synchronization calls.
- MPS does not expose an equivalent peak-allocation counter here. The summary records
  `peak_gpu_memory_bytes: null`; current allocation is not misreported as a peak.
- Hardware metadata records the Apple chip, shared RAM, macOS, selected backend,
  noise device and fallback setting. The manifest hashes the macOS dependency lock.

The simulator pauses during inference. Its 20 Hz control frequency is simulation
time and is not a demonstration of 20 Hz real-time physical robot operation.

## Keep results separate

The checked-in `outputs/compact` data were measured on Linux/RTX 2060 with another
PyTorch/MuJoCo stack. Always use a new Mac output directory. Source, runtime,
configuration and hardware guards intentionally prevent resuming old results with
this changed code or a different device. Existing results are not rewritten.

## Verified smoke check

On 2026-10-22, a new environment created by `macos_setup.sh` was tested on Apple M5,
32 GiB shared RAM, macOS 26.5, Python 3.12.13, PyTorch 2.11.0 and MuJoCo 3.3.7.
All 46 unit tests, Ruff and shell syntax checks passed. Repeating setup also passed.

The first episode was launched directly through the CLI with `--device auto`.
Repeating the experiment through `run_macos.sh` preserved that episode byte for byte
and completed the remaining four K values. The task was
`pick_up_the_alphabet_soup_and_place_it_in_the_basket`, state 0, policy seed 11.

| K | Control steps | Model calls | Task success |
|---|---:|---:|---|
| 1 | 207 | 207 | Yes |
| 5 | 142 | 29 | Yes |
| 10 | 143 | 15 | Yes |
| 25 | 137 | 6 | Yes |
| 50 | 126 | 3 | Yes |

The audit checked identical initial state and observation hashes, all executed
actions against the clipped prefixes of full 50x7 predictions, sampling seeds,
and video frame counts at 20 fps. First predictions were bit-identical across all
five K values. Frames from the start, middle and end of K=50 were visually checked:
both the external and wrist cameras rendered correctly. A separate real-MPS noise
check confirmed equal seeds reproduce noise without advancing the global CPU RNG.

[Machine-readable validation record](macos-validation.json) includes source/lock
hashes, versions, episode timings and video hashes. Local rollout files are ignored
by Git; the original published Linux results remain unchanged. This five-episode
check establishes execution on the tested Mac, not an accuracy benchmark or physical
robot validation. A fresh NVIDIA rollout was not available; CUDA behavior is covered
by unit tests, including its original generator and memory counters.
