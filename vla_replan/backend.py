"""Version-pinned LeRobot adapter. Imports/downloads occur only for real Linux runs."""

import hashlib
import os
import platform
from contextlib import nullcontext
from importlib.metadata import version

from .core import Observation


def observation_hash(data):
    import numpy as np

    result = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value):
                result.update(key.encode())
                visit(value[key])
        else:
            array = np.ascontiguousarray(value)
            result.update(str(array.dtype).encode())
            result.update(str(array.shape).encode())
            result.update(array.tobytes())

    visit(data)
    return result.hexdigest()


def check_linux():
    if platform.system() != "Linux":
        raise RuntimeError(
            "Real LIBERO runs require Linux. Planning, tests, analysis and video work on macOS."
        )
    if version("lerobot") != "0.6.1":
        raise RuntimeError("This adapter requires lerobot==0.6.1. Install requirements/linux.lock.txt.")


class LiberoEnvironment:
    def __init__(self, config):
        check_linux()
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark

        self.benchmark = benchmark
        self.config = config
        self.env = None
        self.task_key = None
        self.instruction = ""

    def _observation(self, raw):
        import numpy as np

        # Display the same orientation as LiberoProcessorStep supplies to the VLA.
        frame = np.concatenate([raw["pixels"][key][::-1, ::-1] for key in ("image", "image2")], axis=1).copy()
        return Observation(
            payload={"raw": raw, "instruction": self.instruction},
            eef=raw["robot_state"]["eef"]["pos"].tolist(),
            gripper=raw["robot_state"]["gripper"]["qpos"].tolist(),
            frame=frame,
        )

    def reset(self, episode):
        import numpy as np
        from lerobot.envs.libero import LiberoEnv

        key = (episode.suite, episode.task)
        if key != self.task_key:
            self.close()
            suite = self.benchmark.get_benchmark_dict()[episode.suite]()
            matches = [
                i
                for i in range(suite.n_tasks)
                if suite.get_task(i).name == episode.task
                or suite.get_task(i).name.endswith("_" + episode.task)
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one task matching {key}, found {matches}")
            self.env = LiberoEnv(
                task_suite=suite,
                task_id=matches[0],
                task_suite_name=episode.suite,
                obs_type="pixels_agent_pos",
                observation_width=256,
                observation_height=256,
                control_freq=self.config.control_hz,
                control_mode="relative",
                init_states=True,
                hard_reset=True,
                num_steps_wait=self.config.settle_steps,
            )
            self.task_key = key
        if episode.state >= len(self.env._init_states):
            raise ValueError(f"Initial state {episode.state} unavailable; refusing modulo wraparound")
        # Upstream increments this on reset. Explicitly restore it for every treatment.
        self.env.init_state_id = episode.state
        raw, _ = self.env.reset(seed=episode.env_seed)
        self.instruction = self.env.task_description
        state = np.ascontiguousarray(self.env._init_states[episode.state])
        return self._observation(raw), {
            "instruction": self.instruction,
            "resolved_task_id": self.env.task_id,
            "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "initial_observation_sha256": observation_hash(raw),
        }

    def step(self, action):
        import numpy as np

        obs, _, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        return self._observation(obs), bool(info["is_success"]), bool(terminated or truncated)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.task_key = None


class SmolVLAPolicy:
    def __init__(self, experiment, device="cuda"):
        check_linux()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as HFSmolVLA
        from lerobot.processor.env_processor import LiberoProcessorStep

        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Use the Linux NVIDIA machine for measured runs (--device cuda)")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.device = device
        self.precision = experiment.precision
        model_path = snapshot_download(experiment.model_id, revision=experiment.model_revision)
        backbone_path = snapshot_download(
            experiment.backbone_id,
            revision=experiment.backbone_revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
        )
        cfg = SmolVLAConfig.from_pretrained(model_path)
        if cfg.chunk_size != experiment.chunk_size:
            raise ValueError("Checkpoint horizon does not match the experiment")
        cfg.device = device
        cfg.load_vlm_weights = (
            False  # Full checkpoint already contains the backbone; avoid BF16 init on Turing.
        )
        cfg.vlm_model_name = backbone_path
        cfg.num_steps = experiment.num_denoising_steps
        cfg.n_action_steps = experiment.chunk_size  # Our explicit executor owns K.
        cfg.use_amp = False
        self.policy = HFSmolVLA.from_pretrained(model_path, config=cfg, strict=True).to(device).float().eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": backbone_path},
            },
        )
        self.env_processor = LiberoProcessorStep()
        self.preprocess_observation = preprocess_observation
        self.runtime = {
            "device": torch.cuda.get_device_name(),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
            "precision": self.precision,
            "deterministic_algorithms": True,
            "torch_cpu_threads": torch.get_num_threads(),
        }

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def predict(self, observation, seed):
        import numpy as np

        torch = self.torch

        # All robot state leaves need a batch dimension for quaternion conversion.
        def batch(value):
            return (
                {k: batch(v) for k, v in value.items()}
                if isinstance(value, dict)
                else np.expand_dims(value, 0)
            )

        self.synchronize()
        with torch.inference_mode():
            raw = self.preprocess_observation(batch(observation.payload["raw"]))
            raw = self.env_processor.observation(raw)
            raw["task"] = [observation.payload["instruction"]]
            inputs = self.pre(raw)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(
                (1, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
            context = (
                torch.autocast("cuda", dtype=torch.float16) if self.precision == "fp16" else nullcontext()
            )
            with context:
                actions = self.policy.predict_action_chunk(inputs, noise=noise)
            # Processor acts on [batch, action_dim]; retain the checkpoint's unnormalizer.
            result = [
                self.post(actions[:, i, :].float()).squeeze(0).cpu().tolist() for i in range(actions.shape[1])
            ]
        self.synchronize()
        return result

    def synchronize(self):
        self.torch.cuda.synchronize()

    def peak_memory(self):
        return self.torch.cuda.max_memory_allocated()

    def reset_peak_memory(self):
        self.torch.cuda.reset_peak_memory_stats()
