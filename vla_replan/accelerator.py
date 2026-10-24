"""Native CUDA/MPS execution; no model imports until a rollout is requested."""

import os
import platform
import subprocess
from contextlib import nullcontext


def prepare_platform():
    system = platform.system()
    if system == "Darwin" and platform.machine() == "arm64":
        os.environ.setdefault("MUJOCO_GL", "glfw")
        # Must be set before importing torch, not after MPS has initialized.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif system == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        raise RuntimeError("Real rollouts require Linux/CUDA or native Apple Silicon macOS/MPS")
    return system


def mac_hardware():
    values = {}
    for key, name in (("device", "machdep.cpu.brand_string"), ("shared_memory_bytes", "hw.memsize")):
        value = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        values[key] = int(value) if key.endswith("bytes") else value
    return values


class Accelerator:
    def __init__(self, device="auto", precision="fp32"):
        system = prepare_platform()
        import torch

        self.torch = torch
        self.device = ("mps" if system == "Darwin" else "cuda") if device == "auto" else device
        self.precision = precision
        expected = "mps" if system == "Darwin" else "cuda"
        if self.device != expected:
            raise RuntimeError(f"Use --device {expected} on {system}; got {self.device}")
        available = torch.backends.mps.is_available() if self.device == "mps" else torch.cuda.is_available()
        if not available:
            raise RuntimeError(f"PyTorch {self.device.upper()} is unavailable; run `vla-replan doctor`")
        if self.device == "mps" and precision != "fp32":
            raise ValueError("The macOS/MPS protocol supports fp32 only; use a separate fp32 config")
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    def noise(self, shape, seed):
        # A private CPU generator works across supported MPS versions and leaves global RNG untouched.
        # CUDA retains the original on-device noise stream, preserving the Linux protocol.
        sampling_device = "cpu" if self.device == "mps" else "cuda"
        generator = self.torch.Generator(device=sampling_device).manual_seed(seed)
        result = self.torch.randn(shape, generator=generator, device=sampling_device, dtype=self.torch.float32)
        return result.to(self.device) if self.device == "mps" else result

    def autocast(self):
        return self.torch.autocast("cuda", dtype=self.torch.float16) if self.precision == "fp16" else nullcontext()

    def synchronize(self):
        (self.torch.mps if self.device == "mps" else self.torch.cuda).synchronize()

    def peak_memory(self):
        # MPS exposes current/driver allocation, not a CUDA-equivalent peak counter.
        return None if self.device == "mps" else self.torch.cuda.max_memory_allocated()

    def reset_peak_memory(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def metadata(self):
        torch = self.torch
        result = {
            "backend": self.device,
            "precision": self.precision,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_cpu_threads": torch.get_num_threads(),
            "noise_device": "cpu" if self.device == "mps" else "cuda",
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") if self.device == "mps" else None,
        }
        if self.device == "cuda":
            result.update(device=torch.cuda.get_device_name(), vram_bytes=torch.cuda.get_device_properties(0).total_memory, cuda=torch.version.cuda)
        else:
            result.update(mac_hardware())
            result.update(vram_bytes=None, cuda=None, macos=platform.mac_ver()[0], peak_memory_supported=False)
        return result
