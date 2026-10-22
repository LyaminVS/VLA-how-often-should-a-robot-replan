import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_replan import accelerator, cli, runner
from vla_replan.storage import dependency_lock_path


@pytest.fixture
def torch_stub(monkeypatch):
    calls = []

    class Generator:
        def __init__(self, device):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class Tensor:
        def to(self, device):
            calls.append(("transfer", device))
            return self

    def randn(shape, *, generator, device, dtype):
        calls.append(("noise", tuple(shape), generator.device, generator.seed, device, dtype))
        return Tensor()

    torch = SimpleNamespace(
        Generator=Generator, randn=randn, float32="fp32", float16="fp16",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("cuda_sync"),
            max_memory_allocated=lambda: 123,
            reset_peak_memory_stats=lambda: calls.append("cuda_reset"),
        ),
        mps=SimpleNamespace(synchronize=lambda: calls.append("mps_sync")),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True),
            cudnn=SimpleNamespace(benchmark=True, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        ),
        use_deterministic_algorithms=lambda enabled: calls.append(("deterministic", enabled)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(accelerator.platform, "machine", lambda: "arm64")
    return torch, calls


@pytest.mark.parametrize("system,renderer", [("Linux", "egl"), ("Darwin", "glfw")])
def test_graphics_defaults_and_explicit_override(monkeypatch, system, renderer):
    monkeypatch.setattr(accelerator.platform, "system", lambda: system)
    monkeypatch.setattr(accelerator.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    accelerator.prepare_platform()
    assert accelerator.os.environ["MUJOCO_GL"] == renderer
    if system == "Darwin":
        assert accelerator.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    else:
        assert accelerator.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("MUJOCO_GL", "custom")
    accelerator.prepare_platform()
    assert accelerator.os.environ["MUJOCO_GL"] == "custom"


def test_mps_noise_uses_private_cpu_generator_and_mps_sync(monkeypatch, torch_stub):
    _, calls = torch_stub
    monkeypatch.setattr(accelerator.platform, "system", lambda: "Darwin")
    runtime = accelerator.Accelerator()
    runtime.noise((1, 50, 32), 1234)
    runtime.synchronize()
    assert ("noise", (1, 50, 32), "cpu", 1234, "cpu", "fp32") in calls
    assert ("transfer", "mps") in calls and "mps_sync" in calls
    assert "cuda_sync" not in calls
    assert runtime.peak_memory() is None
    runtime.reset_peak_memory()
    assert "cuda_reset" not in calls


def test_cuda_keeps_original_generator_and_memory_counters(monkeypatch, torch_stub):
    torch, calls = torch_stub
    monkeypatch.setattr(accelerator.platform, "system", lambda: "Linux")
    runtime = accelerator.Accelerator()
    runtime.noise((1, 50, 32), 1234)
    runtime.synchronize()
    runtime.reset_peak_memory()
    assert ("noise", (1, 50, 32), "cuda", 1234, "cuda", "fp32") in calls
    assert not any(isinstance(c, tuple) and c[0] == "transfer" for c in calls)
    assert "cuda_sync" in calls and "cuda_reset" in calls
    assert runtime.peak_memory() == 123
    assert not torch.backends.cuda.matmul.allow_tf32


def test_mps_rejects_unsupported_precision_and_unavailable_backend(monkeypatch, torch_stub):
    torch, _ = torch_stub
    monkeypatch.setattr(accelerator.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="fp32"):
        accelerator.Accelerator("mps", "fp16")
    with pytest.raises(RuntimeError, match="Use --device mps"):
        accelerator.Accelerator("cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        accelerator.Accelerator("mps")


def test_intel_mac_rejected_before_model_import(monkeypatch):
    monkeypatch.setattr(accelerator.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(accelerator.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        accelerator.prepare_platform()


def test_platform_specific_dependency_lock():
    assert dependency_lock_path("Darwin").name == "macos.lock.txt"
    assert dependency_lock_path("Linux").name == "linux.lock.txt"


@pytest.mark.parametrize("device", ["auto", "mps", "cuda"])
def test_cli_forwards_requested_device(monkeypatch, device, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "run", lambda cfg, output, selected, limit: calls.append(selected))
    config = Path(__file__).resolve().parents[1] / "configs/smoke.json"
    cli.main(["run", str(config), "--output", str(tmp_path), "--device", device])
    assert calls == [device]
