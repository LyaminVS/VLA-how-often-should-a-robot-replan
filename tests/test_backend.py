import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from vla_replan import backend
from vla_replan.config import Episode


def test_libero_adapter_pins_reset_index_and_camera_orientation(monkeypatch):
    """Emulate upstream's increment-on-reset contract without installing the simulator."""

    class FakeLibero:
        def __init__(self, **kwargs):
            self._init_states = np.array([[0.0, 1.0], [2.0, 3.0]])
            self.init_state_id = 0
            self.task_id = kwargs["task_id"]
            self.task_description = "Instruction from environment"

        def reset(self, seed):
            selected = self.init_state_id
            self.init_state_id += 1
            image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
            return {
                "pixels": {"image": image, "image2": image.copy()},
                "robot_state": {
                    "eef": {"pos": np.array([selected, 0.0, 0.0])},
                    "gripper": {"qpos": np.zeros(2)},
                },
            }, {}

        def close(self):
            pass

    suite = SimpleNamespace(n_tasks=1, get_task=lambda i: SimpleNamespace(name="test_task"))
    benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_object": lambda: suite})
    monkeypatch.setattr(backend, "check_linux", lambda: None)
    monkeypatch.setitem(sys.modules, "libero.libero", SimpleNamespace(benchmark=benchmark))
    monkeypatch.setitem(sys.modules, "lerobot.envs.libero", SimpleNamespace(LiberoEnv=FakeLibero))
    config = SimpleNamespace(control_hz=20, settle_steps=10)
    env = backend.LiberoEnvironment(config)
    episode = Episode("libero_object", "test_task", 1, 11, 1, 10)
    a, a_info = env.reset(episode)
    b, b_info = env.reset(replace(episode, k=50, policy_seed=29))
    assert a.eef == b.eef == [1.0, 0.0, 0.0]
    assert a_info == b_info
    np.testing.assert_array_equal(a.frame[:, :4], a.payload["raw"]["pixels"]["image"][::-1, ::-1])
    assert a.payload["instruction"] == "Instruction from environment"
    env.close()
