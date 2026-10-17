import json
from dataclasses import replace

import imageio.v2 as imageio
import numpy as np
import pytest

from vla_replan.config import Episode
from vla_replan.video import EpisodeVideo, compare_videos


def test_streamed_video_and_paired_comparison(tmp_path):
    episode = Episode("libero_object", "TEST_FRAMES_NOT_ROBOT_RESULTS", 0, 11, 1, 10)
    frame = np.zeros((256, 512, 3), dtype=np.uint8)
    frame[:, :256, 1] = 120
    frame[:, 256:, 2] = 150
    paths = []
    for k in (1, 10):
        current = replace(episode, k=k)
        directory = tmp_path / f"k{k}"
        directory.mkdir()
        paths.append(directory)
        recorder = EpisodeVideo(directory / "video.mp4", current, fps=20)
        for step in range(3):
            recorder(
                frame,
                {
                    "step": step,
                    "sim_time_s": step / 20,
                    "replanned": step % k == 0,
                    "model_calls": step // k + 1,
                },
                None,
            )
        recorder(frame, {"step": 3, "sim_time_s": 0.15, "replanned": False, "model_calls": 1}, k == 1)
        recorder.close()
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "pair_id": current.pair_id,
                    "config_fingerprint": "fixture",
                    "initial_state_sha256": "fixture",
                    "initial_observation_sha256": "fixture",
                }
            )
        )
    output = tmp_path / "comparison.mp4"
    compare_videos(*paths, output)
    reader = imageio.get_reader(str(output))
    try:
        assert reader.get_meta_data()["fps"] == 20
        assert reader.get_data(0).shape == (512, 1536, 3)
        assert reader.count_frames() == 23
    finally:
        reader.close()
    with pytest.raises(FileExistsError):
        compare_videos(*paths, output)
