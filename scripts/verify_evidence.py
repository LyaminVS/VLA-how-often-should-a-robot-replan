"""Verify saved predictions, executed K-prefixes, outcomes and optional video alignment.

This is an audit of recorded evidence, not a rerun or a new source of measurements.
Only complete real runs are accepted. Output can be included with a submission.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_replan.analysis import load_results  # noqa: E402
from vla_replan.core import difference, norm, sampling_seed  # noqa: E402
from vla_replan.storage import atomic_json  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(root, decode_video=False):
    root = Path(root)
    cfg, summaries, coverage = load_results(root)
    records = []
    for summary in summaries:
        identity = summary['episode_id']
        episode = root / 'episodes' / identity
        steps = read_jsonl(episode / 'steps.jsonl')
        chunks = read_jsonl(episode / 'chunks.jsonl')
        require(len(steps) == summary['steps'], f'{identity}: step count')
        require(len(chunks) == summary['model_calls'], f'{identity}: chunk count')
        require([c['step'] for c in chunks] == list(range(0, len(steps), summary['k'])),
                f'{identity}: chunk schedule')
        for chunk in chunks:
            require(chunk['sampling_seed'] == sampling_seed(summary['policy_seed'], chunk['step']),
                    f'{identity}: sampling seed')
            require(len(chunk['actions']) == cfg.chunk_size, f'{identity}: prediction horizon')
            require(all(len(a) == 7 and all(math.isfinite(v) for v in a) for a in chunk['actions']),
                    f'{identity}: action shape or nonfinite value')
        clipped = close_events = 0
        revisions, jumps = [], []
        previous = None
        for i, row in enumerate(steps):
            chunk_no, offset = divmod(i, summary['k'])
            chunk = chunks[chunk_no]
            predicted = chunk['actions'][offset]
            action = [max(-1.0, min(1.0, x)) for x in predicted]
            require(row['step'] == i and row['chunk_index'] == offset, f'{identity}: step index')
            require(row['action'] == action, f'{identity}: executed action differs from K-prefix at {i}')
            require(row['replanned'] == (offset == 0), f'{identity}: replan marker')
            require(row['model_calls'] == chunk_no + 1, f'{identity}: call counter')
            require(near(row['sim_time_s'], i / cfg.control_hz), f'{identity}: simulation clock')
            require(near(row['inference_s'], chunk['inference_s'] if offset == 0 else 0),
                    f'{identity}: inference timing')
            revision = None
            if offset == 0 and chunk_no > 0 and summary['k'] < cfg.chunk_size:
                revision = norm(difference(predicted[:3], chunks[chunk_no-1]['actions'][summary['k']][:3]))
            value = row['plan_revision_translation_l2']
            require((value is None and revision is None) or
                    (value is not None and revision is not None and near(value, revision)),
                    f'{identity}: plan revision')
            if revision is not None:
                revisions.append(revision)
            jump = norm(difference(action[:3], previous[:3])) if previous else None
            value = row['action_translation_jump_l2']
            require((value is None and jump is None) or
                    (value is not None and jump is not None and near(value, jump)),
                    f'{identity}: action jump')
            if offset == 0 and jump is not None:
                jumps.append(jump)
            clipped += sum(a != b for a, b in zip(predicted, action, strict=True))
            close_events += int(action[6] > 0 and (previous is None or previous[6] <= 0))
            require(not row['success_after_action'] or i == len(steps)-1,
                    f'{identity}: actions continued after success')
            previous = action
        require(summary['success'] == steps[-1]['success_after_action'], f'{identity}: outcome')
        require(summary['clipped_action_values'] == clipped, f'{identity}: clipping count')
        require(summary['gripper_close_commands'] == close_events, f'{identity}: closure count')
        require(near(summary['inference_s'], sum(c['inference_s'] for c in chunks)),
                f'{identity}: total inference timing')
        require(near(summary['compute_s'], summary['inference_s'] + summary['simulation_s']),
                f'{identity}: total compute timing')
        for key, values in [('mean_plan_revision_translation_l2', revisions),
                            ('mean_replan_jump_translation_l2', jumps)]:
            value = summary[key]
            require((value is None and not values) or
                    (value is not None and values and near(value, sum(values)/len(values))),
                    f'{identity}: {key}')
        frame_count = None
        if decode_video:
            import imageio.v2 as imageio

            reader = imageio.get_reader(str(episode / 'video.mp4'))
            try:
                fps = reader.get_meta_data()['fps']
                require(near(fps, cfg.control_hz), f'{identity}: video FPS')
                frame_count = 0
                for frame in reader:
                    require(frame.shape == (512, 768, 3), f'{identity}: video dimensions')
                    frame_count += 1
                require(frame_count == len(steps) + cfg.control_hz, f'{identity}: video/log alignment')
            finally:
                reader.close()
        hashes = {name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
                  for name in ('summary.json', 'steps.jsonl', 'chunks.jsonl', 'video.mp4')}
        records.append({'episode_id': identity, 'steps': len(steps), 'chunks': len(chunks),
                        'decoded_frames': frame_count, 'sha256': hashes})
    return {'run': str(root), 'coverage': coverage, 'passed': True,
            'checks': ['paired initial states and observations', 'all H=50 predictions',
                       'exact clipped K-prefix actions', 'simulation-step sampling seeds',
                       'replan and model call counters', 'timing sums', 'outcomes',
                       'action clipping counts', 'command revision and jump metrics'],
            'video_decode_checked': decode_video, 'episodes': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--decode-video', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.decode_video)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps({'passed': result['passed'], 'coverage': result['coverage'],
                      'video_decode_checked': args.decode_video, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
