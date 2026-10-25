#!/usr/bin/env python3
"""Registered paired one-shot target relocations, with raw predictions and video."""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / '.cache/libero'))
os.environ.setdefault('MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'egl')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from vla_replan.config import Experiment, Episode, Task, digest
from vla_replan.storage import atomic_json, write_jsonl, exclusive_run, runtime_metadata
from vla_replan.core import run_episode
from moving_target.single_shift import SingleShiftEnvironment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(cfg):
    rng = random.Random(cfg['order_seed'])
    states = list(cfg['base']['initial_states'])
    rng.shuffle(states)
    entries = []
    for state in states:
        conditions = [(k, dx) for k in cfg['base']['intervals'] for dx in cfg['shifts_cm']]
        rng.shuffle(conditions)
        entries.extend({'state': state, 'k': k, 'shift_cm': dx, 'seed': 11} for k, dx in conditions)
    return entries


def verify(folder, entry, fingerprint):
    evidence = json.loads((folder / 'evidence.json').read_text())
    assert evidence['config_fingerprint'] == fingerprint
    for name, expected in evidence['files'].items():
        assert sha(folder / name) == expected, (folder, name)
    summary = json.loads((folder / 'summary.json').read_text())
    assert all(summary[k] == entry[k] for k in ('state', 'k', 'shift_cm'))
    return summary


def validate_geometry(event):
    # Table support contact is expected. No new deep overlap with other objects
    # or robot geometry may be silently converted into an apparent policy failure.
    before = {c['other_geom'] for c in event['contacts_before'] if c['distance_m'] < -.001}
    invalid = [c for c in event['contacts_after'] if c['distance_m'] < -.001
               and 'table' not in (c['other_body'] or '').lower()
               and c['other_geom'] not in before]
    if invalid:
        raise RuntimeError('Invalid relocation geometry: ' + json.dumps(invalid))


class Video:
    def __init__(self, path, entry):
        import imageio.v2 as imageio
        self.writer = imageio.get_writer(str(path), fps=20, codec='libx264', quality=7,
            macro_block_size=16, ffmpeg_log_level='error', ffmpeg_params=['-threads', '1'])
        self.entry = entry

    def __call__(self, frame, row, outcome):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        canvas = Image.new('RGB', (768, 576), '#FFFFFF')
        canvas.paste(Image.fromarray(frame).resize((768, 384)), (0, 192))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=17)
        e = self.entry
        lines = [f"Single shift {e['shift_cm']:+} cm | K={e['k']} | state {e['state']}",
                 f"Simulation t={row['sim_time_s']:.2f}s | calls={row['model_calls']}",
                 'NEW POLICY OBSERVATION' if row['replanned'] else 'Executing the current action block',
                 'SHIFT APPLIED; ordinary physics thereafter' if row['motion']['shift_applied'] else 'Before the single shift at 1.025s']
        if outcome is not None:
            lines.append('SUCCESS' if outcome else 'FAILURE / time limit')
        for i, text in enumerate(lines):
            draw.text((16, 12 + i * 32), text, font=font, fill='#171717')
        self.writer.append_data(np.asarray(canvas))

    def close(self):
        self.writer.close()


def run(config_path, out, preflight=False, limit=None):
    import numpy as np
    cfg = json.loads(Path(config_path).read_text())
    assert cfg['base']['intervals'] == [1, 5, 10, 25, 50]
    assert cfg['shifts_cm'] == [-8, -4, 0, 4, 8]
    assert cfg['base']['initial_states'] == list(range(17, 25))
    raw = copy.deepcopy(cfg['base'])
    raw['tasks'] = tuple(Task(**t) for t in raw['tasks'])
    for field in ('intervals', 'initial_states', 'policy_seeds'):
        raw[field] = tuple(raw[field])
    base = Experiment(**raw)
    base.validate()
    entries = make_plan(cfg)
    assert len(entries) == 200
    out = Path(out)
    env = SingleShiftEnvironment(base, shift_time_s=cfg['shift_time_s'])
    if preflight:
        checks = []
        for state in base.initial_states:
            episode = Episode(base.tasks[0].suite, base.tasks[0].name, state, 11, 10, 280)
            env.shift_cm = 0
            env.reset(episode)
            sim = env._domain.sim
            qpos = sim.data.qpos.copy()
            for dx in cfg['shifts_cm']:
                sim.data.qpos[:] = qpos
                sim.forward()
                before = env.object_contacts()
                sim.data.qpos[env._qadr] += dx / 100
                sim.forward()
                event = {'contacts_before': before, 'contacts_after': env.object_contacts()}
                validate_geometry(event)
                checks.append({'state': state, 'shift_cm': dx, 'position': sim.data.body_xpos[env._body_id].tolist(), **event})
        env.close()
        atomic_json(out / 'preflight.json', {'endpoint_geometry_valid': True, 'checks': checks,
                    'limitation': 'Initial geometry only; the same overlap check is repeated at the actual event in every rollout.'})
        print('Preflight: all 40 initial-state/shift endpoints valid', flush=True)
        return
    sources = [Path(__file__), ROOT / 'moving_target/single_shift.py', ROOT / 'moving_target/environment.py',
               ROOT / 'vla_replan/backend.py', ROOT / 'vla_replan/core.py', ROOT / 'vla_replan/accelerator.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    fingerprint = digest(cfg)
    with exclusive_run(out):
        path = out / 'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            assert manifest['sources'] == hashes and manifest['config_fingerprint'] == fingerprint
            assert manifest['plan'] == entries
        else:
            manifest = {'data_kind': 'single_position_shift', 'created_utc': datetime.now(timezone.utc).isoformat(),
                        'config': cfg, 'config_fingerprint': fingerprint, 'sources': hashes, 'plan': entries,
                        'planned': len(entries), 'status': 'registered_before_policy_load', 'runtime': runtime_metadata()}
            atomic_json(path, manifest)
        from vla_replan.backend import SmolVLAPolicy
        policy = SmolVLAPolicy(base, 'mps')
        manifest.update(status='collecting', hardware=policy.runtime)
        atomic_json(path, manifest)
        warm = Episode(base.tasks[0].suite, base.tasks[0].name, entries[0]['state'], 11, 10, 280)
        env.shift_cm = 0
        obs, _ = env.reset(warm)
        for i in range(base.warmup_calls):
            policy.predict(obs, i)
        identities = {}
        count = 0
        start = time.perf_counter()
        for index, entry in enumerate(entries):
            e = Episode(base.tasks[0].suite, base.tasks[0].name, entry['state'], 11, entry['k'], 280)
            label = f"shift_{entry['shift_cm']:+}cm"
            folder = out / 'episodes' / (e.id + '__' + label)
            if folder.exists():
                summary = verify(folder, entry, fingerprint)
                first = json.loads((folder / 'chunks.jsonl').read_text().splitlines()[0])['actions']
                identity = (summary['initial_state_sha256'], summary['initial_observation_sha256'], first)
                if entry['state'] in identities:
                    assert identities[entry['state']] == identity
                identities[entry['state']] = identity
                continue
            if limit is not None and count >= limit:
                break
            env.shift_cm = entry['shift_cm']
            attempt = out / 'attempts' / (folder.name + '__' + str(time.time_ns()))
            attempt.mkdir(parents=True)
            video = None
            final = {}
            def record(frame, row, outcome):
                nonlocal final
                row['motion'] = copy.deepcopy(env.telemetry())
                row['shift_cm'] = entry['shift_cm']
                if outcome is not None:
                    final = row['motion']
                if video:
                    video(frame, row, outcome)
            tick = time.perf_counter()
            try:
                video = Video(attempt / 'video.mp4', entry)
                summary, rows, chunks = run_episode(env, policy, e, chunk_size=50, control_hz=20, record=record)
                video.close()
                video = None
                event = final['shift_event']
                assert event is not None
                validate_geometry(event)
                expected = 1 if entry['shift_cm'] else 0
                assert final['pose_writes'] == expected
                assert all(r['motion']['pose_writes'] in (0, expected) for r in rows)
                assert np.allclose(np.array(event['pose_after']) - event['pose_before'],
                                   [entry['shift_cm'] / 100, 0, 0, 0, 0, 0, 0], atol=1e-12)
                assert event['velocity_before'] == event['velocity_after']
                assert cfg['shift_time_s'] <= event['actual_sim_time_s'] + 1e-10 <= cfg['shift_time_s'] + .00201
                identity = (summary['initial_state_sha256'], summary['initial_observation_sha256'], chunks[0]['actions'])
                if entry['state'] in identities:
                    assert identities[entry['state']] == identity
                identities[entry['state']] = identity
                actual = np.asarray([r['action'] for r in rows])
                for j, chunk in enumerate(chunks):
                    end = chunks[j + 1]['step'] if j + 1 < len(chunks) else len(rows)
                    assert np.array_equal(actual[chunk['step']:end], np.clip(np.array(chunk['actions'])[:end-chunk['step']], -1, 1))
                summary.update(data_kind='single_position_shift', shift_cm=entry['shift_cm'], condition=label,
                    config_fingerprint=fingerprint, shift_event=event, pose_writes=final['pose_writes'],
                    motion_final=final, episode_wall_s=time.perf_counter()-tick)
                write_jsonl(attempt / 'steps.jsonl', rows)
                write_jsonl(attempt / 'chunks.jsonl', chunks)
                atomic_json(attempt / 'summary.json', summary)
                atomic_json(attempt / 'evidence.json', {'config_fingerprint': fingerprint, 'sources': hashes,
                    'files': {name: sha(attempt / name) for name in ('summary.json', 'steps.jsonl', 'chunks.jsonl', 'video.mp4')}})
                verify(attempt, entry, fingerprint)
                folder.parent.mkdir(parents=True, exist_ok=True)
                attempt.rename(folder)
                count += 1
                completed = len(list((out / 'episodes').glob('*/summary.json')))
                elapsed = time.perf_counter() - start
                print(f"[{completed}/200] state={entry['state']} dx={entry['shift_cm']:+}cm K={entry['k']} success={summary['success']} calls={len(chunks)} wall={summary['episode_wall_s']:.1f}s elapsed={elapsed/60:.1f}min", flush=True)
                atomic_json(out / 'progress.json', {'completed': completed, 'planned': 200, 'last_episode': folder.name, 'elapsed_s': elapsed})
            except BaseException:
                atomic_json(attempt / 'error.json', {'entry': entry, 'traceback': traceback.format_exc()})
                manifest.update(status='failed', failed_entry=entry)
                atomic_json(path, manifest)
                raise
            finally:
                if video:
                    video.close()
        env.close()
        completed = len(list((out / 'episodes').glob('*/summary.json')))
        manifest.update(status='complete' if completed == 200 else 'partial', completed=completed)
        atomic_json(path, manifest)
        print(json.dumps({'completed': completed, 'planned': 200}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.preflight_only, args.limit)
