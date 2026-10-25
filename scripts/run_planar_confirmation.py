#!/usr/bin/env python3
"""Registered matched continuous one-way target motion, with raw predictions and video."""
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
from vla_replan.core import run_episode,sampling_seed
from moving_target.planar_motion import PlanarMotionEnvironment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(cfg):
    rng=random.Random(cfg['order_seed']);settings=[(state,pos) for state in cfg['base']['initial_states'] for pos in cfg['positions_cm']];rng.shuffle(settings)
    entries=[]
    for state,pos in settings:
        conditions=[(k,motion) for k in cfg['base']['intervals'] for motion in cfg['motions']];rng.shuffle(conditions)
        entries.extend(dict(state=state,k=k,offset_cm=pos,axis=motion['axis'],speed_cm_s=motion['speed_cm_s'],condition_id=motion['id'],seed=11) for k,motion in conditions)
    return entries


def verify(folder, entry, fingerprint):
    evidence = json.loads((folder / 'evidence.json').read_text())
    assert evidence['config_fingerprint'] == fingerprint
    for name, expected in evidence['files'].items():
        assert sha(folder / name) == expected, (folder, name)
    summary = json.loads((folder / 'summary.json').read_text())
    assert all(summary[k] == entry[k] for k in ('state','k','offset_cm','axis','speed_cm_s','condition_id'))
    return summary


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
        lines = [f"Drive {e['speed_cm_s']:+}cm/s axis{'xy'[e['axis']]} | K={e['k']} | start{e['state']}",
                 f"Offset {e['offset_cm']}cm | t={row['sim_time_s']:.2f}s | calls={row['model_calls']}",
                 'NEW POLICY OBSERVATION' if row['replanned'] else 'Executing the current action block',
                 'Bilateral grasp: drive permanently released' if row['motion']['drive_released'] else 'Smooth one-way motion; no return']
        m=row['motion'];delta=[100*(a-b) for a,b in zip(m['object_pos'][:2],m['object_initial_pos'][:2])]
        lines.append(f'Actual can delta: x={delta[0]:+.1f}cm y={delta[1]:+.1f}cm')
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
    assert len(cfg['motions'])==5 and len(cfg['positions_cm'])==2
    assert cfg['base']['initial_states']==[30,31,32]
    raw = copy.deepcopy(cfg['base'])
    raw['tasks'] = tuple(Task(**t) for t in raw['tasks'])
    for field in ('intervals', 'initial_states', 'policy_seeds'):
        raw[field] = tuple(raw[field])
    base = Experiment(**raw)
    base.validate()
    entries = make_plan(cfg)
    assert len(entries)==150
    out = Path(out)
    env = PlanarMotionEnvironment(base)
    sources = [Path(__file__), ROOT / 'moving_target/planar_motion.py', ROOT / 'moving_target/linear_motion.py', ROOT / 'moving_target/single_shift.py', ROOT / 'moving_target/environment.py',
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
            manifest = {'data_kind': 'heldout_planar_confirmation', 'created_utc': datetime.now(timezone.utc).isoformat(),
                        'config': cfg, 'config_fingerprint': fingerprint, 'sources': hashes, 'plan': entries,
                        'planned': len(entries), 'status': 'registered_before_policy_load', 'runtime': runtime_metadata()}
            atomic_json(path, manifest)
        from vla_replan.backend import SmolVLAPolicy
        policy = SmolVLAPolicy(base, 'mps')
        if cfg['noise_mode']=='episode_common':
            native=policy
            class CommonNoise:
                def __getattr__(self,name):return getattr(native,name)
                def predict(self,obs,requested_seed):return native.predict(obs,sampling_seed(11,0))
            policy=CommonNoise()
        else:
            assert cfg['noise_mode']=='time_indexed'
        manifest.update(status='collecting', hardware=policy.runtime)
        atomic_json(path, manifest)
        warm = Episode(base.tasks[0].suite, base.tasks[0].name, entries[0]['state'], 11, 10, 280)
        env.speed_cm_s = 0
        obs, _ = env.reset(warm)
        for i in range(base.warmup_calls):
            policy.predict(obs, i)
        identities = {}
        count = 0
        start = time.perf_counter()
        for index, entry in enumerate(entries):
            e = Episode(base.tasks[0].suite, base.tasks[0].name, entry['state'], 11, entry['k'], 280)
            label = f"pos_{entry['offset_cm'][0]:+}_{entry['offset_cm'][1]:+}__axis{entry['axis']}__v{entry['speed_cm_s']:+}"
            folder = out / 'episodes' / (e.id + '__' + label)
            if folder.exists():
                summary = verify(folder, entry, fingerprint)
                first = json.loads((folder / 'chunks.jsonl').read_text().splitlines()[0])['actions']
                identity = (summary['initial_scene_sha256'], summary['initial_observation_sha256'], first)
                pair=(entry['state'],tuple(entry['offset_cm']))
                if pair in identities:
                    assert identities[pair] == identity
                identities[pair] = identity
                continue
            if limit is not None and count >= limit:
                break
            env.offset_cm=entry['offset_cm'];env.axis=entry['axis'];env.speed_cm_s=entry['speed_cm_s']
            attempt = out / 'attempts' / (folder.name + '__' + str(time.time_ns()))
            attempt.mkdir(parents=True)
            video = None
            final = {}
            def record(frame, row, outcome):
                nonlocal final
                row['motion'] = copy.deepcopy(env.telemetry())
                row['speed_cm_s'] = entry['speed_cm_s']
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
                assert final['pose_writes'] == 0
                assert all(r['motion']['pose_writes'] == 0 for r in rows)
                assert entry['speed_cm_s'] or final['velocity_writes'] == 0
                release = [r['motion']['velocity_writes'] for r in rows if r['motion']['drive_released']]
                assert not release or len(set(release)) == 1
                identity = (summary['initial_scene_sha256'], summary['initial_observation_sha256'], chunks[0]['actions'])
                pair=(entry['state'],tuple(entry['offset_cm']))
                if pair in identities:
                    assert identities[pair] == identity
                identities[pair] = identity
                for c in chunks:
                    c['requested_sampling_seed']=c['sampling_seed']
                    if cfg['noise_mode']=='episode_common':c['sampling_seed']=sampling_seed(11,0)
                actual = np.asarray([r['action'] for r in rows])
                for j, chunk in enumerate(chunks):
                    end = chunks[j + 1]['step'] if j + 1 < len(chunks) else len(rows)
                    assert np.array_equal(actual[chunk['step']:end], np.clip(np.array(chunk['actions'])[:end-chunk['step']], -1, 1))
                summary.update(data_kind='heldout_planar_confirmation', offset_cm=entry['offset_cm'],axis=entry['axis'],speed_cm_s=entry['speed_cm_s'],condition_id=entry['condition_id'],noise_mode=cfg['noise_mode'], condition=label,
                    config_fingerprint=fingerprint, pose_writes=final['pose_writes'],
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
                print(f"[{completed}/150] state={entry['state']} speed={entry['speed_cm_s']:+}cm/s K={entry['k']} success={summary['success']} calls={len(chunks)} wall={summary['episode_wall_s']:.1f}s elapsed={elapsed/60:.1f}min", flush=True)
                atomic_json(out / 'progress.json', {'completed': completed, 'planned': 150, 'last_episode': folder.name, 'elapsed_s': elapsed})
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
        manifest.update(status='complete' if completed == 150 else 'partial', completed=completed)
        atomic_json(path, manifest)
        print(json.dumps({'completed': completed, 'planned': 150}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.config, args.output, limit=args.limit)
