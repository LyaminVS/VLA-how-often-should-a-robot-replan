#!/usr/bin/env python3
"""Independent full-grid verification and paired analysis of one-shot shifts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(run, out):
    run, out = Path(run), Path(out)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed'] == 200
    cfg = manifest['config']
    expected = {(r['state'], r['k'], r['shift_cm']) for r in manifest['plan']}
    assert len(expected) == 200
    episodes, seen, initial, prefixes = [], set(), {}, {}
    action_count = call_count = 0
    actual_event_times = set()
    preexisting_deep_contact_events = 0
    for folder in sorted((run / 'episodes').iterdir()):
        s = json.loads((folder / 'summary.json').read_text())
        e = json.loads((folder / 'evidence.json').read_text())
        assert e['sources'] == manifest['sources']
        assert e['config_fingerprint'] == manifest['config_fingerprint'] == s['config_fingerprint']
        for name, h in e['files'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == h
        key = (s['state'], s['k'], s['shift_cm'])
        assert key in expected and key not in seen
        seen.add(key)
        rows = [json.loads(x) for x in (folder / 'steps.jsonl').read_text().splitlines()]
        chunks = [json.loads(x) for x in (folder / 'chunks.jsonl').read_text().splitlines()]
        assert len(rows) == s['steps'] and len(chunks) == s['model_calls']
        assert [c['step'] for c in chunks] == list(range(0, len(rows), s['k']))
        assert [r['step'] for r in rows] == list(range(len(rows)))
        assert all(r['replanned'] == (r['step'] % s['k'] == 0) for r in rows)
        actions = np.asarray([r['action'] for r in rows])
        for j, c in enumerate(chunks):
            assert c['sampling_seed'] == (s['policy_seed'] * 1000003 + c['step'] * 97409) % (2**63 - 1)
            stop = chunks[j + 1]['step'] if j + 1 < len(chunks) else len(rows)
            assert np.array_equal(actions[c['step']:stop], np.clip(np.asarray(c['actions'])[:stop-c['step']], -1, 1))
        identity = (s['initial_state_sha256'], s['initial_observation_sha256'], chunks[0]['actions'])
        if s['state'] in initial:
            assert initial[s['state']] == identity
        initial[s['state']] = identity
        # All treatments of the same K must have identical complete prefixes
        # before the intervention, including poses, commands and predictions.
        prefix = [(r['action'], r['eef'], r['gripper'], r['motion']['object_qpos'])
                  for r in rows if r['sim_time_s'] < cfg['shift_time_s']]
        pred_prefix = [(c['step'], c['sampling_seed'], c['actions']) for c in chunks
                       if c['step'] / 20 < cfg['shift_time_s']]
        prefix_key = (s['state'], s['k'])
        prefix_identity = (prefix, pred_prefix)
        if prefix_key in prefixes:
            assert prefixes[prefix_key] == prefix_identity, ('prefix mismatch', key)
        prefixes[prefix_key] = prefix_identity
        event = s['shift_event']
        actual_event_times.add(round(event['actual_sim_time_s'], 9))
        assert 1.025 - 1e-10 <= event['actual_sim_time_s'] <= 1.02701
        change = np.asarray(event['pose_after']) - event['pose_before']
        assert np.allclose(change, [s['shift_cm']/100, 0, 0, 0, 0, 0, 0], atol=1e-12)
        assert event['velocity_before'] == event['velocity_after']
        writes = int(s['shift_cm'] != 0)
        assert s['pose_writes'] == writes
        for r in rows:
            motion = r['motion']
            assert motion['pose_writes'] == (writes if motion['shift_applied'] else 0)
            if motion['shift_applied']:
                assert motion['shift_event'] == event
            else:
                assert r['sim_time_s'] < event['actual_sim_time_s'] + 1e-8
        deep_before = {c['other_geom'] for c in event['contacts_before'] if c['distance_m'] < -.001}
        preexisting_deep_contact_events += int(any(c['distance_m'] < -.001 and 'table' not in (c['other_body'] or '').lower() for c in event['contacts_before']))
        assert not [c for c in event['contacts_after'] if c['distance_m'] < -.001
                    and 'table' not in (c['other_body'] or '').lower() and c['other_geom'] not in deep_before]
        fresh = [r for r in rows if r['replanned'] and r['motion']['shift_applied']]
        first = fresh[0] if fresh else None
        contact = s['motion_final']['first_contact_sim_time']
        episodes.append({'episode_id': folder.name, 'state': s['state'], 'k': s['k'], 'shift_cm': s['shift_cm'],
            'success': int(s['success']), 'steps': s['steps'], 'model_calls': s['model_calls'],
            'event_time_s': event['actual_sim_time_s'], 'event_x_before_m': event['pose_before'][0],
            'event_x_after_m': event['pose_after'][0], 'pose_writes': writes,
            'contact_before_shift': event['finger_contact_before_shift'],
            'first_contact_s': contact,
            'first_fresh_policy_observation_s': first['sim_time_s'] if first else None,
            'fresh_observation_delay_s': first['sim_time_s']-event['actual_sim_time_s'] if first else None,
            'final_object_x_m': s['motion_final']['object_pos'][0],
            'source_dir': str(folder.resolve())})
        action_count += len(rows)
        call_count += len(chunks)
    assert seen == expected
    states = cfg['base']['initial_states']
    ks = cfg['base']['intervals']
    lookup = {(r['state'], r['k'], r['shift_cm']): r for r in episodes}
    rng = np.random.default_rng(20261025)
    # One set of state-cluster bootstrap draws is shared by all contrasts.
    draws = rng.integers(0, len(states), (10000, len(states)))
    methods, contrasts = [], []
    for dx in cfg['shifts_cm']:
        for k in ks:
            rs = [lookup[state, k, dx] for state in states]
            y = np.array([r['success'] for r in rs])
            base = np.array([lookup[state, k, 0]['success'] for state in states])
            delta = y - base
            ci = np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()
            methods.append({'shift_cm': dx, 'k': k, 'episodes': len(rs), 'successes': int(y.sum()),
                'success_rate': float(y.mean()), 'outcomes': y.tolist(), 'states': states,
                'mean_model_calls': float(np.mean([r['model_calls'] for r in rs])),
                'mean_fresh_observation_delay_s': float(np.mean([r['fresh_observation_delay_s'] for r in rs if r['fresh_observation_delay_s'] is not None])),
                'success_delta_vs_zero': float(delta.mean()), 'paired_bootstrap95_vs_zero': ci,
                'rescues_vs_zero': int(np.sum((base==0)&(y==1))), 'regressions_vs_zero': int(np.sum((base==1)&(y==0)))})
            if k != 1:
                one = np.array([lookup[state, 1, dx]['success'] for state in states])
                d = y-one
                contrasts.append({'shift_cm': dx, 'k': k, 'reference_k': 1, 'pairs': len(states),
                    'success_difference': float(d.mean()),
                    'paired_bootstrap95': np.quantile(d[draws].mean(axis=1), [.025,.975]).tolist(),
                    'k_only_successes': int(np.sum((one==0)&(y==1))),
                    'k1_only_successes': int(np.sum((one==1)&(y==0)))})
    checks = {'validated_episodes': len(episodes), 'validated_action_steps': action_count,
        'validated_policy_calls': call_count, 'matched_initial_states_images_and_first_predictions': True,
        'same_k_treatments_have_identical_pre_event_trajectory_and_predictions': True,
        'all_actual_actions_equal_clipped_saved_predictions': True,
        'sampling_seed_tied_to_absolute_step': True, 'one_additive_position_write_only': True,
        'zero_control_has_no_pose_writes': True, 'no_velocity_changes': True,
        'no_new_deep_non_table_penetration_at_event': True,
        'actual_event_times_s': sorted(actual_event_times),
        'pre_event_finger_contacts': sum(r['contact_before_shift'] for r in episodes),
        'events_with_preexisting_deep_non_table_contact': preexisting_deep_contact_events}
    result = {'complete': True, 'episodes': 200, 'physical_starts': len(states),
        'policy_seeds': [11], 'ks': ks, 'shifts_cm': cfg['shifts_cm'], 'shift_time_s': cfg['shift_time_s'],
        'methods': methods, 'contrasts_vs_k1': contrasts, 'checks': checks,
        'limitations': cfg['limits'], 'demo_selection': {'state': 17, 'shifts_cm': [0,-4,4,-8,8],
            'ks': ks, 'selection': 'Case rule: smallest registered state index (17), chosen during collection; all K and all five displacements are shown without filtering by outcome.'}}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'episodes.csv', episodes)
    write_csv(out / 'methods.csv', methods)
    write_csv(out / 'contrasts_vs_k1.csv', contrasts)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(checks, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
