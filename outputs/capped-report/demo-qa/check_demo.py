"""Independent bounded-CPU decoding and frame/log checks for the final demo."""

import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path('/home/kaban/vla-replan')
REPORT = ROOT / 'outputs/capped-report'
OUT = REPORT / 'demo-qa'
OUT.mkdir(exist_ok=True)
VIDEO = REPORT / 'demo.mp4'
SIDECAR = REPORT / 'demo.json'
EXPECTED_SHA = 'a528ca21551f9e267dad72d4fcc57ebe5dd7871a1e219b3cea31e5a012cf45fa'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def font(size, bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf', size)


meta = read_json(SIDECAR)
assert sha(VIDEO) == meta['sha256'] == EXPECTED_SHA
assert (meta['frames'], meta['fps'], meta['width'], meta['height']) == (901, 20, 2304, 704)
assert len(meta['sections']) == 3
assert meta['sections'][0]['title_start_frame'] == 0
assert meta['sections'][-1]['end_frame_exclusive'] == meta['frames']
for path, digest in meta['input_sha256'].items():
    assert sha(path) == digest, path

selected, sections, sample_metadata = {}, [], []
for index, section in enumerate(meta['sections']):
    if index:
        assert section['title_start_frame'] == meta['sections'][index - 1]['end_frame_exclusive']
    assert section['comparison_start_frame'] - section['title_start_frame'] == 40
    assert [row['method'] for row in section['episodes']] == ['fixed_10', 'dvac_cap_10', 'dvac_cap_50']
    inputs = []
    for entry in section['episodes']:
        matches = [Path(path).parent for path in meta['input_sha256'] if path.endswith('/' + entry['episode_id'] + '/summary.json')]
        assert len(matches) == 1
        directory = matches[0]
        summary = read_json(directory / 'summary.json')
        logs = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines() if line.strip()]
        for key in ('episode_id', 'success', 'steps', 'model_calls'):
            assert summary[key] == entry[key], (key, entry['episode_id'])
        assert len(logs) == entry['steps']
        assert sum(row['replanned'] for row in logs) == entry['model_calls']
        assert summary['state'] == 0 and summary['policy_seed'] == 11
        assert summary['pair_id'] == section['pair_id']
        assert entry['decoded_source_frames'] == entry['steps'] + 20
        inputs.append({'directory': directory, 'summary': summary, 'logs': logs})
    assert len({item['summary']['instruction'] for item in inputs}) == 1
    assert section['end_frame_exclusive'] - section['comparison_start_frame'] == max(row['steps'] for row in section['episodes']) + 20
    adaptive_step = next(row['step'] for row in inputs[1]['logs'] if row['step'] >= 2 and row['replanned'])
    wanted = {
        'title': section['title_start_frame'],
        'first_motion': section['comparison_start_frame'] + 1,
        'fixed_replan': section['comparison_start_frame'] + 10,
        'adaptive_replan': section['comparison_start_frame'] + adaptive_step,
        'first_terminal': section['comparison_start_frame'] + min(row['steps'] for row in section['episodes']),
        'terminal': section['end_frame_exclusive'] - 1,
    }
    for kind, frame in wanted.items():
        selected.setdefault(frame, []).append((index, kind))
    sections.append({'section': section, 'inputs': inputs, 'wanted': wanted})

metrics = {'decoded_frames': 0, 'compared_panel_frames': 0, 'compared_footer_frames': 0,
           'source_panel_max_mean_absolute_error': 0.0, 'source_panel_max_root_mean_square_error': 0.0,
           'footer_max_mean_absolute_error': 0.0, 'held_panel_frames_compared': 0}
per_section = []
frame_number = 0
with imageio.get_reader(str(VIDEO), input_params=['-threads', '1']) as final_reader:
    metadata = final_reader.get_meta_data()
    assert tuple(metadata['size']) == (2304, 704)
    assert math.isclose(metadata['fps'], 20, abs_tol=1e-6)
    iterator = iter(final_reader)
    for sec_index, data in enumerate(sections):
        section, inputs = data['section'], data['inputs']
        section_stats = {'task': section['task'], 'instruction': inputs[0]['summary']['instruction'],
                         'sample_frames': data['wanted'], 'episodes': section['episodes'],
                         'max_panel_mean_absolute_error': 0.0, 'max_footer_mean_absolute_error': 0.0}
        with ExitStack() as stack:
            source_readers = [stack.enter_context(imageio.get_reader(str(item['directory'] / 'video.mp4'),
                                                                    input_params=['-threads', '1'])) for item in inputs]
            source_iterators = [iter(reader) for reader in source_readers]
            current_source = [None] * 3
            decoded_sources = [0] * 3
            for reader in source_readers:
                source_metadata = reader.get_meta_data()
                assert tuple(source_metadata['size']) == (768, 512)
                assert math.isclose(source_metadata['fps'], 20, abs_tol=1e-6)
            while frame_number < section['end_frame_exclusive']:
                frame = next(iterator)
                assert frame.shape == (704, 2304, 3)
                sample = {'frame_index': frame_number, 'task': section['task'], 'panels': []}
                if frame_number >= section['comparison_start_frame']:
                    step = frame_number - section['comparison_start_frame']
                    sample['simulation_step'] = step
                    for panel, item in enumerate(inputs):
                        summary, logs = item['summary'], item['logs']
                        if step <= summary['steps']:
                            current_source[panel] = next(source_iterators[panel])
                            decoded_sources[panel] += 1
                        source = current_source[panel]
                        assert source.shape == (512, 768, 3)
                        # Ignore the two-pixel panel separator, and sample every other pixel.
                        actual = frame[128:640:2, panel * 768 + 4:(panel + 1) * 768 - 4:2]
                        expected = source[::2, 4:764:2]
                        difference = actual.astype(np.float32) - expected.astype(np.float32)
                        mae = float(np.abs(difference).mean())
                        rmse = float(np.sqrt(np.square(difference).mean()))
                        metrics['source_panel_max_mean_absolute_error'] = max(metrics['source_panel_max_mean_absolute_error'], mae)
                        metrics['source_panel_max_root_mean_square_error'] = max(metrics['source_panel_max_root_mean_square_error'], rmse)
                        metrics['compared_panel_frames'] += 1
                        section_stats['max_panel_mean_absolute_error'] = max(section_stats['max_panel_mean_absolute_error'], mae)
                        assert mae < 5 and rmse < 18, ('Source panel does not match expected frame', frame_number, panel, mae, rmse)
                        method = section['episodes'][panel]['method']
                        name = 'Fixed K=10' if panel == 0 else 'DVAC cap ' + method.rsplit('_', 1)[1]
                        if step >= summary['steps']:
                            status = 'SUCCESS' if summary['success'] else 'FAILURE'
                            message = f"{status} | {summary['model_calls']} calls | final frame held"
                            color = '#71e4ac' if summary['success'] else '#ff8d91'
                            expected_log = {'terminal': True, 'success': summary['success'], 'model_calls': summary['model_calls']}
                            metrics['held_panel_frames_compared'] += 1
                        else:
                            log = logs[step]
                            chosen = log.get('chosen_k', summary['k'])
                            status = 'REPLAN' if log['replanned'] else 'Executing'
                            message = f"{status} | step {step} | K={chosen} | calls={log['model_calls']}"
                            color = '#f6c550' if log['replanned'] else '#a4b7cf'
                            expected_log = {'terminal': False, 'replanned': log['replanned'], 'chosen_k': chosen,
                                            'model_calls': log['model_calls']}
                        expected_footer = Image.new('RGB', (768, 64), '#101a2c')
                        draw = ImageDraw.Draw(expected_footer)
                        draw.text((16, 4), name, font=font(21, True), fill='#f2f6fc')
                        draw.text((16, 34), message, font=font(18), fill=color)
                        assert draw.textbbox((16, 34), message, font=font(18))[2] < 768
                        footer_diff = frame[640:704, panel * 768 + 4:(panel + 1) * 768 - 4].astype(np.float32) - np.asarray(expected_footer)[:, 4:764].astype(np.float32)
                        footer_mae = float(np.abs(footer_diff).mean())
                        metrics['footer_max_mean_absolute_error'] = max(metrics['footer_max_mean_absolute_error'], footer_mae)
                        section_stats['max_footer_mean_absolute_error'] = max(section_stats['max_footer_mean_absolute_error'], footer_mae)
                        metrics['compared_footer_frames'] += 1
                        assert footer_mae < 4, ('Footer differs from logged state', frame_number, panel, footer_mae)
                        sample['panels'].append({'method': method, 'source_frame_index': min(step, summary['steps']),
                                                 'status_text': message, **expected_log,
                                                 'source_panel_mae': mae, 'footer_mae': footer_mae})
                if frame_number in selected:
                    sample['files'] = []
                    for sample_section, kind in selected[frame_number]:
                        filename = f'section-{sample_section + 1}-{kind}-frame-{frame_number:04d}.png'
                        path = OUT / filename
                        Image.fromarray(frame).save(path)
                        sample['files'].append({'kind': kind, 'file': filename, 'sha256': sha(path)})
                    sample_metadata.append(sample)
                metrics['decoded_frames'] += 1
                frame_number += 1
            for panel, reader_iterator in enumerate(source_iterators):
                for source in reader_iterator:
                    assert source.shape == (512, 768, 3)
                    decoded_sources[panel] += 1
                assert decoded_sources[panel] == section['episodes'][panel]['steps'] + 20
            section_stats['independently_decoded_source_frames'] = decoded_sources
        per_section.append(section_stats)
        print(f"Verified section {sec_index + 1}: {section['task']}", flush=True)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise AssertionError('Unexpected trailing video frames')

assert metrics['decoded_frames'] == 901
assert metrics['compared_panel_frames'] == (901 - 120) * 3
assert sha(VIDEO) == EXPECTED_SHA
qa = {
    'artifact': 'outputs/capped-report/demo.mp4', 'sha256': EXPECTED_SHA,
    'sidecar_sha256': sha(SIDECAR), 'checker_sha256': sha(__file__),
    'resource_limits': {'host': 'home', 'cpu_affinity': [2], 'cpu_quota_percent': 100,
                        'memory_max_bytes': 2 * 1024 ** 3, 'memory_swap_max_bytes': 0, 'decoder_threads': 1},
    'metadata': {'frames': metrics['decoded_frames'], 'fps': metadata['fps'], 'width': metadata['size'][0],
                 'height': metadata['size'][1], 'duration_s': metrics['decoded_frames'] / metadata['fps']},
    'checks': {'all_bound_inputs_unchanged': True, 'all_three_predetermined_tasks_present': True,
               'panel_order_correct': True, 'timeline_contiguous': True, 'all_source_videos_fully_decoded': True,
               'source_panels_match_expected_frames_with_lossy_tolerance': True,
               'footer_markers_counters_outcomes_match_logs_with_lossy_tolerance': True,
               'terminal_hold_uses_correct_source_frame': True},
    'pixel_comparison': {**metrics, 'channel_scale': '0 to 255',
                         'source_mae_limit': 5, 'source_rmse_limit': 18, 'footer_mae_limit': 4,
                         'note': 'MP4 re-encoding is lossy; panel comparisons sample every other pixel and exclude separator columns. Footer text is independently rendered from original logs and checked on every comparison frame.'},
    'sections': per_section, 'selected_frames': sample_metadata,
    'visual_review': {'complete': False, 'findings': []},
    'passed': False,
}
(OUT / 'qa.json').write_text(json.dumps(qa, indent=2) + '\n')
print(json.dumps({'metadata': qa['metadata'], 'pixel_comparison': qa['pixel_comparison'], 'selected_frames': len(sample_metadata)}, indent=2))
