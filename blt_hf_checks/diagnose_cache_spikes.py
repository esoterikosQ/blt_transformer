"""Recheck the two anomalous A100 cache steps with branch and stage timings.

This diagnostic replays reference-generated prefixes from an immutable probe
report. It does not alter model weights or the normal evaluation backend.
"""

import argparse
import gc
import json
import time
from pathlib import Path


DEFAULT_TARGETS = ((2290, 3), (297, 27))


def first_changed_patch(old, new):
    for index, (left, right) in enumerate(zip(old, new)):
        if left != right:
            return index
    return min(len(old), len(new)) if len(old) != len(new) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('At least two repeats are needed to separate first-use from recurring cost')

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from blt_hf.cache.global_reuse import GlobalPrefixReuse
    from blt_hf.checkpoint import resolve_checkpoint
    from blt_hf.data_adapter import dataset_split_path, encode_prompt, read_tsv
    from blt_hf.manifest import sha256_file, write_json
    from blt_hf.model import load_model

    root = Path(__file__).resolve().parents[1]

    def inside(path):
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f'Path must be inside project: {path}')
        return resolved

    report_path, output_path = inside(args.reference_report), inside(args.output)
    checkpoint_path, model_path = inside(args.checkpoint), inside(args.model_path)
    conversion_path = inside(args.conversion_report)
    if output_path.exists():
        raise ValueError(f'Output already exists: {output_path}')
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete' or report['probe_token_parity'] != 'passed':
        raise ValueError('Reference report must be a completed, parity-passing probe')
    if not report['reuse_decoder'] or report['data']['dataset'] != 'native' or report['data']['split'] != 'val':
        raise ValueError('Expected the native validation decoder-reuse report')
    for key, path in (('model_code_sha256', root / 'blt_hf/patched/modeling_blt.py'),
                      ('reuse_code_sha256', root / 'blt_hf/cache/global_reuse.py')):
        if sha256_file(path) != report[key]:
            raise ValueError(f'{key} differs from the original probe')
    checkpoint, metadata = resolve_checkpoint(checkpoint_path, verify_optimizer=False)
    if (str(checkpoint.relative_to(root)) != report['checkpoint']['path'] or
            metadata['files']['model.safetensors'] != report['checkpoint']['model_sha256']):
        raise ValueError('Checkpoint differs from the original probe')
    tsv = dataset_split_path(root / 'data/Preprocessed', 'native', 'val')
    if sha256_file(tsv) != report['data']['tsv_sha256']:
        raise ValueError('Validation TSV differs from the original probe')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required')

    model = load_model(model_path, conversion_path, attention_mode='osc', device='cuda')
    model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows = read_tsv(tsv)
    cases = {int(case['name'].split(':', 1)[0]): case for case in report['cases']}

    def next_id(logits):
        values = logits.detach().float()[0, -1].clone()
        values[[0, 1, 3]] = -float('inf')
        return int(values.argmax())

    def timed_reference(tokens):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = model(input_ids=tokens, use_cache=False)
        end.record()
        end.synchronize()
        return output, start.elapsed_time(end)

    def run_trial(case, target_step, mode, profile):
        index = int(case['name'].split(':', 1)[0])
        row = rows[index]
        if f'{index}:{row.sample_id}' != case['name']:
            raise ValueError(f'Sample identity changed: {index}')
        ids = list(encode_prompt(tokenizer, row.source, sample_id=row.sample_id))
        if len(ids) != case['observations'][0]['input_length']:
            raise ValueError(f'Prompt length changed: {index}')
        reuse = GlobalPrefixReuse(model, reuse_decoder=mode)
        with torch.inference_mode():
            for step in range(1, target_step):
                if len(ids) != case['observations'][step - 1]['input_length']:
                    raise ValueError(f'Prefix length changed: {index}, step {step}')
                prefix = torch.tensor([ids], dtype=torch.long, device='cuda')
                # Replay the original bench's alternating reference/candidate
                # order so allocator and kernel warmup follow the same path.
                def replay_candidate():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    reuse.run(prefix)
                    end.record()
                    end.synchronize()

                if (step - 1) % 2:
                    replay_candidate()
                    timed_reference(prefix)
                else:
                    timed_reference(prefix)
                    replay_candidate()
                ids.append(case['observations'][step - 1]['reference_next_id'])
            expected = case['observations'][target_step - 1]
            if len(ids) != expected['input_length']:
                raise ValueError(f'Target prefix length changed: {index}')
            tokens = torch.tensor([ids], dtype=torch.long, device='cuda')
            before = list(reuse.starts)
            # Both original anomaly steps ran reference before candidate.
            if (target_step - 1) % 2 == 0:
                reference, reference_ms = timed_reference(tokens)
            else:
                reference, reference_ms = None, None
            stage_events = {}
            stage_cpu_start = {}
            stage_cpu_ms = {}
            handles = []
            if profile:
                modules = {'patcher': model.model.patcher,
                           'local_encoder': model.model.local_encoder,
                           'global_transformer': model.model.global_transformer,
                           'local_decoder': model.model.local_decoder}

                def pre_hook(name):
                    def record(_module, _inputs):
                        event = torch.cuda.Event(enable_timing=True)
                        event.record()
                        stage_events[name] = [event]
                        stage_cpu_start[name] = time.perf_counter()
                    return record

                def post_hook(name):
                    def record(_module, _inputs, _output):
                        event = torch.cuda.Event(enable_timing=True)
                        event.record()
                        stage_events[name].append(event)
                        stage_cpu_ms[name] = 1000 * (time.perf_counter() - stage_cpu_start[name])
                    return record

                for name, module in modules.items():
                    handles.append(module.register_forward_pre_hook(pre_hook(name)))
                    handles.append(module.register_forward_hook(post_hook(name)))
            gc_starts = []
            gc_ms = []

            def gc_callback(phase, _info):
                if phase == 'start':
                    gc_starts.append(time.perf_counter())
                elif gc_starts:
                    gc_ms.append(1000 * (time.perf_counter() - gc_starts.pop()))

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            memory_before = {'allocated': torch.cuda.memory_allocated(),
                             'reserved': torch.cuda.memory_reserved()}
            gc_count_before = gc.get_count()
            torch.cuda.reset_peak_memory_stats()
            gc.callbacks.append(gc_callback)
            try:
                cpu_start = time.perf_counter()
                start.record()
                candidate, count, skipped, decoder_reused = reuse.run(tokens)
                end.record()
                end.synchronize()
                wall_ms = 1000 * (time.perf_counter() - cpu_start)
            finally:
                gc.callbacks.remove(gc_callback)
                for handle in handles:
                    handle.remove()
            after = list(reuse.starts)
            if reference is None:
                reference, reference_ms = timed_reference(tokens)
            matched = next_id(candidate.logits) == next_id(reference.logits) == expected['reference_next_id']
            result = {'mode_reuse_decoder': mode, 'profile_stages': profile,
                      'sample_index': index, 'step': target_step, 'input_length': len(ids),
                      'last_input_id': ids[-1], 'patch_starts_before': before,
                      'patch_starts_after': after, 'first_changed_patch': first_changed_patch(before, after),
                      'skipped_global': skipped, 'reused_decoder': decoder_reused,
                      'reused_closed_patches': count, 'next_id_equal': matched,
                      'reference_cuda_event_ms': reference_ms,
                      'cuda_event_ms': start.elapsed_time(end), 'wall_ms': wall_ms,
                      'gc_ms': gc_ms, 'stages': {
                          name: {'cuda_event_ms': events[0].elapsed_time(events[1]),
                                 'host_ms': stage_cpu_ms[name]}
                          for name, events in stage_events.items()
                      },
                      'memory_bytes': {'before': memory_before,
                                       'after': {'allocated': torch.cuda.memory_allocated(),
                                                 'reserved': torch.cuda.memory_reserved()},
                                       'peak_allocated': torch.cuda.max_memory_allocated()},
                      'gc_count_before': gc_count_before, 'gc_count_after': gc.get_count()}
            if not matched:
                raise AssertionError(f'Next byte changed at sample {index}, step {target_step}')
            return result

    trials = []
    for repetition in range(args.repeats):
        modes = (True, False) if repetition % 2 == 0 else (False, True)
        for index, step in DEFAULT_TARGETS:
            case = cases[index]
            if case['name'] not in report['data']['sample_ids'] or len(case['observations']) < step:
                raise ValueError(f'Reference report lacks sample {index}, step {step}')
            for mode in modes:
                for profile in (False, True):
                    result = run_trial(case, step, mode, profile)
                    expected = case['observations'][step - 1]
                    if (result['skipped_global'] != expected['skipped_global'] or
                            result['reused_decoder'] != (mode and expected['reused_decoder'])):
                        raise AssertionError(f'Cache branch changed at sample {index}, step {step}')
                    result['repetition'] = repetition + 1
                    trials.append(result)
                    print(json.dumps({key: result[key] for key in
                                      ('sample_index', 'step', 'mode_reuse_decoder',
                                       'profile_stages', 'repetition', 'first_changed_patch',
                                       'skipped_global', 'reused_decoder', 'cuda_event_ms', 'wall_ms', 'gc_ms')}),
                          flush=True)
    write_json(output_path, {
        'status': 'complete', 'scope': 'two anomalous steps, replayed prefixes, alternating decoder modes',
        'source_report': str(report_path.relative_to(root)),
        'source_report_sha256': sha256_file(report_path),
        'checkpoint_model_sha256': report['checkpoint']['model_sha256'],
        'validation_tsv_sha256': report['data']['tsv_sha256'],
        'diagnostic_code_sha256': sha256_file(Path(__file__)),
        'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'repeats': args.repeats, 'trials': trials,
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
