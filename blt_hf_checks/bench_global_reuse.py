"""Compare an experimental global-prefix-reuse step with no-cache BF16 forward."""
import argparse
import json
from pathlib import Path
from statistics import mean, median

from .check_patch_causality import fixture_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--reuse-decoder', action='store_true')
    parser.add_argument('--checkpoint', help='Optional immutable fine-tuned checkpoint or best.json')
    parser.add_argument('--dataset', choices=('native', 'korean_learner', 'lang8', 'union'), default='native')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()
    if min(args.steps, args.samples) < 1:
        parser.error('--steps and --samples must be positive')

    import torch
    from transformers import AutoTokenizer
    from blt_hf.cache.global_reuse import GlobalPrefixReuse
    from blt_hf.manifest import sha256_file, write_json
    from blt_hf.model import load_model

    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    model_path = (root / args.model_path).resolve()
    conversion = (root / args.conversion_report).resolve()
    if any(not path.is_relative_to(root) for path in (output, model_path, conversion)) or output.exists():
        raise ValueError('Paths must be inside the project and output must be new')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required')

    model = load_model(model_path, conversion, attention_mode='osc', device='cuda')
    checkpoint_info = None
    if args.checkpoint:
        from safetensors.torch import load_file
        from blt_hf.checkpoint import resolve_checkpoint
        checkpoint_path = (root / args.checkpoint).resolve()
        if not checkpoint_path.is_relative_to(root):
            raise ValueError('Checkpoint must be inside the project')
        checkpoint, metadata = resolve_checkpoint(checkpoint_path, verify_optimizer=False)
        if metadata['run_manifest']['dataset'] != args.dataset:
            raise ValueError('Checkpoint and requested dataset differ')
        model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
        checkpoint_info = {'path': str(checkpoint.relative_to(root)),
                           'model_sha256': metadata['files']['model.safetensors']}
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if args.checkpoint:
        from blt_hf.data_adapter import dataset_split_path, encode_prompt, read_tsv
        tsv = dataset_split_path(root / 'data/Preprocessed', args.dataset, args.split)
        rows = read_tsv(tsv)
        choices = sorted((len(encode_prompt(tokenizer, row.source, sample_id=row.sample_id)), index)
                         for index, row in enumerate(rows))
        if len(choices) < args.samples:
            raise ValueError('Fewer dataset rows than requested benchmark samples')
        selected = [choices[(2 * i + 1) * len(choices) // (2 * args.samples)][1]
                    for i in range(args.samples)]
        fixtures = {f'{index}:{rows[index].sample_id}': encode_prompt(tokenizer, rows[index].source,
                                                                      sample_id=rows[index].sample_id)
                    for index in selected}
        data_info = {'dataset': args.dataset, 'split': args.split,
                     'tsv_sha256': sha256_file(tsv), 'sample_ids': list(fixtures)}
    else:
        all_fixtures = fixture_inputs(tokenizer)
        fixtures = {name: all_fixtures[name]
                    for name in ('korean_short', 'mixed', 'window')}
        data_info = None

    def timed(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, start.elapsed_time(end)

    def best(logits):
        values = logits.detach().float()[0, -1].clone()
        values[[0, 1, 3]] = -float('inf')
        return int(values.argmax())

    cases = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for name, fixture_ids in fixtures.items():
            ids = list(fixture_ids)
            reuse = GlobalPrefixReuse(model, reuse_decoder=args.reuse_decoder)
            observations = []
            for step in range(args.steps):
                tokens = torch.tensor([ids], dtype=torch.long, device='cuda')
                if step % 2:
                    (candidate, count, skipped, decoder_reused), reuse_ms = timed(lambda: reuse.run(tokens))
                    reference, reference_ms = timed(lambda: model(input_ids=tokens, use_cache=False))
                else:
                    reference, reference_ms = timed(lambda: model(input_ids=tokens, use_cache=False))
                    (candidate, count, skipped, decoder_reused), reuse_ms = timed(lambda: reuse.run(tokens))
                ref_id, reused_id = best(reference.logits), best(candidate.logits)
                difference = (reference.logits[0, -1].float() -
                              candidate.logits[0, -1].float()).abs().max().item()
                observations.append({
                    'step': step + 1, 'input_length': len(ids),
                    'reused_closed_patches': count, 'reference_next_id': ref_id,
                    'skipped_global': skipped,
                    'reused_decoder': decoder_reused,
                    'reuse_next_id': reused_id, 'next_id_equal': ref_id == reused_id,
                    'max_abs_logit_difference': difference,
                    'reference_ms': reference_ms, 'reuse_ms': reuse_ms,
                })
                if ref_id == 2:
                    break
                ids.append(ref_id)
            measured = observations[1:]  # first step builds the full cache
            cases.append({
                'name': name, 'steps': len(observations), 'observations': observations,
                'mismatches': sum(not item['next_id_equal'] for item in observations),
                'global_skips': sum(item['skipped_global'] for item in observations),
                'decoder_reuse_steps': sum(item['reused_decoder'] for item in observations),
                'timing_excluding_initial': {
                    'reference_total_ms': sum(item['reference_ms'] for item in measured),
                    'reuse_total_ms': sum(item['reuse_ms'] for item in measured),
                    'reference_mean_ms': mean(item['reference_ms'] for item in measured) if measured else None,
                    'reuse_mean_ms': mean(item['reuse_ms'] for item in measured) if measured else None,
                    'reference_median_ms': median(item['reference_ms'] for item in measured) if measured else None,
                    'reuse_median_ms': median(item['reuse_ms'] for item in measured) if measured else None,
                },
            })
            print(json.dumps({'case': name, 'steps': len(observations),
                              'mismatches': cases[-1]['mismatches'],
                              'timing': cases[-1]['timing_excluding_initial']}), flush=True)

    mismatches = sum(case['mismatches'] for case in cases)
    reference_total = sum(case['timing_excluding_initial']['reference_total_ms'] for case in cases)
    reuse_total = sum(case['timing_excluding_initial']['reuse_total_ms'] for case in cases)
    result = {
        'status': 'complete', 'probe_token_parity': 'passed' if mismatches == 0 else 'failed',
        'scope': ('experimental patch/decoder reuse, fine-tuned checkpoint, sampled greedy validation steps'
                  if args.checkpoint else
                  'experimental patch/decoder reuse, pretrained 1B, greedy fixture steps; not fine-tuned A100 parity'),
        'reuse_decoder': args.reuse_decoder,
        'checkpoint': checkpoint_info, 'data': data_info,
        'model_code_sha256': sha256_file(root / 'blt_hf/patched/modeling_blt.py'),
        'reuse_code_sha256': sha256_file(root / 'blt_hf/cache/global_reuse.py'),
        'frontier_code_sha256': sha256_file(root / 'blt_hf/cache/frontier.py'),
        'bench_code_sha256': sha256_file(root / 'blt_hf_checks/bench_global_reuse.py'),
        'conversion_report_sha256': sha256_file(conversion),
        'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'timing_excluding_initial': {
            'reference_total_ms': reference_total, 'reuse_total_ms': reuse_total,
            'speedup': reference_total / reuse_total if reuse_total else None,
        },
        'cases': cases,
    }
    write_json(output, result)
    print(json.dumps({'status': result['status'], 'probe_token_parity': result['probe_token_parity'],
                      'output': str(output)}), flush=True)
    return int(mismatches != 0)


if __name__ == '__main__':
    raise SystemExit(main())
