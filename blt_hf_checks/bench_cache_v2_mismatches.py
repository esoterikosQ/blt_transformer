"""Replay all v1 mismatch rows plus length-stratified controls with a new backend."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--control-report', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=('global-prefix-greedy-v2',
                                              'global-prefix-guarded-greedy-v1'),
                        default='global-prefix-greedy-v2')
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from blt_hf.checkpoint import resolve_checkpoint
    from blt_hf.data_adapter import dataset_split_path, read_tsv
    from blt_hf.evaluation import collect_records
    from blt_hf.generation import GenerationConfig, generate_batch
    from blt_hf.manifest import sha256_file, split_identity, write_json
    from blt_hf.model import load_model
    from blt_hf.runtime import ROOT, local_path, require_neuron_job, tokenizer_identity
    from blt_hf_checks.compare_cache_eval import (
        OUTPUT_FIELDS, compare_records, validate_manifests, verified_manifest,
    )

    require_neuron_job()
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('One BF16-capable GPU is required')
    comparison_path = local_path(args.comparison)
    controls_path = local_path(args.control_report)
    output_path = local_path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    comparison = json.loads(comparison_path.read_text())
    controls = json.loads(controls_path.read_text())
    if comparison['status'] != 'mismatch' or controls['status'] != 'complete':
        raise ValueError('Expected completed v1 comparison and 12-sample control report')
    reference_dir = local_path(comparison['reference_dir'])
    candidate_dir = local_path(comparison['candidate_dir'])
    reference_manifest = verified_manifest(reference_dir)
    candidate_manifest = verified_manifest(candidate_dir)
    validate_manifests(reference_manifest, candidate_manifest)
    if (reference_manifest['fingerprint'] != comparison['reference_fingerprint'] or
            candidate_manifest['fingerprint'] != comparison['candidate_fingerprint']):
        raise ValueError('Comparison fingerprints differ from stored runs')
    # Candidate backends change only these generation paths; the checkpoint
    # and model runtime must still be the same as the saved HF reference.
    changed_generation_paths = {'blt_hf/generation.py', 'blt_hf/eval.py',
                                'blt_hf/cache/global_reuse.py'}
    for path, expected in reference_manifest['code_files'].items():
        if path not in changed_generation_paths and sha256_file(ROOT / path) != expected:
            raise ValueError(f'Model/evaluation dependency changed: {path}')
    reference = collect_records(reference_dir, reference_manifest)
    candidate = collect_records(candidate_dir, candidate_manifest)
    mismatches = compare_records(reference, candidate)
    if mismatches != comparison['mismatches']:
        raise ValueError('Saved mismatch list changed')
    mismatch_indices = {item['index'] for item in mismatches}
    control_indices = {case['sample_index'] for case in controls['cases']}
    if (controls['checkpoint_model_sha256'] != comparison['checkpoint_hash'] or
            controls['validation_tsv_sha256'] != reference_manifest['tsv_hash'] or
            len(control_indices) != 12 or control_indices & mismatch_indices):
        raise ValueError('Control sample identity does not match the validation comparison')

    checkpoint, metadata = resolve_checkpoint(local_path(args.checkpoint), verify_optimizer=False)
    if (metadata['files']['model.safetensors'] != comparison['checkpoint_hash'] or
            metadata['run_manifest']['dataset'] != 'native' or
            metadata['run_manifest']['mode'] != 'train'):
        raise ValueError('Checkpoint differs from the native HF evaluation')
    conversion = local_path(args.conversion_report)
    if metadata['run_manifest']['conversion_hash'] != sha256_file(conversion):
        raise ValueError('Conversion report differs from training')
    tsv = dataset_split_path(ROOT / 'data/Preprocessed', 'native', 'val')
    split = split_identity(tsv, tsv.with_suffix('.m2'), dataset='native', split='val')
    if any(reference_manifest[key] != value for key, value in split.items()):
        raise ValueError('Validation data changed')
    rows = read_tsv(tsv)
    model_path = local_path(args.model_path)
    if tokenizer_identity(model_path) != reference_manifest['tokenizer_hash']:
        raise ValueError('Tokenizer differs from HF evaluation')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = load_model(model_path, conversion, attention_mode='osc', device='cuda')
    model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
    model.eval().requires_grad_(False)
    config = GenerationConfig(max_new_bytes=reference_manifest['max_new_bytes'],
                              max_sequence_bytes=reference_manifest['max_sequence_bytes'],
                              backend=args.backend)

    cases = []
    for index in sorted(mismatch_indices | control_indices):
        row = rows[index]
        if reference[index]['source'] != row.source:
            raise ValueError(f'Source changed at row {index}')
        torch.cuda.synchronize()
        started = time.monotonic()
        result = generate_batch(model, tokenizer, [row.source], config)[0]
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        different = [key for key in OUTPUT_FIELDS if key != 'source' and
                     reference[index].get(key) != result.get(key)]
        case = {'index': index, 'set': 'v1_mismatch' if index in mismatch_indices else 'control',
                'same_token_ids': reference[index]['token_ids'] == result['token_ids'],
                'different_fields': different, 'generation_seconds': seconds}
        if 'cache_preliminary_skips' in result:
            case['cache_preliminary_skips'] = result['cache_preliminary_skips']
            case['cache_guard_refreshes'] = result['cache_guard_refreshes']
        cases.append(case)
        print(json.dumps(case), flush=True)

    remaining = [case['index'] for case in cases if case['different_fields']]
    report = {'status': 'complete', 'scope': 'complete generation on 32 v1 mismatches and 12 controls',
              'comparison': str(comparison_path.relative_to(ROOT)),
              'comparison_sha256': sha256_file(comparison_path),
              'control_report': str(controls_path.relative_to(ROOT)),
              'control_report_sha256': sha256_file(controls_path),
              'checkpoint_model_sha256': comparison['checkpoint_hash'],
              'validation_tsv_sha256': sha256_file(tsv),
              'backend': args.backend,
              'generation_code_sha256': sha256_file(ROOT / 'blt_hf/generation.py'),
              'reuse_code_sha256': sha256_file(ROOT / 'blt_hf/cache/global_reuse.py'),
              'diagnostic_code_sha256': sha256_file(Path(__file__)),
              'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
              'v1_mismatch_count': len(mismatch_indices), 'control_count': len(control_indices),
              'remaining_mismatch_count': len(remaining), 'remaining_mismatch_indices': remaining,
              'generation_seconds': sum(case['generation_seconds'] for case in cases),
              'cache_preliminary_skips': sum(case.get('cache_preliminary_skips', 0) for case in cases),
              'cache_guard_refreshes': sum(case.get('cache_guard_refreshes', 0) for case in cases),
              'cases': cases}
    write_json(output_path, report)
    print(json.dumps({'status': report['status'], 'remaining_mismatch_count': len(remaining),
                      'control_count': len(control_indices)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
