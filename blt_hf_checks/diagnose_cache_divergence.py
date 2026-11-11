"""Replay the first differing byte of each full-validation cache mismatch.

The reference continuation is teacher-forced only until its first disagreement
with the cached continuation. No weights or evaluation outputs are changed.
"""

import argparse
import json
from pathlib import Path


def continuation(record):
    tokens = list(record['token_ids'])
    if record['eos_reached']:
        tokens.append(2)
    return tokens


def first_difference(reference, candidate):
    for index in range(max(len(reference), len(candidate))):
        left = reference[index] if index < len(reference) else None
        right = candidate[index] if index < len(candidate) else None
        if left != right:
            return index, left, right
    raise ValueError('No token difference to diagnose')


def classify_case(*, replayed, skipped_global, cached_id, recorded_candidate_id,
                  full_id, recorded_reference_id, repeat_full_id):
    if not replayed or cached_id != recorded_candidate_id:
        return 'candidate_not_reproduced'
    if full_id != repeat_full_id:
        return 'full_forward_not_repeatable'
    if full_id != recorded_reference_id:
        return 'full_forward_differs_from_hf_reference'
    if skipped_global:
        return 'global_skip_changes_greedy_choice'
    return 'divergence_without_global_skip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from blt_hf.cache.global_reuse import GlobalPrefixReuse
    from blt_hf.checkpoint import resolve_checkpoint
    from blt_hf.data_adapter import dataset_split_path, encode_prompt, read_tsv
    from blt_hf.evaluation import collect_records
    from blt_hf.manifest import sha256_file, split_identity, write_json
    from blt_hf.model import load_model
    from blt_hf.runtime import ROOT, local_path, require_neuron_job, tokenizer_identity
    from blt_hf_checks.compare_cache_eval import (
        compare_records, validate_manifests, verified_manifest,
    )

    require_neuron_job()
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('One BF16-capable GPU is required')
    comparison_path, output_path = local_path(args.comparison), local_path(args.output)
    if output_path.exists():
        raise ValueError(f'Output already exists: {output_path}')
    comparison = json.loads(comparison_path.read_text())
    if comparison['status'] != 'mismatch' or comparison['exact_output_parity'] != 'failed':
        raise ValueError('Expected an immutable full-validation mismatch report')
    reference_dir = local_path(comparison['reference_dir'])
    candidate_dir = local_path(comparison['candidate_dir'])
    reference_manifest = verified_manifest(reference_dir)
    candidate_manifest = verified_manifest(candidate_dir)
    validate_manifests(reference_manifest, candidate_manifest)
    if (comparison['reference_fingerprint'] != reference_manifest['fingerprint'] or
            comparison['candidate_fingerprint'] != candidate_manifest['fingerprint']):
        raise ValueError('Comparison does not identify these evaluation outputs')
    for path, expected in reference_manifest['code_files'].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f'Evaluation code has changed since generation: {path}')
    reference = collect_records(reference_dir, reference_manifest)
    candidate = collect_records(candidate_dir, candidate_manifest)
    mismatches = compare_records(reference, candidate)
    if mismatches != comparison['mismatches'] or not mismatches:
        raise ValueError('Comparison mismatch list changed')
    if any('token_ids' not in item['fields'] for item in mismatches):
        raise ValueError('A mismatch has no differing token IDs')

    checkpoint, metadata = resolve_checkpoint(local_path(args.checkpoint), verify_optimizer=False)
    if (metadata['files']['model.safetensors'] != comparison['checkpoint_hash'] or
            metadata['files']['model.safetensors'] != reference_manifest['checkpoint_hash']):
        raise ValueError('Checkpoint differs from both evaluation runs')
    if metadata['run_manifest']['dataset'] != 'native' or metadata['run_manifest']['mode'] != 'train':
        raise ValueError('Expected the native trained checkpoint')
    conversion_path = local_path(args.conversion_report)
    if metadata['run_manifest']['conversion_hash'] != sha256_file(conversion_path):
        raise ValueError('Conversion report differs from the trained checkpoint')
    tsv = dataset_split_path(ROOT / 'data/Preprocessed', 'native', 'val')
    m2 = tsv.with_suffix('.m2')
    identity = split_identity(tsv, m2, dataset='native', split='val')
    if any(reference_manifest[key] != value for key, value in identity.items()):
        raise ValueError('Native validation files differ from evaluated data')
    rows = read_tsv(tsv)
    model_path = local_path(args.model_path)
    if tokenizer_identity(model_path) != reference_manifest['tokenizer_hash']:
        raise ValueError('Tokenizer differs from the evaluated model')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = load_model(model_path, conversion_path,
                       attention_mode='osc', device='cuda')
    model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
    model.eval().requires_grad_(False)

    def greedy_logits(output):
        logits = output.logits[0, -1].detach().float().clone()
        logits[[0, 1, 3]] = -float('inf')
        return logits

    cases = []
    with torch.inference_mode():
        for item in mismatches:
            index = item['index']
            left, right, row = reference[index], candidate[index], rows[index]
            if left['source'] != row.source or right['source'] != row.source:
                raise ValueError(f'Source changed at row {index}')
            reference_ids, candidate_ids = continuation(left), continuation(right)
            step, expected_reference, expected_candidate = first_difference(reference_ids, candidate_ids)
            if expected_reference is None or expected_candidate is None:
                raise ValueError(f'Unexpected missing token at first difference: {index}')
            prompt = list(encode_prompt(tokenizer, row.source,
                                        max_sequence_bytes=reference_manifest['max_sequence_bytes'],
                                        max_new_bytes=reference_manifest['max_new_bytes'],
                                        sample_id=row.sample_id))
            reuse = GlobalPrefixReuse(model, reuse_decoder=False)
            replayed = True
            prior_failure = None
            for offset in range(step + 1):
                tokens = torch.tensor([prompt + reference_ids[:offset]],
                                      dtype=torch.long, device='cuda')
                previous_starts = list(reuse.starts) if reuse.starts is not None else None
                cached_output, closed_count, skipped, decoder_reused = reuse.run(tokens)
                cached_logits = greedy_logits(cached_output)
                cached_id = int(cached_logits.argmax())
                if offset < step and cached_id != reference_ids[offset]:
                    replayed = False
                    prior_failure = {'step': offset, 'expected_id': reference_ids[offset],
                                     'replayed_id': cached_id}
                    break
            if not replayed:
                case = {'index': index, 'first_difference_step': step,
                        'classification': 'candidate_not_reproduced',
                        'prior_failure': prior_failure}
            else:
                # The direct full forward has the same input and suppression as
                # the cached step, but recomputes global from current patch states.
                full_logits = greedy_logits(model(input_ids=tokens, use_cache=False))
                repeated_logits = greedy_logits(model(input_ids=tokens, use_cache=False))
                # HF generate passes a full attention mask, while the cache
                # backend's manual forward does not. Record this separately so
                # an input-call difference cannot be mistaken for global reuse.
                hf_style_logits = greedy_logits(model(input_ids=tokens,
                                                      attention_mask=torch.ones_like(tokens),
                                                      use_cache=False))
                # Transformers 5.16.1 generate() explicitly passes
                # logits_to_keep=1 when the model supports it. The v1 manual
                # cache path uses the forward default (0), which projects the
                # full sequence and may select a different BF16 GEMM kernel.
                hf_shape_logits = greedy_logits(model(input_ids=tokens,
                                                      attention_mask=torch.ones_like(tokens),
                                                      use_cache=False, logits_to_keep=1))
                hf_shape_repeat_logits = greedy_logits(model(input_ids=tokens,
                                                             attention_mask=torch.ones_like(tokens),
                                                             use_cache=False, logits_to_keep=1))
                full_id, repeated_id = int(full_logits.argmax()), int(repeated_logits.argmax())
                hf_style_id = int(hf_style_logits.argmax())
                hf_shape_id = int(hf_shape_logits.argmax())
                hf_shape_repeat_id = int(hf_shape_repeat_logits.argmax())
                finite = torch.isfinite(full_logits) & torch.isfinite(cached_logits)
                max_abs_delta = float((full_logits[finite] - cached_logits[finite]).abs().max())
                classification = classify_case(
                    replayed=True, skipped_global=skipped, cached_id=cached_id,
                    recorded_candidate_id=expected_candidate, full_id=full_id,
                    recorded_reference_id=expected_reference, repeat_full_id=repeated_id)
                case = {
                    'index': index, 'first_difference_step': step,
                    'input_length': int(tokens.shape[1]),
                    'reference_id': expected_reference, 'candidate_id': expected_candidate,
                    'replayed_candidate_id': cached_id, 'full_forward_id': full_id,
                    'repeat_full_forward_id': repeated_id,
                    'hf_style_full_forward_id': hf_style_id,
                    'hf_generate_shape_forward_id': hf_shape_id,
                    'hf_generate_shape_repeat_id': hf_shape_repeat_id,
                    'hf_generate_shape_recovers_reference': hf_shape_id == expected_reference == hf_shape_repeat_id,
                    'skipped_global': skipped, 'reused_closed_patches': closed_count,
                    'decoder_reused': decoder_reused,
                    'same_patch_starts': previous_starts == reuse.starts if previous_starts is not None else False,
                    'patch_count_before': len(previous_starts) if previous_starts is not None else None,
                    'patch_count_after': len(reuse.starts),
                    'max_abs_finite_logit_delta': max_abs_delta,
                    'full_ref_minus_candidate_logit': float(full_logits[expected_reference] - full_logits[expected_candidate]),
                    'cached_ref_minus_candidate_logit': float(cached_logits[expected_reference] - cached_logits[expected_candidate]),
                    'hf_generate_shape_ref_minus_candidate_logit': float(
                        hf_shape_logits[expected_reference] - hf_shape_logits[expected_candidate]),
                    'classification': classification,
                }
            cases.append(case)
            print(json.dumps({'index': index, 'step': step,
                              'classification': case['classification']}, ensure_ascii=False), flush=True)

    classes = {name: sum(case['classification'] == name for case in cases)
               for name in sorted({case['classification'] for case in cases})}
    shape_summary = {
        'hf_shape_recovers_reference': sum(case.get('hf_generate_shape_recovers_reference', False)
                                           for case in cases),
        'full0_differs_but_hf_shape_recovers_reference': sum(
            case['classification'] == 'full_forward_differs_from_hf_reference'
            and case.get('hf_generate_shape_recovers_reference', False) for case in cases),
    }
    report = {'status': 'complete', 'scope': 'first differing token on all native validation cache mismatches',
              'comparison': str(comparison_path.relative_to(ROOT)),
              'comparison_sha256': sha256_file(comparison_path),
              'reference_fingerprint': reference_manifest['fingerprint'],
              'candidate_fingerprint': candidate_manifest['fingerprint'],
              'checkpoint_model_sha256': comparison['checkpoint_hash'],
              'validation_tsv_sha256': sha256_file(tsv),
              'diagnostic_code_sha256': sha256_file(Path(__file__)),
              'generation_code_sha256': sha256_file(ROOT / 'blt_hf/generation.py'),
              'reuse_code_sha256': sha256_file(ROOT / 'blt_hf/cache/global_reuse.py'),
              'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
              'mismatch_count': len(mismatches), 'classifications': classes,
              'logit_shape_summary': shape_summary, 'cases': cases}
    write_json(output_path, report)
    print(json.dumps({'status': report['status'], 'mismatch_count': len(mismatches),
                      'classifications': classes, 'logit_shape_summary': shape_summary}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
