"""Compare complete native validation generations for HF and global-prefix backends."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--max-new-bytes', type=int, default=768)
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()
    if min(args.samples, args.max_new_bytes) < 1:
        parser.error('Samples and max-new-bytes must be positive')

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from blt_hf.checkpoint import resolve_checkpoint
    from blt_hf.data_adapter import dataset_split_path, encode_prompt, read_tsv
    from blt_hf.generation import (GenerationConfig, generate_batch,
                                   HF_BACKEND, GLOBAL_PREFIX_BACKEND)
    from blt_hf.manifest import sha256_file, write_json
    from blt_hf.model import load_model

    root = Path(__file__).resolve().parents[1]

    def inside(path):
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f'Path must be inside project: {path}')
        return resolved

    output = inside(args.output)
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    checkpoint, metadata = resolve_checkpoint(inside(args.checkpoint), verify_optimizer=False)
    if metadata['run_manifest']['dataset'] != 'native' or metadata['run_manifest']['mode'] != 'train':
        raise ValueError('The probe requires a native main-training checkpoint')
    model_path = inside(args.model_path)
    conversion = inside(args.conversion_report)
    tsv = dataset_split_path(root / 'data/Preprocessed', 'native', 'val')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required')

    model = load_model(model_path, conversion, attention_mode='osc', device='cuda')
    model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows = read_tsv(tsv)
    if args.samples > len(rows):
        raise ValueError('More samples requested than validation rows')
    choices = sorted((len(encode_prompt(tokenizer, row.source, sample_id=row.sample_id)), index)
                     for index, row in enumerate(rows))
    indices = [choices[(2 * i + 1) * len(choices) // (2 * args.samples)][1]
               for i in range(args.samples)]
    configs = {
        HF_BACKEND: GenerationConfig(max_new_bytes=args.max_new_bytes),
        GLOBAL_PREFIX_BACKEND: GenerationConfig(max_new_bytes=args.max_new_bytes,
                                                backend=GLOBAL_PREFIX_BACKEND),
    }

    results = []
    for order_index, index in enumerate(indices):
        row = rows[index]
        order = ((HF_BACKEND, GLOBAL_PREFIX_BACKEND) if order_index % 2 == 0 else
                 (GLOBAL_PREFIX_BACKEND, HF_BACKEND))
        outcomes = {}
        times = {}
        for backend in order:
            torch.cuda.synchronize()
            began = time.monotonic()
            outcomes[backend] = generate_batch(model, tokenizer, [row.source], configs[backend])[0]
            torch.cuda.synchronize()
            times[backend] = time.monotonic() - began
        reference, candidate = outcomes[HF_BACKEND], outcomes[GLOBAL_PREFIX_BACKEND]
        equal = reference == candidate
        record = {'sample_index': index, 'sample_id': row.sample_id,
                  'prompt_bytes': len(encode_prompt(tokenizer, row.source, sample_id=row.sample_id)),
                  'reference_seconds': times[HF_BACKEND],
                  'cached_seconds': times[GLOBAL_PREFIX_BACKEND],
                  'same_token_ids': reference['token_ids'] == candidate['token_ids'],
                  'same_complete_output': equal,
                  'reference_new_bytes': len(reference['token_ids']),
                  'candidate_new_bytes': len(candidate['token_ids']),
                  'reference_eos': reference['eos_reached'],
                  'candidate_eos': candidate['eos_reached'],
                  'reference_budget_exhausted': reference['budget_exhausted'],
                  'candidate_budget_exhausted': candidate['budget_exhausted']}
        if not equal:
            record['reference_output'] = reference
            record['candidate_output'] = candidate
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    reference_total = sum(record['reference_seconds'] for record in results)
    cached_total = sum(record['cached_seconds'] for record in results)
    passed = all(record['same_complete_output'] for record in results)
    report = {'status': 'complete' if passed else 'mismatch',
              'scope': 'complete greedy generations on length-stratified native validation samples',
              'checkpoint': str(checkpoint.relative_to(root)),
              'checkpoint_model_sha256': metadata['files']['model.safetensors'],
              'validation_tsv_sha256': sha256_file(tsv),
              'generation_code_sha256': sha256_file(root / 'blt_hf/generation.py'),
              'reuse_code_sha256': sha256_file(root / 'blt_hf/cache/global_reuse.py'),
              'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
              'num_beams': 1, 'batch_size': 1, 'max_new_bytes': args.max_new_bytes,
              'decoder_kv_reuse': False, 'samples': len(results), 'cases': results,
              'reference_generation_seconds': reference_total,
              'cached_generation_seconds': cached_total,
              'generation_speedup': reference_total / cached_total if cached_total else None,
              'complete_output_parity': 'passed' if passed else 'failed'}
    write_json(output, report)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
