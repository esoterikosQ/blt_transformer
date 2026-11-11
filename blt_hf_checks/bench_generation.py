"""Compare single-row and equal-length batched BLT generation on one real checkpoint."""
import argparse
import json
import time
from collections import defaultdict

from blt_hf.checkpoint import resolve_checkpoint
from blt_hf.data_adapter import dataset_split_path, encode_prompt, read_tsv
from blt_hf.generation import GenerationConfig, generate_batch
from blt_hf.manifest import sha256_file, write_json
from blt_hf.runtime import CONVERSION, MODEL, ROOT, local_path, require_neuron_job


def select_groups(rows, tokenizer, *, batch_size, group_count, max_new_bytes):
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        prompt = encode_prompt(tokenizer, row.source, max_new_bytes=max_new_bytes,
                               sample_id=row.sample_id)
        buckets[len(prompt)].append(index)
    candidates = [indices[:batch_size] for _, indices in sorted(buckets.items())
                  if len(indices) >= batch_size]
    if len(candidates) < group_count:
        raise ValueError(f'Only {len(candidates)} exact-length groups of size {batch_size}')
    return [candidates[(2 * group + 1) * len(candidates) // (2 * group_count)]
            for group in range(group_count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', choices=('native', 'korean_learner', 'lang8', 'union'), default='native')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--num-beams', type=int, choices=(1, 4), default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--groups', type=int, default=16)
    parser.add_argument('--max-new-bytes', type=int, default=768)
    args = parser.parse_args()
    require_neuron_job()
    if min(args.batch_size, args.groups, args.max_new_bytes) < 1:
        raise ValueError('Batch size, group count and generation budget must be positive')
    if args.batch_size == 1:
        raise ValueError('Benchmark batch size must exceed one')
    output = local_path(args.output)
    if output.exists():
        raise FileExistsError(output)

    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from blt_hf.model import load_model

    if torch.cuda.device_count() != 1:
        raise ValueError('Benchmark requires exactly one allocated GPU')
    checkpoint, metadata = resolve_checkpoint(local_path(args.checkpoint), verify_optimizer=False)
    if metadata['run_manifest']['dataset'] != args.dataset:
        raise ValueError('Checkpoint and benchmark dataset differ')
    model_path = ROOT / MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tsv = dataset_split_path(ROOT / 'data/Preprocessed', args.dataset, args.split)
    rows = read_tsv(tsv)
    groups = select_groups(rows, tokenizer, batch_size=args.batch_size,
                           group_count=args.groups, max_new_bytes=args.max_new_bytes)
    single_config = GenerationConfig(num_beams=args.num_beams, batch_size=1,
                                     max_new_bytes=args.max_new_bytes)
    batch_config = GenerationConfig(num_beams=args.num_beams, batch_size=args.batch_size,
                                    max_new_bytes=args.max_new_bytes)
    model = load_model(model_path, ROOT / CONVERSION, attention_mode='osc', device='cpu')
    model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
    model.eval().requires_grad_(False).to('cuda')

    def timed(sources, config):
        torch.cuda.synchronize()
        started = time.monotonic()
        result = generate_batch(model, tokenizer, sources, config)
        torch.cuda.synchronize()
        return result, time.monotonic() - started

    warmup = [rows[index].source for index in groups[0]]
    for source in warmup:
        timed([source], single_config)
    timed(warmup, batch_config)
    total_single = total_batch = 0.0
    mismatches = []
    for group_number, group in enumerate(groups, 1):
        sources = [rows[index].source for index in group]
        singles = []
        for source in sources:
            result, seconds = timed([source], single_config)
            singles.append(result[0]); total_single += seconds
        batched, seconds = timed(sources, batch_config)
        total_batch += seconds
        mismatches.extend(index for index, left, right in zip(group, singles, batched)
                          if left['token_ids'] != right['token_ids'])
        print(json.dumps({'completed_groups': group_number, 'total_groups': len(groups),
                          'single_seconds': round(total_single, 2),
                          'batch_seconds': round(total_batch, 2)}), flush=True)
    report = {'status': 'passed' if not mismatches else 'failed',
              'dataset': args.dataset, 'split': args.split, 'tsv_sha256': sha256_file(tsv),
              'checkpoint': str(checkpoint.relative_to(ROOT)),
              'checkpoint_model_sha256': metadata['files']['model.safetensors'],
              'generation_code_sha256': sha256_file(ROOT / 'blt_hf/generation.py'),
              'beams': args.num_beams, 'batch_size': args.batch_size,
              'group_count': args.groups, 'sample_count': args.groups * args.batch_size,
              'max_new_bytes': args.max_new_bytes, 'indices': groups,
              'single_seconds': total_single, 'batch_seconds': total_batch,
              'speedup': total_single / total_batch if total_batch > 0 else None,
              'token_id_mismatches': mismatches,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return int(bool(mismatches))


if __name__ == '__main__':
    raise SystemExit(main())
