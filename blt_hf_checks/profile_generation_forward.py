"""Measure a no-cache BLT forward by component on a BF16 CUDA GPU."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()
    if min(*args.lengths, args.warmup, args.repeats) < 1:
        parser.error('Lengths, warmup and repeats must be positive')

    import torch
    from transformers import AutoTokenizer
    from blt_hf.manifest import sha256_file, write_json
    from blt_hf.model import load_model

    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    model_path = (root / args.model_path).resolve()
    conversion = (root / args.conversion_report).resolve()
    if any(not path.is_relative_to(root) for path in (output, model_path, conversion)) or output.exists():
        raise ValueError('Inputs must be inside the project and output must be new')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required')

    model = load_model(model_path, conversion, attention_mode='osc', device='cuda')
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    payload = tokenizer.encode('가' * (max(args.lengths) + 10), add_special_tokens=False, truncation=False)
    modules = {'patcher': model.model.patcher,
               'local_encoder': model.model.local_encoder,
               'global_transformer': model.model.global_transformer,
               'local_decoder': model.model.local_decoder}
    samples = []

    for length in args.lengths:
        token_ids = torch.tensor([[1] + payload[:length - 1]], device='cuda', dtype=torch.long)
        if token_ids.shape[1] != length:
            raise ValueError('Insufficient fixture bytes')
        torch.cuda.reset_peak_memory_stats()
        measures = {name: [] for name in ('total', *modules)}
        patch_counts = []
        event_pairs = {}

        def before(name):
            def hook(_module, _args):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                event_pairs[name] = [start, None]
            return hook

        def after(name):
            def hook(_module, _args, result):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                event_pairs[name][1] = end
                if name == 'patcher':
                    patch_counts.append(result[1].detach())
            return hook

        handles = []
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(before(name)))
            handles.append(module.register_forward_hook(after(name)))
        try:
            with torch.inference_mode():
                for run in range(args.warmup + args.repeats):
                    event_pairs.clear()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(input_ids=token_ids, use_cache=False)
                    end.record()
                    end.synchronize()
                    if run >= args.warmup:
                        measures['total'].append(start.elapsed_time(end))
                        for name in modules:
                            begin, finish = event_pairs[name]
                            measures[name].append(begin.elapsed_time(finish))
        finally:
            for handle in handles:
                handle.remove()
        item = {'input_bytes': length, 'patches': int(patch_counts[-1].ne(0).sum().item()),
                'warmup': args.warmup, 'repeats': args.repeats,
                'milliseconds': {name: {'mean': mean(values), 'median': median(values), 'runs': values}
                                 for name, values in measures.items()},
                'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        samples.append(item)
        print(json.dumps({'input_bytes': length, 'patches': item['patches'],
                          'mean_ms': {name: round(value['mean'], 2)
                                      for name, value in item['milliseconds'].items()}}), flush=True)

    result = {'status': 'complete', 'scope': 'pretrained BLT-1B full forward; not a fine-tuned generation benchmark',
              'model_code_sha256': sha256_file(root / 'blt_hf/patched/modeling_blt.py'),
              'conversion_report_sha256': sha256_file(conversion),
              'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
              'samples': samples}
    write_json(output, result)
    print(json.dumps({'status': result['status'], 'output': str(output)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
