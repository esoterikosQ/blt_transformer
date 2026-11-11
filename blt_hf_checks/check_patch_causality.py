"""Check whether extending an OSC byte prefix changes earlier patch starts.

Run on itcerdo with the converted 1B artifact. The pure comparison helpers also
support CPU-only unit tests; this does not claim that encoder/decoder caches work.
"""
import argparse
import json
from pathlib import Path

from blt_hf.cache.frontier import patch_starts

def compare_prefix(*, full_entropies, full_lengths, prefix_entropies, prefix_lengths, threshold=None):
    """Compare observed prefix boundaries to the corresponding full-input boundaries."""
    total = len(full_entropies)
    prefix = len(prefix_entropies)
    if not 2 <= prefix <= total:
        raise ValueError('Prefix must contain at least two bytes and fit in full input')
    full_starts = patch_starts(full_lengths, input_length=total)
    prefix_starts = patch_starts(prefix_lengths, input_length=prefix)
    expected = [position for position in full_starts if position <= prefix]
    differences = [abs(float(left) - float(right))
                   for left, right in zip(full_entropies[:prefix], prefix_entropies)]
    changed = sorted(set(prefix_starts) ^ set(expected))
    boundary_values = [
        {'start': start, 'entropy_index': start - 1,
         'full_entropy': float(full_entropies[start - 1]),
         'prefix_entropy': float(prefix_entropies[start - 1]),
         'threshold': threshold}
        for start in changed if 0 < start <= prefix
    ]
    return {'prefix_length': prefix, 'prefix_starts': prefix_starts,
            'full_starts_through_prefix': expected,
            'starts_stable': prefix_starts == expected,
            'changed_start_positions': changed,
            'changed_boundary_entropies': boundary_values,
            'max_abs_entropy_difference': max(differences),
            'first_entropy_difference': next((index for index, value in enumerate(differences) if value != 0), None)}


def selected_prefix_lengths(full_entropies, full_lengths, *, threshold, max_prefixes):
    """Exercise early bytes, patch transitions, threshold ties, and 512-window edges."""
    total = len(full_entropies)
    starts = patch_starts(full_lengths, input_length=total)
    proposed = set(range(2, min(total, 24) + 1))
    proposed.update(position for position in (31, 32, 63, 64, 127, 255, 511, 512, 513, total)
                    if 2 <= position <= total)
    for boundary in starts[:16]:
        proposed.update(position for position in (boundary - 1, boundary, boundary + 1)
                        if 2 <= position <= total)
    near = sorted(range(1, total), key=lambda index: abs(float(full_entropies[index]) - threshold))[:8]
    for index in near:
        proposed.update(position for position in (index, index + 1, index + 2)
                        if 2 <= position <= total)
    # Preserve critical long-context probes even if the requested cap is small.
    critical = {position for position in (511, 512, 513, total) if 2 <= position <= total}
    if len(proposed) > max_prefixes:
        ordinary = sorted(proposed - critical)
        capacity = max(0, max_prefixes - len(critical))
        ordinary = ([ordinary[(2 * i + 1) * len(ordinary) // (2 * capacity)]
                     for i in range(capacity)] if capacity else [])
        proposed = set(ordinary) | critical
    return sorted(proposed)


def fixture_inputs(tokenizer):
    def ids(text):
        return [1] + tokenizer.encode(text, add_special_tokens=False, truncation=False)

    return {'korean_short': ids('맞춤법을 검사합니다.'),
            'ascii_short': ids('A short example.'),
            'mixed': ids('오늘은 Monday입니다. 감사합니다!'),
            'window': ids('가' * 190 + ' 끝.'),
            'eos_segments': ids('첫 문장.') + [2] + ids('둘째 문장.')[1:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New JSON evidence path')
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    parser.add_argument('--max-prefixes', type=int, default=56)
    args = parser.parse_args()
    if args.max_prefixes < 4:
        parser.error('--max-prefixes must be at least 4')

    import torch
    from transformers import AutoTokenizer
    from blt_hf.manifest import sha256_file, sha256_json, write_json
    from blt_hf.model import load_model

    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError('Output must be a new path inside the project')
    model_path = (root / args.model_path).resolve()
    report_path = (root / args.conversion_report).resolve()
    if not model_path.is_relative_to(root) or not report_path.is_relative_to(root):
        raise ValueError('Model and conversion paths must be inside the project')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This 1B probe requires a BF16 CUDA GPU')

    model = load_model(model_path, report_path, attention_mode='osc', device='cuda')
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    cases = []
    with torch.inference_mode():
        for name, token_ids in fixture_inputs(tokenizer).items():
            def patch(ids):
                tensor = torch.tensor([ids], device='cuda', dtype=torch.long)
                entropy, lengths, _ = model.model.patcher(
                    tensor, patch_size=model.config.patch_size,
                    threshold=model.config.patching_threshold,
                    max_patch_length=model.config.max_patch_length, use_cache=False)
                return entropy[0].tolist(), lengths[0].tolist(), entropy.dtype

            full_entropy, full_lengths, dtype = patch(token_ids)
            threshold = torch.tensor(model.config.patching_threshold, dtype=dtype).item()
            prefix_lengths = selected_prefix_lengths(full_entropy, full_lengths,
                                                      threshold=threshold, max_prefixes=args.max_prefixes)
            observations = []
            for length in prefix_lengths:
                entropy, lengths, _ = patch(token_ids[:length])
                observations.append(compare_prefix(full_entropies=full_entropy,
                                                   full_lengths=full_lengths,
                                                   prefix_entropies=entropy,
                                                   prefix_lengths=lengths,
                                                   threshold=threshold))
            failures = [observation for observation in observations if not observation['starts_stable']]
            cases.append({'name': name, 'input_length': len(token_ids),
                          'input_ids_sha256': sha256_json(token_ids),
                          'effective_threshold': threshold, 'entropy_dtype': str(dtype),
                          'full_starts': patch_starts(full_lengths, input_length=len(token_ids)),
                          'prefixes_checked': len(observations),
                          'max_abs_entropy_difference': max(o['max_abs_entropy_difference'] for o in observations),
                          'first_entropy_difference_prefixes': [o['prefix_length'] for o in observations
                                                                if o['first_entropy_difference'] is not None],
                          'boundary_failures': failures})
            print(json.dumps({'case': name, 'length': len(token_ids),
                              'prefixes': len(observations), 'boundary_failures': len(failures)}), flush=True)
    status = 'passed' if all(not case['boundary_failures'] for case in cases) else 'failed'
    result = {'status': status, 'scope': 'OSC entropy patch starts under prefix extension; no main-model cache parity claim',
              'model_code_sha256': sha256_file(root / 'blt_hf/patched/modeling_blt.py'),
              'patching_code_sha256': sha256_file(root / 'blt_hf/patching.py'),
              'conversion_report_sha256': sha256_file(report_path),
              'device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
              'cases': cases}
    write_json(output, result)
    print(json.dumps({'status': status, 'output': str(output), 'cases': len(cases)}), flush=True)
    return int(status != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
