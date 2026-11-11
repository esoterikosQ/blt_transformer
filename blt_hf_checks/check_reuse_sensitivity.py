"""Probe one-step token sensitivity to reusing structurally closed global outputs.

This replaces outputs after a full global forward. It measures correctness risk,
not speed; it is neither a cache backend nor a fine-tuned checkpoint benchmark.
"""
import argparse
import json
from pathlib import Path

from blt_hf.cache.frontier import shared_closed_patch_count
from .check_patch_causality import fixture_inputs, patch_starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')

    import torch
    from transformers import AutoTokenizer
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
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def run(ids, prior=None):
        captured = {}

        def patch_hook(_module, _inputs, result):
            captured['starts'] = patch_starts(result[1][0].tolist(), input_length=len(ids))

        def global_hook(_module, _inputs, result):
            captured['structural_reuse_count'] = 0
            if prior is None:
                captured['global'] = result.detach().clone()
                return None
            count = shared_closed_patch_count(prior['starts'], captured['starts'])
            captured['structural_reuse_count'] = count
            if count:
                modified = result.clone()
                modified[:, :count] = prior['global'][:, :count]
                return modified
            return None

        handles = [model.model.patcher.register_forward_hook(patch_hook),
                   model.model.global_transformer.register_forward_hook(global_hook)]
        try:
            tokens = torch.tensor([ids], device='cuda', dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids=tokens, use_cache=False).logits[0, -1].detach().float().cpu()
        finally:
            for handle in handles:
                handle.remove()
        captured['logits'] = logits
        return captured

    def top2(logits):
        scores = logits.clone()
        scores[[0, 1, 3]] = -float('inf')
        values, indices = torch.topk(scores, 2)
        return int(indices[0]), float(values[0] - values[1])

    cases = []
    fixtures = fixture_inputs(tokenizer)
    for name in ('korean_short', 'mixed', 'window'):
        ids = list(fixtures[name])
        prior = run(ids)
        observations = []
        for step in range(args.steps):
            token, _ = top2(prior['logits'])
            if token == 2:
                break
            ids.append(token)
            reference = run(ids)
            substituted = run(ids, prior=prior)
            reference_token, margin = top2(reference['logits'])
            substituted_token, _ = top2(substituted['logits'])
            difference = (reference['logits'] - substituted['logits']).abs()
            observations.append({
                'step': step + 1, 'input_length': len(ids),
                'structural_reuse_count': substituted['structural_reuse_count'],
                'old_starts': prior['starts'], 'new_starts': reference['starts'],
                'reference_next_id': reference_token,
                'substituted_next_id': substituted_token,
                'next_id_equal': reference_token == substituted_token,
                'reference_top1_margin': margin,
                'max_abs_logit_difference': float(difference.max()),
            })
            prior = reference
        cases.append({'name': name, 'steps': len(observations), 'observations': observations})
        print(json.dumps({'case': name, 'steps': len(observations),
                          'mismatches': sum(not o['next_id_equal'] for o in observations)}), flush=True)

    result = {
        'status': 'complete',
        'scope': 'pretrained 1B one-step global output substitution; not cache speed or trained-output parity',
        'model_code_sha256': sha256_file(root / 'blt_hf/patched/modeling_blt.py'),
        'conversion_report_sha256': sha256_file(conversion),
        'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
        'cases': cases,
    }
    write_json(output, result)
    print(json.dumps({'status': result['status'], 'output': str(output)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
