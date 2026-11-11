"""Locate which BLT activations change when the same prefix is extended."""
import argparse
import json
from pathlib import Path

from .check_patch_causality import fixture_inputs, patch_starts


def comparison(left, right):
    if left.shape != right.shape:
        raise ValueError(f'Activation shapes differ: {left.shape} vs {right.shape}')
    difference = (left.float() - right.float()).abs()
    changed = difference.ne(0)
    positions = changed.reshape(-1, changed.shape[-1]).any(-1).nonzero().flatten()
    return {'max_abs': difference.max().item() if difference.numel() else 0.0,
            'changed_positions': int(positions.numel()),
            'first_changed_position': int(positions[0]) if positions.numel() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-path', default='artifacts/converted/blt-1b-hf-own')
    parser.add_argument('--conversion-report', default='blt_hf_checks/manifests/conversion_B_20260915.json')
    args = parser.parse_args()

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

    model = load_model(model_path, conversion, attention_mode='osc', device='cuda')
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    cases = fixture_inputs(tokenizer)
    selected = {'mixed': [18, 20, 22, 23, 24, 28, 29, 30, 40],
                'window': [64, 128, 256, 511, 512, 513]}
    reports = []

    def run(token_ids):
        captured = {}

        def save(name):
            def hook(_module, _inputs, result):
                if name == 'patcher':
                    entropy, lengths, _logits = result
                    captured['entropy'] = entropy.detach().float().cpu().clone()
                    captured['lengths'] = lengths[0].detach().cpu().tolist()
                elif name == 'encoder':
                    byte_hidden, patch_hidden = result
                    captured['encoder_bytes'] = byte_hidden.detach().float().cpu().clone()
                    captured['encoder_patches'] = patch_hidden.detach().float().cpu().clone()
                else:
                    captured[name] = result.detach().float().cpu().clone()
            return hook

        modules = {'patcher': model.model.patcher,
                   'encoder': model.model.local_encoder,
                   'global': model.model.global_transformer,
                   'decoder': model.model.local_decoder}
        handles = [module.register_forward_hook(save(name)) for name, module in modules.items()]
        try:
            tokens = torch.tensor([token_ids], device='cuda', dtype=torch.long)
            with torch.inference_mode():
                result = model(input_ids=tokens, use_cache=False)
            captured['logits'] = result.logits.detach().float().cpu().clone()
        finally:
            for handle in handles:
                handle.remove()
        captured['starts'] = patch_starts(captured['lengths'], input_length=len(token_ids))
        return captured

    for name, lengths in selected.items():
        token_ids = cases[name]
        full = run(token_ids)
        observations = []
        for prefix_length in lengths:
            if prefix_length >= len(token_ids):
                continue
            prefix = run(token_ids[:prefix_length])
            repeated_prefix = run(token_ids[:prefix_length])
            common = 0
            for old, new in zip(prefix['starts'], full['starts']):
                if old != new:
                    break
                common += 1
            # The last shared patch can grow or acquire a different end; exclude it.
            safe_patches = max(0, common - 1)
            full_encoder_patches = full['encoder_patches'].reshape(1, len(full['starts']), -1)
            prefix_encoder_patches = prefix['encoder_patches'].reshape(1, len(prefix['starts']), -1)
            item = {'prefix_length': prefix_length,
                    'prefix_starts': prefix['starts'],
                    'full_starts_through_prefix': [s for s in full['starts'] if s <= prefix_length],
                    'shared_closed_patches': safe_patches,
                    'entropy': comparison(full['entropy'][:, :prefix_length], prefix['entropy']),
                    'encoder_bytes': comparison(full['encoder_bytes'][:, :prefix_length],
                                                prefix['encoder_bytes']),
                    'encoder_closed_patches': comparison(full_encoder_patches[:, :safe_patches],
                                                         prefix_encoder_patches[:, :safe_patches]),
                    'global_closed_patches': comparison(full['global'][:, :safe_patches],
                                                        prefix['global'][:, :safe_patches]),
                    'decoder_bytes': comparison(full['decoder'][:, :prefix_length], prefix['decoder']),
                    'last_logits': comparison(full['logits'][:, prefix_length - 1:prefix_length],
                                              prefix['logits'][:, -1:])}
            item['same_shape_repeat'] = {
                name: comparison(prefix[name], repeated_prefix[name])
                for name in ('entropy', 'encoder_bytes', 'encoder_patches',
                             'global', 'decoder', 'logits')
            }
            item['last_argmax_equal'] = bool(full['logits'][0, prefix_length - 1].argmax() ==
                                            prefix['logits'][0, -1].argmax())
            observations.append(item)
            print(json.dumps({'case': name, 'prefix': prefix_length,
                              'global_closed_max_abs': item['global_closed_patches']['max_abs'],
                              'decoder_max_abs': item['decoder_bytes']['max_abs'],
                              'last_argmax_equal': item['last_argmax_equal']}), flush=True)
        reports.append({'name': name, 'full_length': len(token_ids), 'observations': observations})

    result = {'status': 'complete', 'scope': 'same BF16 weights, prefix-vs-full activation stability; not a cache implementation',
              'model_code_sha256': sha256_file(root / 'blt_hf/patched/modeling_blt.py'),
              'conversion_report_sha256': sha256_file(conversion),
              'device': torch.cuda.get_device_name(), 'cases': reports}
    write_json(output, result)
    print(json.dumps({'status': result['status'], 'output': str(output)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
