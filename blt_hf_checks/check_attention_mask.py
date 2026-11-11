"""Audit actual B attention inputs against independent visibility rules (no legacy runtime)."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def segments(row, eos=2):
    result, current = [], 0
    for token in row:
        result.append(current)
        if token == eos:
            current += 1
    return result


def membership(lengths, size):
    # Interval expansion is deliberately independent of the runtime cumsum/search formula.
    ids = []
    for patch, length in enumerate(lengths):
        if length < 0:
            raise ValueError("Negative patch length")
        ids.extend([patch] * length)
    if len(ids) < size:
        raise ValueError("Patch lengths do not cover all tokens")
    return ids[:size]


def causal_reference(rows, device, window=None):
    import torch
    seg = torch.tensor([segments(row) for row in rows], device=device)
    pos = torch.arange(len(rows[0]), device=device)
    # Open interval q-window < k <= q, within the EOS-delimited segment.
    allowed = (pos[None, :] <= pos[:, None]) & (seg[:, :, None] == seg[:, None, :])
    if window is not None:
        allowed &= pos[None, :] > pos[:, None] - window
    return allowed.unsqueeze(1)


def check_mask(mask, expected):
    import torch
    if mask is None or mask.shape != expected.shape:
        raise AssertionError(f"Mask shape mismatch: {getattr(mask, 'shape', None)} vs {expected.shape}")
    # Check the entire matrix, including the actual additive sentinel, not just counts.
    valid_block = torch.isneginf(mask) | mask.eq(torch.finfo(mask.dtype).min)
    if not torch.all(torch.where(expected, mask.eq(0), valid_block)):
        bad = torch.nonzero(~torch.where(expected, mask.eq(0), valid_block))[0].tolist()
        raise AssertionError(f"Mask visibility/value mismatch at {bad}")
    return {"shape": list(mask.shape), "allowed": int(expected.sum()),
            "empty_query_rows": int((~expected.any(-1)).sum())}


def check_patch_causality(rows, lengths):
    """Conservative byte-dependency bound through pooling, global and decoder cross attention.

    Empty encoder cross rows may softmax to a uniform distribution with finite-min masks.
    Treat them as depending on EVERY input byte; prove they cannot reach active predictions.
    This is a topology check, not a numerical equivalence or complete EOS isolation claim.
    """
    size = len(rows[0])
    empty_count = 0
    for row, lens in zip(rows, lengths):
        if not lens or lens[0] != 1 or sum(lens) != size + 1:
            raise AssertionError("Expected first patch length 1 and S+1 prediction slots")
        positive = [v for v in lens if v > 0]
        if lens != positive + [0] * (len(lens) - len(positive)):
            raise AssertionError("Only right-hand zero patch padding is supported")
        enc, dec = membership(lens, size), membership(lens[1:], size)
        maxima, patch_tokens = [], [0] * len(lens)
        for patch in range(len(lens)):
            members = [i for i, p in enumerate(enc) if p == patch]
            maxima.append(max(members) if members else size - 1)
            empty_count += not members
        for i, token in enumerate(row):
            if token == 2:
                patch_tokens[enc[i]] = 2
        seg = segments(patch_tokens)
        global_max = [max(maxima[k] for k in range(q + 1) if seg[k] == seg[q])
                      for q in range(len(lens))]
        for q, patch in enumerate(dec):
            if global_max[patch] > q:
                raise AssertionError(f"Future byte can reach prediction at {q} via patch {patch}")
    return {"future_dependency_violations": 0, "conservative_empty_encoder_patches": empty_count}


class MaskAudit:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.expected_names = []
        body = model.model
        self.handles.append(body.patcher.register_forward_hook(self.capture_lengths))
        for role, component in (("entropy", body.patcher), ("encoder", body.local_encoder),
                                ("global", body.global_transformer), ("decoder", body.local_decoder)):
            for i, layer in enumerate(component.layers):
                self.attach(f"{role}.self.{i}", role, layer.self_attn)
        for role, component in (("encoder_cross", body.local_encoder), ("decoder_cross", body.local_decoder)):
            for i, layer in enumerate(component.cross_attn_layers):
                self.attach(f"{role}.{i}", role, layer)

    def capture_lengths(self, module, args, output):
        self.lengths = output[1].detach().cpu().tolist()

    def attach(self, name, role, module):
        self.expected_names.append(name)
        def hook(module, args, kwargs):
            if name in self.observed:
                raise AssertionError(f"Unexpected duplicate attention call: {name}")
            mask = kwargs["attention_mask"]
            if role not in self.expected:
                self.expected[role] = self.reference(role, mask.device)
            self.observed[name] = check_mask(mask, self.expected[role])
        self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def reference(self, role, device):
        import torch
        spec = self.model.config.original_spec
        if role in ("entropy", "encoder", "decoder"):
            window = spec["entropy_sliding_window" if role == "entropy" else "local_attention_window_len"]
            return causal_reference(self.rows, device, window)
        if self.lengths is None:
            raise AssertionError("Missing patcher output")
        size, count = len(self.rows[0]), len(self.lengths[0])
        enc = [membership(lens, size) for lens in self.lengths]
        if role == "global":
            tokens = [[0] * count for _ in self.rows]
            for b, row in enumerate(self.rows):
                for i, token in enumerate(row):
                    if token == 2:
                        tokens[b][enc[b][i]] = 2
            return causal_reference(tokens, device)
        k = self.model.config.cross_attn_k
        patch = torch.arange(count, device=device)
        if role == "encoder_cross":
            ids = torch.tensor(enc, device=device)
            expected = patch[None, :, None] == ids[:, None, :]
            return expected.repeat_interleave(k, dim=1).unsqueeze(1)
        dec = torch.tensor([membership(lens[1:], size) for lens in self.lengths], device=device)
        expected = dec[:, :, None] == patch[None, None, :]
        return expected.repeat_interleave(k, dim=-1).unsqueeze(1)

    def run(self, rows, *, lengths=None):
        import torch
        self.rows, self.lengths = rows, lengths
        self.observed, self.expected = {}, {}
        device = next(self.model.parameters()).device
        ids = torch.tensor(rows, device=device)
        kwargs = {} if lengths is None else {"patch_lengths": torch.tensor(lengths, device=device)}
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=False, **kwargs)
            if not torch.isfinite(out.logits).all():
                raise AssertionError("Nonfinite logits")
        wanted = set(self.expected_names)
        if lengths is not None:
            wanted = {name for name in wanted if not name.startswith("entropy.")}
        if set(self.observed) != wanted:
            raise AssertionError(f"Attention hook coverage mismatch: {set(self.observed) ^ wanted}")
        return {"batch_size": len(rows), "sequence_length": len(rows[0]),
                "patch_source": "internal_entropy" if lengths is None else "explicit_fixture",
                "patch_lengths": self.lengths, "modules_checked": len(self.observed),
                "modules": self.observed, **check_patch_causality(rows, self.lengths)}

    def close(self):
        for handle in self.handles:
            handle.remove()


def fixture_cases():
    def row(size):
        return [1] + [4 + (i * 37 % 256) for i in range(size - 1)]
    yield "eos_batch", [[1, 40, 50, 2, 2, 60, 70, 2] + [80] * 16,
                        [1, 60, 2] + [90] * 20 + [2]], None
    for size in (511, 512, 513, 1380, 2048):
        yield f"length_{size}", [row(size)], None
    # Force a virtual future patch, then a different number of patches per batch row.
    yield "virtual_and_zero_patches", [row(12), row(12)], [[1] * 13, [1, 4, 8] + [0] * 10]


def main():
    import torch
    from blt_hf.manifest import sha256_file, sha256_json, write_json
    from blt_hf.model import load_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Choose a new evidence path")
    conv = ROOT / "blt_hf_checks/manifests/conversion_B_20260915.json"
    weights = ROOT / "blt_hf_checks/results/weights_p1_validation_20260915.json"
    report = {"scope": "weight_preservation_and_attention_masks", "attention_mode": "osc",
              "backend": "eager", "dtype": "bfloat16", "use_cache": False,
              "original_forward_equivalence": "not_required_not_claimed",
              "weight_evidence": {"path": str(weights.relative_to(ROOT)), "sha256": sha256_file(weights)},
              "conversion_report_hash": sha256_file(conv), "code_hashes": {}, "cases": []}
    for name in ("blt_hf_checks/check_attention_mask.py", "blt_hf/model.py", "blt_hf/attention.py",
                 "blt_hf/patching.py", "blt_hf/patched/modeling_blt.py"):
        report["code_hashes"][name] = sha256_file(ROOT / name)
    audit = None
    try:
        prior = json.loads(weights.read_text())
        if prior["status"] != "passed" or prior["failures"] or prior["conversion_report_hash"] != sha256_file(conv):
            raise AssertionError("Prior weight report is not valid for this conversion")
        if prior["checker_hash"] != sha256_file(ROOT / "blt_hf_checks/check_weight_conversion.py"):
            raise AssertionError("Weight checker changed; rerun weight checks")
        model = load_model(ROOT / "artifacts/converted/blt-1b-hf-own", conv, attention_mode="osc").eval()
        report["weight_checks"] = "passed"  # load_model rehashes every B artifact before strict loading.
        report["artifact_hashes_rechecked"] = True
        report["runtime_config_hash"] = sha256_json(model.config.to_dict())
        report["torch"] = torch.__version__
        report["gpu"] = torch.cuda.get_device_name()
        audit = MaskAudit(model)
        for name, rows, lengths in fixture_cases():
            start = time.perf_counter()
            result = audit.run(rows, lengths=lengths)
            torch.cuda.synchronize()
            result.update(name=name, fixture_hash=sha256_json({"ids": rows, "lengths": lengths}),
                          seconds=time.perf_counter() - start)
            report["cases"].append(result)
            print(json.dumps({"case": name, "modules_checked": result["modules_checked"], "status": "passed"}), flush=True)
        report.update(status="passed", attention_mask_checks="passed",
                      training_checks="not_run", evaluation_checks="not_run",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status="failed", attention_mask_checks="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        if audit is not None:
            audit.close()
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2), flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
