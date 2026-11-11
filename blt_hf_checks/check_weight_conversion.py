"""Independent, streaming A->B tensor mapping check; never imports the converter."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blt_hf.manifest import sha256_file, write_json, verify_file_hashes


def target_for(source, key, vocab_size):
    """Map semantic path components; self-attention RoPE layout is interleaved in both stacks."""
    if source not in ("main", "entropy"):
        raise ValueError("Unknown source")
    hash_match = re.fullmatch(r"encoder_hash_tok_embedding\.(\d+)\.weight", key)
    if source == "main" and hash_match:
        index = int(hash_match[1])
        return "model.encoder_hash_tok_embedding.weight", [index * vocab_size, (index + 1) * vocab_size]
    names = {"attention": "self_attn", "feed_forward": "mlp", "attention_norm": "input_layernorm",
             "ffn_norm": "post_attention_layernorm", "tok_embeddings": "embed_tokens",
             "cross_attn_norm_q": "q_norm", "cross_attn_norm_kv": "k_norm",
             "wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj",
             "w1": "gate_proj", "w2": "down_proj", "w3": "up_proj", "output": "lm_head"}
    if source == "main" and key == "local_decoder.output.weight":
        return "lm_head.weight", None
    path = ".".join(names.get(part, part) for part in key.split("."))
    return ("model.patcher." if source == "entropy" else "model.") + path, None


def compare_tensors(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def main():
    import torch
    from safetensors import safe_open
    from transformers import BltConfig, BltForCausalLM
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--conversion-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mapping-output", required=True)
    args = parser.parse_args()
    model_path, report_path = Path(args.model_path).resolve(), Path(args.conversion_report).resolve()
    if not model_path.is_relative_to(ROOT) or not report_path.is_relative_to(ROOT):
        raise ValueError("Artifacts must be project-local")
    if Path(args.output).exists() or Path(args.mapping_output).exists():
        raise FileExistsError("Choose new evidence paths")
    conversion = json.loads(report_path.read_text())
    verify_file_hashes(model_path, conversion["output_files"])
    for entry in conversion["inputs"].values():
        path = Path(entry["path"]).resolve()
        if not path.is_relative_to(ROOT) or sha256_file(path) != entry["sha256"]:
            raise ValueError("Original source path/hash mismatch")
    config = BltConfig.from_pretrained(model_path, local_files_only=True)
    with torch.device("meta"):
        schema_model = BltForCausalLM(config)
    schema = {name: list(t.shape) for name, t in schema_model.state_dict().items()}
    del schema_model
    source_paths = {"main": conversion["inputs"]["model.safetensors"]["path"],
                    "entropy": conversion["inputs"]["legacy_entropy/model.safetensors"]["path"]}
    mapping, failures, covered = [], [], {}
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as target:
        target_keys = set(target.keys())
        for source, path in source_paths.items():
            with safe_open(path, framework="pt", device="cpu") as original:
                for key in original.keys():
                    dest, row_slice = target_for(source, key, config.encoder_hash_byte_group_vocab)
                    entry = {"source": source, "source_key": key, "target_key": dest,
                             "operation": "fuse_rows" if row_slice else "rename", "rows": row_slice}
                    a = original.get_tensor(key)
                    entry.update(shape=list(a.shape), dtype=str(a.dtype))
                    if dest not in target_keys:
                        failures.append({**entry, "reason": "missing_target"})
                    else:
                        b = target.get_slice(dest)[slice(*row_slice)] if row_slice else target.get_tensor(dest)
                        if not compare_tensors(a, b):
                            failures.append({**entry, "reason": "shape_dtype_or_value"})
                        covered.setdefault(dest, []).append(row_slice)
                        del b
                    mapping.append(entry)
                    del a
        for name in target_keys:
            rows = covered.get(name, [])
            shape = target.get_slice(name).get_shape()
            if not rows:
                failures.append({"target_key": name, "reason": "unconsumed_target"})
            elif rows != [None]:
                if any(r is None for r in rows):
                    failures.append({"target_key": name, "reason": "collision"})
                else:
                    intervals = sorted(rows)
                    if intervals[0][0] != 0 or intervals[-1][1] != shape[0] or any(
                        a[1] != b[0] for a, b in zip(intervals, intervals[1:])
                    ):
                        failures.append({"target_key": name, "reason": "fused_coverage_gap_or_overlap"})
            if schema.get(name) != shape:
                failures.append({"target_key": name, "reason": "model_schema_shape_or_unexpected"})
        for name in set(schema) - target_keys:
            failures.append({"target_key": name, "reason": "model_schema_missing"})
    write_json(args.mapping_output, {"conversion_report_hash": sha256_file(report_path), "mappings": mapping})
    report = {"status": "passed" if not failures else "failed", "source_tensor_count": len(mapping),
              "target_tensor_count": len(target_keys), "model_schema_tensor_count": len(schema),
              "failures": failures, "conversion_report_hash": sha256_file(report_path),
              "mapping_hash": sha256_file(args.mapping_output), "checker_hash": sha256_file(__file__),
              "forward_parity_passed": False, "validation_status": "pending"}
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return int(bool(failures))

if __name__ == "__main__":
    raise SystemExit(main())
