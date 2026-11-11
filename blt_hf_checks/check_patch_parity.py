"""Segmentation and entropy arithmetic against scalar/source formula, not legacy forward parity."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blt_hf_checks.plain_patcher import patch_lengths
from blt_hf.data_adapter import read_tsv
from blt_hf.manifest import sha256_file, sha256_json, write_json
from blt_hf.model import load_model

def main():
    import torch
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Choose a new evidence path")
    root = ROOT / "artifacts/converted/blt-1b-hf-own"
    conv = ROOT / "blt_hf_checks/manifests/conversion_B_20260915.json"
    model = load_model(root, conv, attention_mode="osc").eval()
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    korean = read_tsv(ROOT / "data/Preprocessed/native/native_train.txt")[:10]
    prompts = ["My name is", "Hello world.", "A small example.", "The weather is sunny.", "One two three."]
    prompts += [row.source for row in korean]
    prompts += ["오늘은 Monday입니다.", "한국어와 English", "AI를 test합니다.", "12345 !? @#$", "()[]{} <> / + ="]
    cases = []
    with torch.inference_mode():
        for text in prompts:
            ids = [1] + tok.encode(text, add_special_tokens=False, truncation=False)
            tokens = torch.tensor([ids], device="cuda")
            entropy, lengths, logits = model.model.patcher(tokens, patch_size=model.config.patch_size,
                threshold=model.config.patching_threshold, max_patch_length=model.config.max_patch_length, use_cache=False)
            logp = logits.log_softmax(-1)
            reference_entropy = -(logp.exp() * logp).sum(-1)
            # The source compares a tensor to a Python scalar in the tensor dtype.
            # Record quantization explicitly; also preserve the ideal-real comparison.
            effective_threshold = torch.tensor(model.config.patching_threshold, dtype=entropy.dtype).item()
            expected = patch_lengths(reference_entropy[0].tolist(), effective_threshold)
            ideal = patch_lengths(reference_entropy[0].tolist(), model.config.patching_threshold)
            observed = [n for n in lengths[0].tolist() if n]
            stock_entropy = torch.distributions.Categorical(logits=logits).entropy()
            cases.append({"text": text, "input_ids": ids, "expected_lengths": expected, "actual_lengths": observed,
                          "patch_equal": observed == expected,
                          "entropy_dtype": str(entropy.dtype), "configured_threshold": model.config.patching_threshold,
                          "effective_threshold": effective_threshold, "ideal_real_threshold_lengths": ideal,
                          "ideal_real_threshold_differs": ideal != observed,
                          "entropy_equal": torch.equal(entropy, reference_entropy),
                          "minimum_threshold_margin": (entropy.float() - model.config.patching_threshold).abs().min().item(),
                          "stock_entropy_max_abs_difference": (stock_entropy.float()-entropy.float()).abs().max().item()})
    report = {"status": "passed" if all(c["patch_equal"] and c["entropy_equal"] for c in cases) else "failed",
              "scope": "same-logits entropy arithmetic and scalar segmentation with source dtype threshold; entropy forward parity remains pending",
              "legacy_reference_valid": False, "validation_status": "pending", "cases": cases,
              "fixture_hash": sha256_json(prompts), "conversion_report_hash": sha256_file(conv),
              "model_code_hash": sha256_file(ROOT / "blt_hf/patched/modeling_blt.py"),
              "patching_code_hash": sha256_file(ROOT / "blt_hf/patching.py"),
              "plain_reference_hash": sha256_file(ROOT / "blt_hf_checks/plain_patcher.py")}
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "cases": len(cases), "scope": report["scope"]}), flush=True)
    return int(report["status"] != "passed")

if __name__ == "__main__":
    raise SystemExit(main())
