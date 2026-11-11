"""Actual unmodified HF model smoke; explicitly full-causal, not an OSC result."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blt_hf.data_adapter import encode_pair, encode_prompt
from blt_hf.manifest import write_json, verify_file_hashes, sha256_file

def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value

def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, BltForCausalLM
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--conversion-report", required=True)
    parser.add_argument("--backend", choices=["eager", "sdpa"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Evidence exists")
    conversion = json.loads(Path(args.conversion_report).read_text())
    verify_file_hashes(args.model_path, conversion["output_files"])
    report = {"status": "running", "attention_mode": "hf_full_causal_unmodified",
              "validation_status": "pending", "backend": args.backend, "dtype": "bfloat16",
              "use_cache": False, "torch": torch.__version__, "transformers": transformers.__version__,
              "conversion_report_hash": sha256_file(args.conversion_report), "checker_hash": sha256_file(__file__)}
    try:
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model, info = BltForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=args.backend, output_loading_info=True)
        model.eval()
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - start
        report["loading_info"] = json_safe(info)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Non-strict checkpoint load")
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        example = encode_pair(tok, "오늘 날씨가 조아요.", "오늘 날씨가 좋아요.")
        ids = torch.tensor([example.input_ids], device="cuda")
        labels = torch.tensor([example.labels], device="cuda")
        with torch.inference_mode():
            output = model(input_ids=ids, labels=labels, use_cache=False)
            expected_loss = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().reshape(-1, 260), labels[:, 1:].reshape(-1))
            report["loss"] = output.loss.item()
            report["manual_shifted_loss"] = expected_loss.item()
            if not torch.isfinite(output.logits).all() or not torch.allclose(output.loss, expected_loss, rtol=1e-5, atol=1e-5):
                raise ValueError("Non-finite logits or label-shift mismatch")
            report["generation"] = []
            for prompt in ("My name is", "안녕하세요. 오늘 날씨가"):
                prompt_ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
                inputs = torch.tensor([prompt_ids], device="cuda")
                torch.cuda.synchronize()
                start = time.perf_counter()
                generated = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           use_cache=False, pad_token_id=3, eos_token_id=2)
                torch.cuda.synchronize()
                continuation = generated[0, len(prompt_ids):].tolist()
                raw = bytes(i - 4 for i in continuation if 4 <= i < 260)
                try:
                    decoded = raw.decode("utf-8", errors="strict")
                    valid_utf8 = True
                except UnicodeDecodeError:
                    decoded, valid_utf8 = raw.decode("utf-8", errors="replace"), False
                report["generation"].append({"prompt": prompt, "ids": continuation, "text": decoded,
                                             "valid_utf8": valid_utf8, "seconds": time.perf_counter() - start})
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return int(report["status"] != "passed")

if __name__ == "__main__":
    raise SystemExit(main())
