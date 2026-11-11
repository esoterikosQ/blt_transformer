"""Forward-only checks of the original-mask candidate on the real B weights."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blt_hf.manifest import sha256_file, write_json, sha256_json
from blt_hf.model import load_model
from blt_hf.data_adapter import encode_pair, encode_prompt

def main():
    import torch
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Choose new evidence path")
    root = ROOT / "artifacts/converted/blt-1b-hf-own"
    conv = ROOT / "blt_hf_checks/manifests/conversion_B_20260915.json"
    report = {"validation_status": "pending", "attention_mode": "osc", "backend": "eager",
              "implementation_hash": sha256_file(ROOT / "blt_hf/patched/modeling_blt.py"),
              "mask_code_hash": sha256_file(ROOT / "blt_hf/attention.py"), "use_cache": False,
              "patching_code_hash": sha256_file(ROOT / "blt_hf/patching.py"),
              "conversion_report_hash": sha256_file(conv)}
    try:
        model = load_model(root, conv, attention_mode="osc").eval()
        tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
        report["runtime_config_hash"] = sha256_json(model.config.to_dict())
        captures = {}
        def hook(name):
            def capture(module, inputs, kwargs):
                mask = kwargs["attention_mask"]
                captures[name] = {"shape": list(mask.shape), "blocked": int((mask < 0).sum().item())}
            return capture
        handles = [component.layers[0].self_attn.register_forward_pre_hook(hook(name), with_kwargs=True)
                   for name, component in (("entropy", model.model.patcher), ("encoder", model.model.local_encoder),
                                            ("global", model.model.global_transformer), ("decoder", model.model.local_decoder))]
        report["forward"] = []
        with torch.inference_mode():
            for source, target in (("오늘 날씨가 조아요.", "오늘 날씨가 좋아요."), ("가나다 " * 90, "가나다 " * 90)):
                item = encode_pair(tok, source, target)
                ids = torch.tensor([item.input_ids], device="cuda")
                labels = torch.tensor([item.labels], device="cuda")
                start = time.perf_counter()
                out = model(input_ids=ids, labels=labels, use_cache=False)
                torch.cuda.synchronize()
                if not torch.isfinite(out.logits).all() or not torch.isfinite(out.loss):
                    raise ValueError("Nonfinite model output")
                report["forward"].append({"sequence_length": len(item.input_ids), "loss": out.loss.item(),
                                          "seconds": time.perf_counter()-start, "masks": dict(captures)})
            ids = torch.tensor([encode_prompt(tok, "오늘 날씨가 조아요.")], device="cuda")
            generated = model.generate(ids, max_new_tokens=64, do_sample=False, use_cache=False,
                                       pad_token_id=3, eos_token_id=2)
            report["generated_ids"] = generated[0, ids.shape[1]:].tolist()
            report["generated_text"] = tok.decode(report["generated_ids"], skip_special_tokens=True)
        for handle in handles:
            handle.remove()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return int(report["status"] != "passed")

if __name__ == "__main__":
    raise SystemExit(main())
