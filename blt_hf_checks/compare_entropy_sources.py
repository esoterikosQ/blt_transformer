"""Exact CPU comparison of the two original entropy checkpoints."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from blt_hf.manifest import write_json

def main():
    manifest = json.loads((ROOT / "blt_hf_checks/manifests/original_downloads_2026-09-15.json").read_text())
    main_files = manifest["sources"]["facebook/blt-1b"]["files"]
    separate_files = manifest["sources"]["facebook/blt-entropy"]["files"]
    embedded = torch.load(main_files["entropy_model/consolidated.pth"]["path"], map_location="cpu", weights_only=True)
    for wrapper in ("model", "state_dict"):
        if wrapper in embedded:
            embedded = embedded[wrapper]
            break
    separate = load_file(separate_files["model.safetensors"]["path"])
    differences = []
    for key in sorted(embedded.keys() | separate.keys()):
        a, b = embedded.get(key), separate.get(key)
        if a is None or b is None:
            differences.append({"key": key, "reason": "missing"})
        elif a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            differences.append({"key": key, "reason": "shape_dtype_or_value"})
    report = {"tensor_equal": not differences, "embedded_tensor_count": len(embedded),
              "separate_tensor_count": len(separate), "differences": differences,
              "serialized_files_equal": manifest["entropy_serialized_files_equal"],
              "selected_entropy_model_id": "facebook/blt-entropy",
              "selected_entropy_revision": manifest["sources"]["facebook/blt-entropy"]["revision"],
              "legacy_actual_revision_verified": False,
              "selection_reason": "Pinned separate source specified by P1; historical downloaded revision is not recovered"}
    write_json(ROOT / "blt_hf_checks/results/entropy_sources_20260915.json", report)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
