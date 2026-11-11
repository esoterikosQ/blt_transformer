"""Supplement the pinned source review; do not import the legacy stack."""
import sys
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blt_hf.manifest import sha256_file, write_json

COMMIT = "9774ed4fcc78313f9f218295f3d7e4decdadf2ae"
entries = {}
for name in ("bytelatent/entropy_model.py", "bytelatent/transformer.py"):
    path = ROOT / "blt_hf_checks/vendor/model_sources" / name.replace("/", "__")
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/facebookresearch/blt/{COMMIT}/{name}", timeout=30) as r:
        data = r.read()
    with path.open("xb") as f:
        f.write(data)
    entries[name] = {"sha256": sha256_file(path), "commit": COMMIT}
write_json(ROOT / "blt_hf_checks/manifests/entropy_source_review.json", entries)
