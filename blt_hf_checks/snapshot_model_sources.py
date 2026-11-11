"""Record implementation sources for review, without modifying installed packages."""
import inspect
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blt_hf.manifest import sha256_file, write_json

def main():
    output = ROOT / "blt_hf_checks/vendor/model_sources"
    output.mkdir(parents=True, exist_ok=False)
    import transformers
    from transformers.models.blt import configuration_blt, modeling_blt
    sources = {}
    for module in (configuration_blt, modeling_blt):
        target = output / (module.__name__.split(".")[-1] + ".py")
        target.write_text(inspect.getsource(module))
        sources[target.name] = {"sha256": sha256_file(target), "version": transformers.__version__}
    def fetch(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "phdq3-source-review"}), timeout=60) as r:
            return r.read()
    commit = json.loads(fetch("https://api.github.com/repos/facebookresearch/blt/commits/main"))["sha"]
    for name in ("bytelatent/model/utils.py", "bytelatent/model/blt.py", "bytelatent/model/local_models.py",
                 "bytelatent/model/latent_transformer.py", "bytelatent/base_transformer.py",
                 "bytelatent/data/patcher.py", "LICENSE"):
        target = output / name.replace("/", "__")
        target.write_bytes(fetch(f"https://raw.githubusercontent.com/facebookresearch/blt/{commit}/{name}"))
        sources[target.name] = {"sha256": sha256_file(target), "commit": commit, "original_path": name}
    write_json(ROOT / "blt_hf_checks/manifests/model_source_review.json",
               {"sources": sources, "legacy_actual_commit": None, "review_source_commit": commit})
    print(json.dumps({"source_snapshot": str(output), "review_source_commit": commit}))

if __name__ == "__main__":
    main()
