"""Download pinned original artifacts to the project cache, without conversion."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blt_hf_checks.fetch_source_configs import SOURCES
from blt_hf_checks.auth import read_project_token
from blt_hf.manifest import sha256_file, write_json
from huggingface_hub import HfApi, hf_hub_download

def main():
    output = ROOT / "blt_hf_checks/manifests/original_downloads_2026-09-15.json"
    if output.exists():
        raise FileExistsError("Download manifest exists; preserve existing evidence")
    token = read_project_token(ROOT / "artifacts/hf_home/.hf_access", ROOT)
    report = {"sources": {}, "converted": False, "legacy_actual_revision_verified": False}
    for repo, (revision, configs) in SOURCES.items():
        api = HfApi(token=token)
        info = api.model_info(repo, revision=revision, files_metadata=True)
        available = {s.rfilename: s for s in info.siblings}
        names = list(configs) + ["model.safetensors"]
        if repo == "facebook/blt-1b":
            names.append("entropy_model/consolidated.pth")
        names.extend(n for n in available if n.lower() in ("license", "license.txt", "license.md", "readme.md"))
        entries = {}
        for name in names:
            print(json.dumps({"downloading": repo + "/" + name,
                              "bytes": available[name].size}), flush=True)
            try:
                path = hf_hub_download(repo, name, revision=revision, token=token,
                                       cache_dir=str(ROOT / "artifacts/hub"))
            except Exception as exc:
                print(json.dumps({"failed": repo + "/" + name, "type": type(exc).__name__,
                                  "http_status": getattr(getattr(exc, "response", None), "status_code", None)}), flush=True)
                return 1
            entries[name] = {"path": path, "bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
            print(json.dumps({"complete": repo + "/" + name, "bytes": entries[name]["bytes"]}), flush=True)
        report["sources"][repo] = {"revision": revision, "files": entries}
    report["entropy_serialized_files_equal"] = (
        report["sources"]["facebook/blt-1b"]["files"]["entropy_model/consolidated.pth"]["sha256"] ==
        report["sources"]["facebook/blt-entropy"]["files"]["model.safetensors"]["sha256"])
    write_json(output, report)
    print(json.dumps({"status": "downloaded", "manifest": str(output)}), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
