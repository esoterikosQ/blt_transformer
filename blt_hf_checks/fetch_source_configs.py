"""Fetch pinned metadata with a project token; never print credential/error bodies."""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HOME"] = str(ROOT / "artifacts/hf_home")
os.environ["HF_HUB_CACHE"] = str(ROOT / "artifacts/hub")

from huggingface_hub import hf_hub_download
from blt_hf_checks.auth import read_project_token
from blt_hf.manifest import sha256_file, write_json

SOURCES = {
    "facebook/blt-1b": ("8134b32f0b1d25d1248c30e8c7bdfd442d3bb380",
                        ["config.json", "entropy_model/params.json"]),
    "facebook/blt-entropy": ("f2aae511e44e2086b1204bc4ddec6ac6c9651332", ["config.json"]),
}

def main():
    token = read_project_token(ROOT / "artifacts/hf_home/.hf_access", ROOT)
    report = {"credential_parsed": True, "free_disk_bytes": shutil.disk_usage(ROOT).free,
              "sources": {}, "weights_downloaded": False}
    failed = False
    for repo, (revision, names) in SOURCES.items():
        entries = {}
        for name in names:
            try:
                path = hf_hub_download(repo, name, revision=revision, token=token,
                                       cache_dir=str(ROOT / "artifacts/hub"))
                entries[name] = {"status": "ok", "path": path,
                                 "sha256": sha256_file(path),
                                 "config": json.loads(Path(path).read_text())}
            except Exception as exc:
                failed = True
                entries[name] = {"status": "failed", "error_type": type(exc).__name__,
                                 "http_status": getattr(getattr(exc, "response", None), "status_code", None)}
        report["sources"][repo] = {"revision": revision, "files": entries}
    report["status"] = "failed" if failed else "passed"
    write_json(ROOT / "blt_hf_checks/manifests/authenticated_source_configs_2026-09-15.json", report)
    print(json.dumps(report, indent=2))
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
