"""Pin public converter source and inspect HF artifact metadata, without weights.

Run from the project root. Does not read a global HF cache or saved credentials.
An existing HF_TOKEN environment variable may be used; its value is never logged.
Gated terms are never accepted by this command.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt_hf.manifest import write_json


def main():
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, default=Path("blt_hf_checks/vendor"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    for path in (args.output, args.vendor_dir):
        if not path.resolve().is_relative_to(root):
            parser.error("All outputs must stay inside the project")
    if args.output.exists() or args.vendor_dir.exists():
        parser.error("Use new output/vendor paths; existing evidence is not overwritten")
    session = requests.Session()
    endpoint = "https://api.github.com/repos/huggingface/transformers/commits/main"
    response = session.get(endpoint, timeout=30)
    response.raise_for_status()
    commit = response.json()["sha"]
    base = f"https://raw.githubusercontent.com/huggingface/transformers/{commit}"
    payloads = {}
    for name, relative in {
        "upstream_convert_blt_weights_to_hf.py": "src/transformers/models/blt/convert_blt_weights_to_hf.py",
        "LICENSE.transformers": "LICENSE",
    }.items():
        response = session.get(f"{base}/{relative}", timeout=30)
        response.raise_for_status()
        payloads[name] = response.content
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "converter_upstream_commit": commit, "legacy_actual_commit": None,
              "converter_modified": False, "weights_downloaded": False,
              "hf_token_environment_present": bool(os.environ.get("HF_TOKEN")),
              "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
              "models": {}}
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for repo in ("facebook/blt-1b", "facebook/blt-entropy", "itazap/blt-1b-hf"):
        response = session.get(f"https://huggingface.co/api/models/{repo}", headers=headers, timeout=30)
        item = {"http_status": response.status_code}
        if response.ok:
            data = response.json()
            item.update(revision=data.get("sha"), gated=data.get("gated"),
                        files=[f["rfilename"] for f in data.get("siblings", [])])
            revision = data.get("sha")
            if revision:
                probe = session.get(f"https://huggingface.co/{repo}/resolve/{revision}/config.json",
                                    headers=headers, timeout=30)
                item["config_http_status"] = probe.status_code
                if probe.ok:
                    item["config"] = probe.json()
        report["models"][repo] = item
    args.vendor_dir.mkdir(parents=True)
    for name, data in payloads.items():
        (args.vendor_dir / name).write_bytes(data)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
