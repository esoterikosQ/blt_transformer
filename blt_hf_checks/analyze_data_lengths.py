"""Read the nine canonical splits, without rewriting any source file."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt_hf.data_adapter import SEPARATOR, canonical_split_paths, read_tsv
from blt_hf.manifest import sha256_file, split_identity, write_json


def analyze(root: Path, limit: int = 2048) -> dict:
    splits = {}
    total, max_train, max_prompt, max_target, over1024, over_limit = 0, 0, 0, 0, 0, 0
    for name, path in canonical_split_paths(root).items():
        rows = read_tsv(path)
        lengths = [2 + len(SEPARATOR.encode()) + len(r.source.encode()) + len(r.target.encode()) for r in rows]
        prompts = [1 + len(SEPARATOR.encode()) + len(r.source.encode()) for r in rows]
        targets = [len(r.target.encode()) for r in rows]
        with path.open(encoding="utf-8") as f:
            blanks = sum(not line.strip() for line in f)
        item = {"rows": len(rows), "blank_lines": blanks, "tsv_sha256": sha256_file(path),
                "max_sequence_bytes": max(lengths), "max_prompt_bytes": max(prompts),
                "max_target_bytes": max(targets), "over_1024": sum(n > 1024 for n in lengths),
                "over_limit": sum(n > limit for n in lengths)}
        dataset, split = name.split("/")
        if split in ("val", "test"):
            item["evaluation_identity"] = split_identity(path, path.with_suffix(".m2"), dataset=dataset, split=split)
        splits[name] = item
        total += len(rows)
        max_train, max_prompt, max_target = max(max_train, max(lengths)), max(max_prompt, max(prompts)), max(max_target, max(targets))
        over1024 += item["over_1024"]
        over_limit += item["over_limit"]
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "check": "raw_utf8_lengths_and_m2_source_alignment", "model_executed": False,
            "tokenizer_executed": False, "status": "pass" if not over_limit else "fail",
            "max_allowed": limit, "splits": splits,
            "total": {"rows": total, "max_sequence_bytes": max_train, "max_prompt_bytes": max_prompt,
                      "max_target_bytes": max_target, "over_1024": over1024, "over_limit": over_limit}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/Preprocessed"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    report = analyze(args.data_root, args.max_length)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
