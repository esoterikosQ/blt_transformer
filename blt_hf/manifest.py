"""Run identity and immutable evidence files, independent of HF model classes."""
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .data_adapter import read_tsv

FINGERPRINT_KEYS = (
    "model_id", "model_revision", "checkpoint_hash", "transformers_version",
    "attn_implementation", "attention_mode", "model_config_hash", "tokenizer_hash",
    "generation_backend", "use_cache", "dataset", "split", "source_hash", "target_hash",
    "m2_hash", "num_beams", "length_penalty", "batch_size", "max_sequence_bytes",
    "max_new_bytes", "code_hash",
)


class ManifestMismatch(ValueError):
    pass


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(manifest: Mapping) -> dict:
    missing = [key for key in FINGERPRINT_KEYS if key not in manifest or manifest[key] is None or manifest[key] == ""]
    if missing:
        raise ManifestMismatch("Missing identity fields: " + ", ".join(missing))
    return {key: manifest[key] for key in FINGERPRINT_KEYS}


def fingerprint(manifest: Mapping) -> str:
    return sha256_json(_identity(manifest))


def assert_compatible(manifests: Sequence[Mapping]) -> str:
    if not manifests:
        raise ManifestMismatch("No shards supplied")
    reference = _identity(manifests[0])
    for index, manifest in enumerate(manifests[1:], 1):
        other = _identity(manifest)
        different = [k for k in FINGERPRINT_KEYS if canonical_json(reference[k]) != canonical_json(other[k])]
        if different:
            raise ManifestMismatch(f"Shard {index} differs: {', '.join(different)}")
    return sha256_json(reference)


def split_identity(tsv_path: Path | str, m2_path: Path | str, *, dataset: str, split: str) -> dict:
    rows = read_tsv(tsv_path)
    with Path(m2_path).open(encoding="utf-8") as f:
        sources = [line[2:].removesuffix("\n") for line in f if line.startswith("S ")]
    if len(sources) != len(rows):
        raise ValueError(f"M2/source count mismatch: {len(sources)} vs {len(rows)}")
    for index, (source, row) in enumerate(zip(sources, rows)):
        if source.split() != row.source.split():
            raise ValueError(f"M2/source mismatch at logical row {index}: {row.sample_id}")
    return {"dataset": dataset, "split": split, "sample_count": len(rows),
            "source_hash": sha256_json([row.source for row in rows]),
            "target_hash": sha256_json([row.target for row in rows]),
            "m2_hash": sha256_file(m2_path), "tsv_hash": sha256_file(tsv_path),
            "reader": "utf8-universal-newlines-two-columns-blank-lines-skipped-v1"}


def verify_file_hashes(root: Path | str, hashes: Mapping[str, str]) -> None:
    root = Path(root).resolve()
    if not hashes:
        raise ManifestMismatch("No artifact file hashes")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Artifact path escapes root: {name}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ManifestMismatch(f"Artifact file mismatch: {name}")


def write_json(path: Path | str, value) -> None:
    """Publish a complete report atomically, refusing to replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication on the same filesystem.
    finally:
        os.unlink(temporary)
