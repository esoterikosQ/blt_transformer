"""Loss-masked causal GEC encoding, with explicit BOS/EOS and no truncation.

The pure-Python contracts are usable on macOS. Torch is imported only by collate.
Labels are unshifted: the causal-LM loss must shift them exactly once.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

SEPARATOR = "\n<BLT_GEC_SEP>\n"
MAX_SEQUENCE_LENGTH = 2048
DATASETS = ("native", "korean_learner", "union")
EXPERIMENT_DATASETS = (*DATASETS, "lang8")
SPLITS = ("train", "val", "test")
BYTE_OFFSET = 4


class Tokenizer(Protocol):
    bos_token_id: int
    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool) -> list[int]: ...


@dataclass(frozen=True)
class GecExample:
    source: str
    target: str
    line_number: int
    sample_id: str


@dataclass(frozen=True)
class GecEncoding:
    input_ids: list[int]
    labels: list[int]
    source_len: int


def canonical_split_paths(root: Path | str) -> dict[str, Path]:
    root = Path(root)
    return {f"{d}/{s}": root / d / f"{d}_{s}.txt" for d in DATASETS for s in SPLITS}


def dataset_split_path(root: Path | str, dataset: str, split: str) -> Path:
    root = Path(root)
    if dataset not in EXPERIMENT_DATASETS or split not in SPLITS:
        raise ValueError(f"Unsupported dataset/split: {dataset}/{split}")
    if dataset == "lang8":
        return root / "lang8" / f"lang8_{split}.txt"
    return canonical_split_paths(root)[f"{dataset}/{split}"]


def read_tsv(path: Path | str) -> list[GecExample]:
    """Skip only blank lines, preserving all nonempty records and text whitespace."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8", newline=None) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.removesuffix("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected two TSV columns, got {len(fields)}")
            rows.append(GecExample(*fields, line_number, f"{path}:{line_number}"))
    if not rows:
        raise ValueError(f"{path}: no examples")
    return rows


def _special_ids(tok: Tokenizer) -> tuple[int, int]:
    # A different vocabulary needs an explicit, reviewed mapping, not silent adaptation.
    if tok.bos_token_id != 1 or tok.eos_token_id != 2:
        raise ValueError("Expected BLT BOS=1, EOS=2; verify tokenizer artifact")
    return tok.bos_token_id, tok.eos_token_id


def _body_ids(tok: Tokenizer, text: str, sample_id: str) -> list[int]:
    ids = list(tok.encode(text, add_special_tokens=False, truncation=False))
    expected = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
    if ids != expected:
        raise ValueError(f"{sample_id}: tokenizer is not raw byte+4 encoding; check specials/normalization")
    return ids


def encode_pair(tok: Tokenizer, source: str, target: str,
                max_length: int = MAX_SEQUENCE_LENGTH, *, sample_id: str = "sample") -> GecEncoding:
    bos, eos = _special_ids(tok)
    prefix = [bos] + _body_ids(tok, source + SEPARATOR, sample_id)
    target_ids = _body_ids(tok, target, sample_id) + [eos]
    ids = prefix + target_ids
    if len(ids) > max_length:
        raise ValueError(f"{sample_id}: sequence length {len(ids)} exceeds {max_length}; truncation forbidden")
    return GecEncoding(ids, [-100] * len(prefix) + target_ids, len(prefix))


def encode_prompt(tok: Tokenizer, source: str, *, max_sequence_bytes: int = 2048,
                  max_new_bytes: int = 768, sample_id: str = "sample") -> list[int]:
    bos, _ = _special_ids(tok)
    ids = [bos] + _body_ids(tok, source + SEPARATOR, sample_id)
    if max_new_bytes < 1 or len(ids) + max_new_bytes > max_sequence_bytes:
        raise ValueError(f"{sample_id}: prompt/generation budget {len(ids)}+{max_new_bytes} exceeds {max_sequence_bytes}")
    return ids


class GecDataset:
    """Map-style dataset accepted by torch DataLoader without importing torch here."""
    def __init__(self, path: Path | str, tokenizer: Tokenizer, max_length: int = 2048):
        self.examples = read_tsv(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = self.examples[index]
        return encode_pair(self.tokenizer, row.source, row.target, self.max_length, sample_id=row.sample_id)


def prepare_unpadded_batch(batch: Sequence[GecEncoding]) -> dict[str, list[list[int]]]:
    if not batch:
        raise ValueError("Empty batch")
    lengths = {len(row.input_ids) for row in batch}
    if len(lengths) != 1:
        raise ValueError("Unverified padding path: all examples must have the same length (or batch_size=1)")
    if any(len(row.labels) != len(row.input_ids) for row in batch):
        raise ValueError("Input/label length mismatch")
    return {"input_ids": [row.input_ids for row in batch],
            "labels": [row.labels for row in batch],
            "attention_mask": [[1] * len(row.input_ids) for row in batch]}


def collate(batch: Sequence[GecEncoding]) -> dict:
    import torch
    return {key: torch.tensor(value, dtype=torch.long) for key, value in prepare_unpadded_batch(batch).items()}
