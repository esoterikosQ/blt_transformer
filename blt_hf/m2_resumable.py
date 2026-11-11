"""Resumable, sentence-level execution of the bundled NUS M2 scorer."""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1
SCORER_DIR = Path(__file__).resolve().parent / "vendor" / "m2"


@dataclass(frozen=True)
class M2EvaluationResult:
    precision: float | None
    recall: float | None
    f_score: float | None
    status: str
    completed: int
    total: int
    unresolved: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, float | int | str | None]:
        result = asdict(self)
        result["f0.5"] = result.pop("f_score")
        return result


def _load_levenshtein():
    from .vendor.m2 import levenshtein
    return levenshtein


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _write_ids_atomic(path: Path, sentence_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    contents = "".join(f"{sentence_id}\n" for sentence_id in sentence_ids)
    temp_path.write_text(contents, encoding="utf-8")
    temp_path.replace(path)


def _iter_m2_paragraphs(path: Path) -> Iterator[list[str]]:
    """Yield one logical sentence per record, even when blank separators are missing."""
    paragraph: list[str] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.rstrip("\r\n")
            if not line:
                if paragraph:
                    yield paragraph
                    paragraph = []
                continue
            if line.startswith("S ") and paragraph:
                yield paragraph
                paragraph = []
            paragraph.append(line)
    if paragraph:
        yield paragraph


def load_m2_annotations(
    gold_path: str | Path,
) -> tuple[list[str], list[dict[int, list[tuple[int, int, str, list[str]]]]]]:
    """Load M2 data with the same sentence/edit interpretation as m2scorer.py."""

    source_sentences: list[str] = []
    gold_edits: list[dict[int, list[tuple[int, int, str, list[str]]]]] = []

    for item in _iter_m2_paragraphs(Path(gold_path)):
        sentences = [line[2:].strip() for line in item if line.startswith("S ")]
        if len(sentences) != 1:
            raise ValueError("Each logical M2 record must contain exactly one S line")

        annotations: dict[int, list[tuple[int, int, str, list[str]]]] = {}
        for line in item[1:]:
            if line.startswith(("I ", "S ")):
                continue
            if not line.startswith("A "):
                raise ValueError(f"Unexpected M2 line: {line}")
            fields = line[2:].split("|||")
            if len(fields) < 6:
                raise ValueError(f"Malformed M2 annotation: {line}")
            offsets = fields[0].split()
            start_offset = int(offsets[0])
            end_offset = int(offsets[1])
            if fields[1] == "noop":
                start_offset = -1
                end_offset = -1
            corrections = [
                correction.strip() if correction != "-NONE-" else ""
                for correction in fields[2].split("||")
            ]
            original = " ".join(
                " ".join(sentences).split()[start_offset:end_offset]
            )
            annotator = int(fields[5])
            annotations.setdefault(annotator, []).append(
                (start_offset, end_offset, original, corrections)
            )

        token_offset = 0
        for sentence in sentences:
            token_offset += len(sentence.split())
            source_sentences.append(sentence)
            sentence_edits = {
                annotator: [
                    edit
                    for edit in annotation
                    if edit[0] <= token_offset
                    and edit[1] <= token_offset
                    and edit[0] >= 0
                    and edit[1] >= 0
                ]
                for annotator, annotation in annotations.items()
            }
            if not sentence_edits:
                sentence_edits[0] = []
            gold_edits.append(sentence_edits)

    return source_sentences, gold_edits


def _score_sentence(
    candidate: str,
    source: str,
    golds_set: dict[int, list[tuple[int, int, str, list[str]]]],
    max_unchanged_words: int,
    ignore_whitespace_casing: bool,
) -> list[dict[str, int]]:
    """Return per-annotator sufficient statistics for one sentence."""

    levenshtein = _load_levenshtein()
    candidate_tokens = candidate.split()
    source_tokens = source.split()

    matrix1, backpointers1 = levenshtein.levenshtein_matrix(
        source_tokens, candidate_tokens, 1, 1, 1
    )
    matrix2, backpointers2 = levenshtein.levenshtein_matrix(
        source_tokens, candidate_tokens, 1, 1, 2
    )
    vertices1, edges1, distances1, edits1 = levenshtein.edit_graph(
        matrix1, backpointers1
    )
    vertices2, edges2, distances2, edits2 = levenshtein.edit_graph(
        matrix2, backpointers2
    )
    vertices, edges, distances, edits = levenshtein.merge_graph(
        vertices1,
        vertices2,
        edges1,
        edges2,
        distances1,
        distances2,
        edits1,
        edits2,
    )
    vertices, edges, distances, edits = levenshtein.transitive_arcs(
        vertices,
        edges,
        distances,
        edits,
        max_unchanged_words,
        False,
    )

    annotator_stats: list[dict[str, int]] = []
    for annotator, gold in golds_set.items():
        local_distances = levenshtein.set_weights(
            edges, distances, edits, gold, False, False
        )
        edit_sequence = levenshtein.best_edit_seq_bf(
            vertices, edges, local_distances, edits, False
        )
        if ignore_whitespace_casing:
            edit_sequence = [
                edit
                for edit in edit_sequence
                if not levenshtein.equals_ignore_whitespace_casing(edit[2], edit[3])
            ]
        correct = levenshtein.matchSeq(
            edit_sequence, gold, ignore_whitespace_casing, False
        )
        annotator_stats.append(
            {
                "annotator": int(annotator),
                "correct": len(correct),
                "proposed": len(edit_sequence),
                "gold": len(gold),
            }
        )
    return annotator_stats


def _sentence_worker(
    connection,
    sentence_id: int,
    candidate: str,
    source: str,
    golds_set: dict[int, list[tuple[int, int, str, list[str]]]],
    max_unchanged_words: int,
    ignore_whitespace_casing: bool,
) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    started_at = time.monotonic()
    try:
        stats = _score_sentence(
            candidate,
            source,
            golds_set,
            max_unchanged_words,
            ignore_whitespace_casing,
        )
        connection.send(
            {
                "status": "completed",
                "sentence_id": sentence_id,
                "annotators": stats,
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
    except BaseException as exc:
        connection.send(
            {
                "status": "internal_error",
                "sentence_id": sentence_id,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
    finally:
        connection.close()


def _multiprocessing_context():
    methods = mp.get_all_start_methods()
    return mp.get_context("fork" if "fork" in methods else methods[0])


def _run_sentence_processes(
    sentence_ids: list[int],
    hypotheses: list[str],
    sources: list[str],
    gold_edits: list[dict[int, list[tuple[int, int, str, list[str]]]]],
    timeout_seconds: float,
    workers: int,
    max_unchanged_words: int,
    ignore_whitespace_casing: bool,
    should_stop=None,
) -> Iterator[dict[str, Any]]:
    context = _multiprocessing_context()
    waiting = deque(sentence_ids)
    active: dict[int, tuple[Any, Any, float]] = {}

    try:
        while waiting or active:
            if should_stop is not None and bool(should_stop):
                return
            while waiting and len(active) < workers:
                sentence_id = waiting.popleft()
                parent_connection, child_connection = context.Pipe(duplex=False)
                process = context.Process(
                    target=_sentence_worker,
                    args=(
                        child_connection,
                        sentence_id,
                        hypotheses[sentence_id],
                        sources[sentence_id],
                        gold_edits[sentence_id],
                        max_unchanged_words,
                        ignore_whitespace_casing,
                    ),
                )
                process.start()
                child_connection.close()
                active[sentence_id] = (
                    process,
                    parent_connection,
                    time.monotonic(),
                )

            made_progress = False
            now = time.monotonic()
            for sentence_id, (process, connection, started_at) in list(active.items()):
                elapsed = now - started_at
                try:
                    has_result = connection.poll()
                except (EOFError, OSError):
                    has_result = False

                if has_result:
                    try:
                        result = connection.recv()
                    except (EOFError, OSError) as exc:
                        result = {
                            "status": "internal_error",
                            "sentence_id": sentence_id,
                            "error": f"Worker result unavailable: {exc}",
                            "elapsed_seconds": elapsed,
                        }
                    process.join(timeout=1)
                    connection.close()
                    del active[sentence_id]
                    made_progress = True
                    yield result
                    continue

                if elapsed >= timeout_seconds:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    connection.close()
                    del active[sentence_id]
                    made_progress = True
                    yield {
                        "status": "timeout",
                        "sentence_id": sentence_id,
                        "elapsed_seconds": elapsed,
                    }
                    continue

                if not process.is_alive():
                    process.join(timeout=1)
                    connection.close()
                    del active[sentence_id]
                    made_progress = True
                    yield {
                        "status": "internal_error",
                        "sentence_id": sentence_id,
                        "error": f"Worker exited with code {process.exitcode}",
                        "elapsed_seconds": elapsed,
                    }

            if not made_progress:
                time.sleep(0.05)
    finally:
        for process, connection, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            connection.close()


def _load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        # Preserve interrupted bytes before repairing the mutable append journal.
        backup = path.with_name(path.name + ".torn-" + str(time.time_ns()))
        backup.write_bytes(raw[boundary:])
        raw = raw[:boundary]
        path.write_bytes(raw)
    for line in raw.decode("utf-8").splitlines():
        record = json.loads(line)
        idx = int(record["sentence_id"])
        if idx in completed:
            raise ValueError("Duplicate M2 completion record")
        completed[idx] = record
    return completed


def _next_pass_id(passes_dir: Path) -> int:
    pending_ids = {
        int(path.stem.split("_")[1])
        for path in passes_dir.glob("pass_*_pending.txt")
    }
    summary_ids = {
        int(path.stem.split("_")[1])
        for path in passes_dir.glob("pass_*_summary.json")
    }
    interrupted = sorted(pending_ids - summary_ids)
    if interrupted:
        return interrupted[0]
    return max(summary_ids, default=-1) + 1


def _compute_f_score(correct: float, proposed: float, gold: float, beta: float) -> float:
    denominator = beta * beta * gold + proposed
    if denominator == 0:
        return 1.0 if correct == 0 else 0.0
    return (1.0 + beta * beta) * correct / denominator


def _reduce_completed(
    completed: dict[int, dict[str, Any]], beta: float
) -> tuple[float, float, float, float, float, float]:
    total_correct = 0.0
    total_proposed = 0.0
    total_gold = 0.0
    squared_beta = beta * beta

    for sentence_id in sorted(completed):
        record = completed[sentence_id]
        best: tuple[float, float, float] | None = None
        best_f_score = -1.0
        max_correct = -1.0
        min_proposed = float("inf")
        min_gold = float("inf")

        for stats in record["annotators"]:
            correct = total_correct + float(stats["correct"])
            proposed = total_proposed + float(stats["proposed"])
            gold = total_gold + float(stats["gold"])
            f_score = _compute_f_score(correct, proposed, gold, beta)
            if (
                best_f_score < f_score
                or (best_f_score == f_score and max_correct < correct)
                or (
                    best_f_score == f_score
                    and max_correct == correct
                    and min_proposed + squared_beta * min_gold
                    > proposed + squared_beta * gold
                )
            ):
                best = (
                    float(stats["correct"]),
                    float(stats["proposed"]),
                    float(stats["gold"]),
                )
                best_f_score = f_score
                max_correct = correct
                min_proposed = proposed
                min_gold = gold

        if best is None:
            raise RuntimeError(f"Sentence {sentence_id} has no annotator statistics")
        total_correct += best[0]
        total_proposed += best[1]
        total_gold += best[2]

    precision = total_correct / total_proposed if total_proposed else 1.0
    recall = total_correct / total_gold if total_gold else 1.0
    denominator = beta * beta * precision + recall
    f_score = (
        (1.0 + beta * beta) * precision * recall / denominator
        if denominator
        else 0.0
    )
    return (
        precision,
        recall,
        f_score,
        total_correct,
        total_proposed,
        total_gold,
    )


def _validate_or_create_manifest(
    manifest_path: Path,
    hypothesis_path: Path,
    source_gold_path: Path,
    total: int,
    settings: dict,
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "scorer_sha256": _file_sha256(SCORER_DIR / "levenshtein.py"),
        "wrapper_sha256": _file_sha256(Path(__file__)),
        "hypothesis_path": str(hypothesis_path.resolve()),
        "hypothesis_sha256": _file_sha256(hypothesis_path),
        "source_gold_path": str(source_gold_path.resolve()),
        "source_gold_sha256": _file_sha256(source_gold_path),
        "examples": total,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                "M2 input manifest changed. Move or remove the existing m2 directory "
                "before evaluating different inputs."
            )
        return
    _write_json_atomic(manifest_path, expected)


def evaluate_m2_resumable(
    hypothesis_path: str | Path,
    source_gold_path: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: float = 30.0,
    timeout_multiplier: float = 4.0,
    max_passes: int = 4,
    workers: int = 1,
    beta: float = 0.5,
    max_unchanged_words: int = 2,
    ignore_whitespace_casing: bool = False,
    should_stop=None,
) -> M2EvaluationResult:
    """Evaluate M2 with sentence-level checkpoints and bounded retries."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if timeout_multiplier < 1:
        raise ValueError("timeout_multiplier must be at least 1")
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    started_at = time.monotonic()
    hypothesis_path = Path(hypothesis_path)
    source_gold_path = Path(source_gold_path)
    m2_dir = Path(output_dir)
    passes_dir = m2_dir / "passes"
    m2_dir.mkdir(parents=True, exist_ok=True)
    passes_dir.mkdir(parents=True, exist_ok=True)

    hypotheses = hypothesis_path.read_text(encoding="utf-8").splitlines()
    sources, gold_edits = load_m2_annotations(source_gold_path)
    if not (len(hypotheses) == len(sources) == len(gold_edits)):
        raise ValueError(
            "M2 input length mismatch: "
            f"hypotheses={len(hypotheses)}, sources={len(sources)}, "
            f"gold_edits={len(gold_edits)}"
        )

    total = len(hypotheses)
    _validate_or_create_manifest(
        m2_dir / "run_config.json",
        hypothesis_path,
        source_gold_path,
        total,
        {"beta": beta, "max_unchanged_words": max_unchanged_words, "ignore_whitespace_casing": ignore_whitespace_casing},
    )

    completed_path = m2_dir / "completed.jsonl"
    failed_path = m2_dir / "failed.jsonl"
    completed = _load_completed(completed_path)
    if any(i < 0 or i >= total for i in completed):
        raise ValueError("M2 completion index outside corpus")
    pending = [sentence_id for sentence_id in range(total) if sentence_id not in completed]
    latest_timeout_ids: list[int] = []
    failed_ids: set[int] = set()
    processed_this_run = 0

    first_pass_id = _next_pass_id(passes_dir)
    with completed_path.open("a", encoding="utf-8", buffering=1) as completed_file, failed_path.open(
        "a", encoding="utf-8", buffering=1
    ) as failed_file:
        for pass_id in range(first_pass_id, first_pass_id + max_passes):
            if not pending or (should_stop is not None and bool(should_stop)):
                break
            pass_timeout = timeout_seconds * (timeout_multiplier**pass_id)
            pass_pending = sorted(pending)
            _write_ids_atomic(
                passes_dir / f"pass_{pass_id:03d}_pending.txt", pass_pending
            )
            print(
                f"M2 pass={pass_id} pending={len(pass_pending)} "
                f"timeout={pass_timeout:.1f}s workers={workers}",
                flush=True,
            )

            latest_timeout_ids = []
            pass_failed_ids: list[int] = []
            pass_started_at = time.monotonic()
            for result in _run_sentence_processes(
                pass_pending,
                hypotheses,
                sources,
                gold_edits,
                pass_timeout,
                workers,
                max_unchanged_words,
                ignore_whitespace_casing,
                should_stop,
            ):
                sentence_id = int(result["sentence_id"])
                processed_this_run += 1
                result["pass_id"] = pass_id
                if result["status"] == "completed":
                    result["source_tokens"] = len(sources[sentence_id].split())
                    result["hypothesis_tokens"] = len(
                        hypotheses[sentence_id].split()
                    )
                    completed_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    completed_file.flush()
                    os.fsync(completed_file.fileno())
                    completed[sentence_id] = result
                elif result["status"] == "timeout":
                    latest_timeout_ids.append(sentence_id)
                else:
                    pass_failed_ids.append(sentence_id)
                    failed_ids.add(sentence_id)
                    failed_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    failed_file.flush()

                if processed_this_run % 10 == 0:
                    progress = {
                        "status": "running",
                        "pass_id": pass_id,
                        "processed_this_run": processed_this_run,
                        "completed": len(completed),
                        "total": total,
                        "elapsed_seconds": time.monotonic() - started_at,
                    }
                    _write_json_atomic(m2_dir / "progress.json", progress)
                    print(
                        f"M2 progress completed={len(completed)}/{total} "
                        f"timeouts={len(latest_timeout_ids)} "
                        f"errors={len(pass_failed_ids)}",
                        flush=True,
                    )

            latest_timeout_ids.sort()
            _write_ids_atomic(
                passes_dir / f"pass_{pass_id:03d}_timeout.txt",
                latest_timeout_ids,
            )
            _write_json_atomic(
                passes_dir / f"pass_{pass_id:03d}_summary.json",
                {
                    "pass_id": pass_id,
                    "timeout_seconds": pass_timeout,
                    "input_examples": len(pass_pending),
                    "completed_total": len(completed),
                    "timeouts": len(latest_timeout_ids),
                    "errors": len(pass_failed_ids),
                    "elapsed_seconds": time.monotonic() - pass_started_at,
                },
            )
            pending = sorted(set(latest_timeout_ids + pass_failed_ids))

    unresolved_ids = sorted(set(range(total)) - set(completed))
    _write_ids_atomic(m2_dir / "unresolved_ids.txt", unresolved_ids)
    _write_ids_atomic(m2_dir / "invalid_ids.txt", sorted(failed_ids))

    durations = [float(record["elapsed_seconds"]) for record in completed.values()]
    average_duration = sum(durations) / len(durations) if durations else 0.0
    slow_ids = sorted(
        sentence_id
        for sentence_id, record in completed.items()
        if float(record["elapsed_seconds"]) > average_duration
    )
    _write_ids_atomic(m2_dir / "slow_ids.txt", slow_ids)

    if completed:
        precision, recall, f_score, correct, proposed, gold = _reduce_completed(
            completed, beta
        )
    else:
        precision = recall = f_score = None
        correct = proposed = gold = 0.0

    status = "complete" if len(completed) == total else "partial"
    if status != "complete":
        precision = recall = f_score = None
    elapsed_seconds = time.monotonic() - started_at
    result = M2EvaluationResult(
        precision=precision,
        recall=recall,
        f_score=f_score,
        status=status,
        completed=len(completed),
        total=total,
        unresolved=len(unresolved_ids),
        elapsed_seconds=elapsed_seconds,
    )
    final_payload = {
        **result.to_dict(),
        "correct": correct,
        "proposed": proposed,
        "gold": gold,
        "coverage": len(completed) / total if total else 1.0,
        "average_sentence_seconds": average_duration,
        "slow_sentences": len(slow_ids),
    }
    _write_json_atomic(m2_dir / "final_metrics.json", final_payload)
    _write_json_atomic(
        m2_dir / "progress.json",
        {**final_payload, "status": status},
    )
    print(json.dumps(final_payload, ensure_ascii=False, indent=2), flush=True)
    return result


def default_worker_count() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        return max(1, int(value))
    except ValueError:
        return 1
