"""Complete-corpus GLEU selection inside a distributed training job."""
from pathlib import Path

from .evaluation import publish_lines
from .generation import HF_BACKEND, group_prompts
from .manifest import sha256_file
from .metrics import compute_gleu
from .runtime import ensure_json


def gleu_improved(score, previous_best):
    """Keep the earlier epoch on an exact GLEU tie."""
    if score is None:
        return False
    return previous_best is None or score > previous_best


def gleu_early_stopping_update(score, previous_best, stale_epochs, patience):
    """Count complete validation epochs without a strictly better corpus GLEU."""
    if score is None or stale_epochs < 0 or patience < 0:
        raise ValueError('Invalid GLEU early-stopping input')
    improved = gleu_improved(score, previous_best)
    stale_epochs = 0 if improved else stale_epochs + 1
    return improved, stale_epochs, patience > 0 and stale_epochs >= patience


def validation_groups(encodings, *, rank, world_size, batch_size):
    """Group one rank's validation rows by exact prompt length, preserving row IDs."""
    if world_size < 1 or not 0 <= rank < world_size or batch_size < 1:
        raise ValueError('Invalid validation rank or batch size')
    indices = list(range(rank, len(encodings), world_size))
    prompts = [encodings[index].input_ids[:encodings[index].source_len] for index in indices]
    return [[indices[local_index] for local_index in group]
            for group in group_prompts(prompts, batch_size)]


def order_predictions(rank_parts, total):
    """Reassemble every validation row exactly once, independent of rank order."""
    if total < 1:
        raise ValueError('Validation set must be nonempty')
    ordered = [None] * total
    for part in rank_parts:
        for item in part:
            index, value = item['index'], item['text']
            if not isinstance(index, int) or not 0 <= index < total or ordered[index] is not None:
                raise ValueError(f'Duplicate or out-of-range validation prediction: {index}')
            if not isinstance(value, str):
                raise ValueError(f'Non-text validation prediction: {index}')
            ordered[index] = value
    if any(value is None for value in ordered):
        raise ValueError('Missing validation predictions')
    return ordered


def score_validation_epoch(run_dir, *, epoch, global_step, sources, references,
                           predictions, num_beams, max_new_bytes, validation_file_hash,
                           batch_size=1, generation_backend=HF_BACKEND,
                           cache_preliminary_skips=0, cache_guard_refreshes=0):
    if not sources or len({len(sources), len(references), len(predictions)}) != 1:
        raise ValueError('Validation GLEU requires the complete aligned split')
    if (not isinstance(cache_preliminary_skips, int) or
            not isinstance(cache_guard_refreshes, int) or
            not 0 <= cache_guard_refreshes <= cache_preliminary_skips):
        raise ValueError('Invalid cache refresh counts')
    root = Path(run_dir) / 'validation' / f'epoch-{epoch:04d}'
    source = root / 'source.txt'
    reference = root / 'reference.txt'
    hypothesis = root / 'hypothesis.txt'
    publish_lines(source, sources)
    publish_lines(reference, references)
    publish_lines(hypothesis, predictions)
    result = {'epoch': epoch, 'global_step': global_step, 'split': 'val',
              'sample_count': len(sources), 'num_beams': num_beams,
              'batch_size': batch_size,
              'generation_backend': generation_backend,
              'cache_preliminary_skips': cache_preliminary_skips,
              'cache_guard_refreshes': cache_guard_refreshes,
              'max_new_bytes': max_new_bytes, 'validation_file_hash': validation_file_hash,
              'source_hash': sha256_file(source), 'reference_hash': sha256_file(reference),
              'hypothesis_hash': sha256_file(hypothesis),
              'gleu': compute_gleu(reference, source, hypothesis)}
    ensure_json(root / 'metrics.json', result)
    return result
