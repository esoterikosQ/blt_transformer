"""Pure training schedule contracts, testable without running optimizer/backward."""
import math
import random


def epoch_batches(count, effective_batch, *, seed, epoch):
    if count < 1 or effective_batch < 1:
        raise ValueError('Positive dataset size and effective batch required')
    ids = list(range(count))
    random.Random(seed + epoch).shuffle(ids)
    return [ids[i:i+effective_batch] for i in range(0, count, effective_batch)]


def rank_work(batch, rank, world_size):
    if not 0 <= rank < world_size or not batch:
        raise ValueError('Invalid rank or empty global batch')
    values = list(batch[rank::world_size])
    return values + [None] * (math.ceil(len(batch)/world_size) - len(values))


def normalize_gradient_scale(world_size, supervised_tokens):
    if supervised_tokens <= 0:
        raise ValueError('No supervised tokens')
    return world_size / supervised_tokens


def lr_factor(completed_steps, total_steps, warmup_steps):
    if completed_steps >= total_steps:
        return 0.
    if completed_steps < warmup_steps:
        return (completed_steps + 1) / max(warmup_steps, 1)
    progress = (completed_steps-warmup_steps)/max(total_steps-warmup_steps, 1)
    return .5 * (1 + math.cos(math.pi * min(progress, 1.)))
