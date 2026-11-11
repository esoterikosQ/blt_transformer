"""Structural invalidation for BLT patch boundaries.

This only identifies patches whose byte spans are unchanged. It does not assert
BF16 hidden-state equality across input lengths; that needs model-level parity.
"""


def patch_starts(lengths, *, input_length):
    """Recover patch starts, accounting for BLT's virtual next-token slot."""
    if input_length < 2 or not lengths:
        raise ValueError('Expected at least two input bytes and nonempty patch lengths')
    lengths = [int(value) for value in lengths]
    if any(value < 0 for value in lengths):
        raise ValueError('Negative patch length')
    positive = [value for value in lengths if value > 0]
    if lengths != positive + [0] * (len(lengths) - len(positive)):
        raise ValueError('Patch padding must follow all positive lengths')
    if sum(positive) != input_length + 1:
        raise ValueError('Patch lengths must include exactly one virtual next-token slot')
    starts = []
    offset = 0
    for length in positive:
        starts.append(offset)
        offset += length
    return starts


def shared_closed_patch_count(old_starts, new_starts):
    """Count leading old patches with identical, closed spans in the new input.

    The final old patch is always provisional because the patcher includes a
    virtual next-token slot. Once either start list diverges, even the patch
    directly before the divergence may acquire a different end position.
    """
    if not old_starts or not new_starts or old_starts[0] != 0 or new_starts[0] != 0:
        raise ValueError('Patch starts must be nonempty and begin at zero')
    if any(b <= a for starts in (old_starts, new_starts) for a, b in zip(starts, starts[1:])):
        raise ValueError('Patch starts must be strictly increasing')
    common = 0
    for old, new in zip(old_starts, new_starts):
        if old != new:
            break
        common += 1
    return max(0, common - 1)
