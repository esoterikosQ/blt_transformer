"""Scalar threshold specification, independent of runtime tensor segmentation.

Source rule: fixed starts 0,1; entropy at i>0 starts a patch at i+1 iff H_i>t.
The main BLT forward includes the next-token slot (total length S+1).
"""
import math

def patch_lengths(entropies, threshold, *, include_next_token=True, max_patch_length=None):
    if len(entropies) < 2 or not math.isfinite(threshold) or not all(map(math.isfinite, entropies)):
        raise ValueError("Finite entropies/threshold and at least two tokens required")
    if max_patch_length is not None and max_patch_length < 1:
        raise ValueError("Positive max patch length required")
    end = len(entropies) + int(include_next_token)
    starts = [0, 1] + [i + 1 for i in range(1, len(entropies)) if entropies[i] > threshold and i + 1 < end]
    lengths = [b - a for a, b in zip(starts, starts[1:] + [end])]
    if max_patch_length is not None:
        split = []
        for length in lengths:
            while length > max_patch_length:
                split.append(max_patch_length)
                length -= max_patch_length
            split.append(length)
        lengths = split
    return lengths
