"""Original block-causal visibility; EOS belongs to the segment it closes."""
import torch


def segment_ids(tokens, eos_id=2):
    ends = tokens.eq(eos_id).long()
    return ends.cumsum(-1) - ends


def block_causal_mask(tokens, dtype, *, window=None, eos_id=2):
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError("Expected nonempty [batch, sequence] token IDs")
    if window is not None and (not isinstance(window, int) or window < 1):
        raise ValueError("window must be a positive integer")
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    distance = positions[:, None] - positions[None, :]
    segments = segment_ids(tokens, eos_id)
    allowed = (distance >= 0) & (segments[:, :, None] == segments[:, None, :])
    if window is not None:
        allowed &= distance < window
    mask = torch.zeros(allowed.shape, dtype=dtype, device=tokens.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min).unsqueeze(1)


def global_block_mask(tokens, patch_ids, num_patches, dtype, *, eos_id=2):
    patch_tokens = tokens.new_zeros((tokens.shape[0], num_patches))
    rows, cols = torch.where(tokens == eos_id)
    patch_tokens[rows, patch_ids[rows, cols]] = eos_id
    return block_causal_mask(patch_tokens, dtype, eos_id=eos_id)


def require_unpadded(input_ids, attention_mask, past_key_values, use_cache):
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("OSC candidate requires at least two explicit input token IDs")
    if past_key_values is not None or use_cache:
        raise ValueError("OSC candidate requires use_cache=False and no past_key_values")
    if attention_mask is not None and (attention_mask.shape != input_ids.shape or not attention_mask.eq(1).all()):
        raise ValueError("OSC candidate requires unpadded samples; padded execution is not verified")
