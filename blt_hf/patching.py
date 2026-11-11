"""Tensor threshold segmentation with the original BLT next-token slot."""
import torch

def prediction_entropy(logits):
    """Preserve source operation order and dtype (natural-log entropy)."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)

def entropy_patch_lengths(entropies, threshold):
    if entropies.ndim != 2 or entropies.shape[1] < 2:
        raise ValueError("Expected at least two token entropies")
    if not torch.isfinite(entropies).all():
        raise ValueError("Nonfinite entropies")
    batch, length = entropies.shape
    candidate_starts = torch.arange(2, length + 1, device=entropies.device).expand(batch, -1)
    active = entropies[:, 1:] > threshold
    end = length + 1
    candidates = torch.where(active, candidate_starts, end).sort(dim=-1).values
    candidates = candidates[:, :int(active.sum(-1).max().item())]
    initial = torch.tensor([0, 1], device=entropies.device).expand(batch, -1)
    starts = torch.cat([initial, candidates], dim=-1)
    ends = torch.cat([starts[:, 1:], torch.full_like(starts[:, :1], end)], dim=-1)
    return ends - starts
