"""Greedy/beam byte generation without KV cache (HF BLT cache is not functional)."""
from dataclasses import dataclass

from .tokens import BOE_ID, BOS_ID, EOS_ID, PAD_ID, decode, encode


@dataclass(frozen=True)
class Generation:
    text: str
    ids: list[int]
    finished: bool


def generate(model, prompt, *, max_new_bytes=256, num_beams=1) -> Generation:
    """Generate from one unpadded prompt (``str`` or BOS-prefixed ids)."""
    import torch
    ids = encode(prompt) if isinstance(prompt, str) else list(prompt)
    if ids[:1] != [BOS_ID]:
        raise ValueError("Prompt ids must start with BOS")
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
            use_cache=False, num_beams=num_beams, max_new_tokens=max_new_bytes,
            bos_token_id=BOS_ID, eos_token_id=EOS_ID, pad_token_id=PAD_ID,
            suppress_tokens=[BOE_ID, BOS_ID, PAD_ID])
    new = output[0, len(ids):].tolist()
    return Generation(text=decode(new), ids=new, finished=EOS_ID in new)
