"""Deterministic HF generation and optional batch-1 global-prefix reuse."""
from dataclasses import asdict, dataclass
from .data_adapter import encode_prompt

HF_BACKEND = 'hf-generate-exact-length-unpadded-v1'
GLOBAL_PREFIX_BACKEND = 'global-prefix-greedy-v1'
GLOBAL_PREFIX_BACKEND_V2 = 'global-prefix-greedy-v2'
GLOBAL_PREFIX_GUARDED_BACKEND = 'global-prefix-guarded-greedy-v1'
GLOBAL_PREFIX_BACKENDS = (GLOBAL_PREFIX_BACKEND, GLOBAL_PREFIX_BACKEND_V2,
                          GLOBAL_PREFIX_GUARDED_BACKEND)

@dataclass(frozen=True)
class GenerationConfig:
    max_sequence_bytes: int = 2048
    max_new_bytes: int = 768
    num_beams: int = 1
    length_penalty: float = 1.
    batch_size: int = 1
    use_cache: bool = False
    backend: str = HF_BACKEND

    def __post_init__(self):
        if self.use_cache or min(self.max_sequence_bytes, self.max_new_bytes, self.num_beams, self.batch_size) < 1:
            raise ValueError('Positive sizes and HF use_cache=False required')
        if self.length_penalty < 0:
            raise ValueError('Nonnegative length penalty required')
        if self.backend not in (HF_BACKEND, *GLOBAL_PREFIX_BACKENDS):
            raise ValueError(f'Unknown generation backend: {self.backend}')
        if self.backend in GLOBAL_PREFIX_BACKENDS and (self.num_beams != 1 or self.batch_size != 1):
            raise ValueError('Global-prefix backend requires beam 1 and unpadded batch 1')

    def to_dict(self):
        return asdict(self)


def group_prompts(prompts, batch_size):
    if batch_size < 1:
        raise ValueError('Positive batch size required')
    buckets = {}
    for i, prompt in enumerate(prompts):
        buckets.setdefault(len(prompt), []).append(i)
    return [indices[i:i+batch_size] for indices in buckets.values() for i in range(0, len(indices), batch_size)]


def decode_generated(ids):
    ids = list(ids)
    ended = 2 in ids
    content = ids[:ids.index(2)] if ended else ids
    valid_ids = all(4 <= token < 260 for token in content)
    # Never silently drop invalid special tokens; preserve a visible replacement character.
    chunks = [bytes([token-4]) if 4 <= token < 260 else '\ufffd'.encode() for token in content]
    raw = b''.join(chunks)
    try:
        decoded = raw.decode('utf-8', errors='strict')
        valid_utf8 = True
    except UnicodeDecodeError:
        decoded, valid_utf8 = raw.decode('utf-8', errors='replace'), False
    # One sentence per scorer line: whitespace normalization applies to predictions only.
    return dict(text=' '.join(decoded.split()), raw_text=decoded, token_ids=content,
                eos_reached=ended, budget_exhausted=not ended, valid_utf8=valid_utf8,
                valid_token_ids=valid_ids)


def generate_batch(model, tokenizer, sources, cfg):
    import torch
    prompts = [encode_prompt(tokenizer, source, max_sequence_bytes=cfg.max_sequence_bytes,
                             max_new_bytes=cfg.max_new_bytes, sample_id=str(i)) for i, source in enumerate(sources)]
    results = [None] * len(sources)
    device = next(model.parameters()).device
    model.eval()
    if cfg.backend in GLOBAL_PREFIX_BACKENDS:
        from .cache.global_reuse import GlobalPrefixReuse
        with torch.inference_mode():
            for index, prompt in enumerate(prompts):
                ids = list(prompt)
                generated = []
                reuse = GlobalPrefixReuse(model, reuse_decoder=False)
                guard_refreshes = preliminary_skips = 0
                for _ in range(cfg.max_new_bytes):
                    tokens = torch.tensor([ids], dtype=torch.long, device=device)
                    output, _, _, _ = reuse.run(
                        tokens, logits_to_keep=0 if cfg.backend == GLOBAL_PREFIX_BACKEND else 1,
                        guard_margin=1.0 if cfg.backend == GLOBAL_PREFIX_GUARDED_BACKEND else None)
                    preliminary_skips += int(reuse.last_preliminary_skip)
                    guard_refreshes += int(reuse.last_guard_refresh)
                    logits = output.logits[0, -1].float().clone()
                    logits[[0, 1, 3]] = -float('inf')
                    next_id = int(logits.argmax())
                    generated.append(next_id)
                    if next_id == 2:
                        break
                    ids.append(next_id)
                results[index] = decode_generated(generated)
                if cfg.backend == GLOBAL_PREFIX_GUARDED_BACKEND:
                    results[index]['cache_preliminary_skips'] = preliminary_skips
                    results[index]['cache_guard_refreshes'] = guard_refreshes
        return results
    for indices in group_prompts(prompts, cfg.batch_size):
        ids = torch.tensor([prompts[i] for i in indices], device=device)
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       do_sample=False, use_cache=False, num_beams=cfg.num_beams,
                                       length_penalty=cfg.length_penalty, max_new_tokens=cfg.max_new_bytes,
                                       bos_token_id=1, eos_token_id=2, pad_token_id=3,
                                       suppress_tokens=[0, 1, 3], return_dict_in_generate=False)
        for index, sequence in zip(indices, generated):
            results[index] = decode_generated(sequence[len(prompts[index]):].tolist())
    return results
