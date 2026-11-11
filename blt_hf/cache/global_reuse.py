"""Experimental batch-1 global patch prefix reuse for no-cache BLT forward.

Entropy and local encoder still run on the full input. Global transformer and
optionally local decoder can reuse state. The optional v1/v2 evaluation paths
are experimental: BF16 hidden states change with input length, and full-split
token parity is unproven.
"""
import math
import torch

from .frontier import patch_starts, shared_closed_patch_count


class _LayerKV:
    def __init__(self, previous=None, count=0):
        self.layers = {
            index: (key[:, :, :count], value[:, :, :count])
            for index, (key, value) in (previous or {}).items()
        }

    def update(self, key, value, layer_idx):
        if layer_idx in self.layers:
            old_key, old_value = self.layers[layer_idx]
            key = torch.cat((old_key, key), dim=2)
            value = torch.cat((old_value, value), dim=2)
        self.layers[layer_idx] = (key.detach(), value.detach())
        return key, value


class GlobalPrefixReuse:
    """Reuse the global result only while patch starts stay unchanged."""

    def __init__(self, model, *, reuse_decoder=False):
        self.model = model
        self.reuse_decoder = reuse_decoder
        self.reset()

    def reset(self):
        self.previous_ids = None
        self.starts = None
        self.global_hidden = None
        self.decoder_kv = None
        self.last_preliminary_skip = False
        self.last_guard_refresh = False
        self.last_guard_gap = None

    def run(self, input_ids, *, logits_to_keep=0, guard_margin=None):
        if self.model.training or torch.is_grad_enabled():
            raise ValueError('Global prefix reuse requires eval and inference_mode')
        if logits_to_keep not in (0, 1):
            raise ValueError('Only full or last-position logits are supported')
        if guard_margin is not None and (logits_to_keep != 1 or self.reuse_decoder or
                                         not math.isfinite(guard_margin) or guard_margin <= 0):
            raise ValueError('Guard requires last-position logits, decoder off, and a positive finite margin')
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
            raise ValueError('Only unpadded batch-1 prefixes of at least two bytes are supported')
        current_ids = tuple(int(value) for value in input_ids[0].tolist())
        if self.previous_ids is not None and current_ids[:-1] != self.previous_ids:
            self.reset()
        self.last_preliminary_skip = False
        self.last_guard_refresh = False
        self.last_guard_gap = None
        captured = {}

        def save_patch(_module, _inputs, result):
            captured['starts'] = patch_starts(result[1][0].tolist(), input_length=len(current_ids))

        global_module = self.model.model.global_transformer
        original_forward = global_module.forward
        decoder_module = self.model.model.local_decoder
        original_decoder_forward = decoder_module.forward

        def forward_global(*args, **kwargs):
            if args or 'past_key_values' in kwargs:
                raise ValueError('Unexpected global transformer call signature')
            starts = captured['starts']
            count = (shared_closed_patch_count(self.starts, starts)
                     if self.starts is not None else 0)
            # The decoder uses the preceding patch's global output. If no new
            # boundary appeared, the current open patch is not queried for the
            # last byte; retaining its old value avoids the entire global pass.
            if current_ids[-1] != 2 and starts == self.starts and len(starts) > 1:
                captured['global_hidden'] = self.global_hidden
                captured['reused_closed_patches'] = count
                captured['skipped_global'] = True
                return self.global_hidden
            # A new boundary can change an earlier BF16 patch decision. Full
            # global recomputation is both cheaper and closer to the reference
            # than rebuilding a short tail with old patch KV on these steps.
            result = original_forward(**kwargs)
            captured['global_hidden'] = result.detach()
            captured['reused_closed_patches'] = 0
            captured['skipped_global'] = False
            return result

        def forward_decoder(*args, **kwargs):
            if args or kwargs.get('past_key_values') is not None:
                raise ValueError('Unexpected local decoder call signature')
            kwargs = dict(kwargs)
            kwargs.pop('past_key_values', None)
            same_boundaries = current_ids[-1] != 2 and captured['starts'] == self.starts
            can_reuse = bool(self.reuse_decoder and same_boundaries and self.decoder_kv)
            cache = _LayerKV(self.decoder_kv, len(self.previous_ids)) if can_reuse else _LayerKV()
            if can_reuse:
                # Cross attention still sees the current global patch table;
                # only the last byte query and its self-attention KV are new.
                kwargs = dict(kwargs)
                kwargs['input_ids'] = kwargs['input_ids'][:, -1:]
                kwargs['inputs_embeds'] = kwargs['inputs_embeds'][:, -1:]
                kwargs['attention_mask'] = kwargs['attention_mask'][:, :, -1:, :]
                kwargs['position_ids'] = kwargs['position_ids'][:, -1:]
                kwargs['encoder_attention_mask'] = kwargs['encoder_attention_mask'][:, :, -1:, :]
            result = original_decoder_forward(**kwargs, past_key_values=cache)
            captured['decoder_kv'] = cache.layers
            captured['reused_decoder'] = can_reuse
            return result

        handle = self.model.model.patcher.register_forward_hook(save_patch)
        global_module.forward = forward_global
        if self.reuse_decoder:
            decoder_module.forward = forward_decoder
        try:
            output = self.model(input_ids=input_ids, use_cache=False,
                                logits_to_keep=logits_to_keep)
        except Exception:
            self.reset()
            raise
        finally:
            global_module.forward = original_forward
            if self.reuse_decoder:
                decoder_module.forward = original_decoder_forward
            handle.remove()
        self.last_preliminary_skip = captured['skipped_global']
        if guard_margin is not None and captured['skipped_global']:
            scores = output.logits[0, -1].detach().float().clone()
            scores[[0, 1, 3]] = -float('inf')
            top_two = scores.topk(2).values
            self.last_guard_gap = float(top_two[0] - top_two[1])
            if self.last_guard_gap <= guard_margin:
                # Recompute the complete reference-style forward only when the
                # cached winner is close to another legal byte. The old global
                # state remains untouched on high-margin steps.
                def save_refreshed_global(_module, _inputs, result):
                    captured['global_hidden'] = result.detach()

                refreshed_hook = global_module.register_forward_hook(save_refreshed_global)
                try:
                    output = self.model(input_ids=input_ids,
                                        attention_mask=torch.ones_like(input_ids),
                                        use_cache=False, logits_to_keep=1)
                except Exception:
                    self.reset()
                    raise
                finally:
                    refreshed_hook.remove()
                captured['skipped_global'] = False
                captured['reused_closed_patches'] = 0
                self.last_guard_refresh = True
        self.previous_ids = current_ids
        self.starts = captured['starts']
        self.global_hidden = captured['global_hidden']
        if self.reuse_decoder:
            self.decoder_kv = captured['decoder_kv']
        return (output, captured['reused_closed_patches'], captured['skipped_global'],
                captured.get('reused_decoder', False))
