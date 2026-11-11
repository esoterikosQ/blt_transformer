import copy
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec("torch"), "prepared HF environment required")
class ModelTests(unittest.TestCase):
    def config(self, mode, window=512):
        from transformers import BltConfig
        part = dict(vocab_size=260, hidden_size=8, hidden_size_global=16, num_attention_heads=2, num_hidden_layers=2,
                    intermediate_size=24, max_position_embeddings=128)
        cfg = BltConfig(encoder_config=dict(part, num_hidden_layers=1), decoder_config=part.copy(),
                        global_config=dict(part, hidden_size=16, intermediate_size=48),
                        patcher_config=part.copy(), cross_attn_k=2,
                        encoder_hash_byte_group_size=[2, 3], encoder_hash_byte_group_vocab=17,
                        encoder_hash_byte_group_nb_functions=1, use_cache=False,
                        bos_token_id=1, eos_token_id=2, pad_token_id=3)
        cfg.attention_mode = mode
        cfg.original_spec = dict(local_attention_window_len=window, entropy_sliding_window=window,
                                 eos_id=2, global_attn_bias_type="block_causal",
                                 entropy_attn_bias_type="local_block_causal")
        for c in (cfg, cfg.encoder_config, cfg.decoder_config, cfg.global_config, cfg.patcher_config):
            c._attn_implementation = "eager"
            c.attention_mode = mode
            c.original_spec = cfg.original_spec
        return cfg

    def test_vendored_full_causal_path_matches_unmodified_hf(self):
        import torch
        from transformers import BltForCausalLM as Original
        from blt_hf.patched.modeling_blt import BltForCausalLM as Patched
        torch.manual_seed(17)
        original = Original(self.config("lre")).eval()
        patched = Patched(self.config("lre")).eval()
        patched.load_state_dict(original.state_dict(), strict=True)
        ids = torch.tensor([[1, 15, 25, 35, 45, 55]])
        with torch.inference_mode():
            self.assertTrue(torch.equal(original(ids, use_cache=False).logits,
                                        patched(ids, use_cache=False).logits))

    def test_osc_short_no_eos_matches_full_causal(self):
        import torch
        from blt_hf.patched.modeling_blt import BltForCausalLM
        a, b = BltForCausalLM(self.config("lre")).eval(), BltForCausalLM(self.config("osc")).eval()
        b.load_state_dict(a.state_dict(), strict=True)
        ids = torch.tensor([[1, 15, 25, 35, 45, 55]])
        with torch.inference_mode():
            # Isolate mask semantics with identical original-style patch lengths.
            patches = torch.tensor([[1, 2, 2, 2]])
            torch.testing.assert_close(a(ids, patch_lengths=patches, use_cache=False).logits,
                                       b(ids, patch_lengths=patches, use_cache=False).logits, rtol=0, atol=0)

    def test_real_local_layers_receive_window_and_eos_mask(self):
        import torch
        from blt_hf.patched.modeling_blt import BltForCausalLM
        model = BltForCausalLM(self.config("osc", window=3)).eval()
        captured = []
        def capture(module, args, kwargs):
            captured.append(kwargs["attention_mask"].detach().clone())
        handles = [layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
                   for component in (model.model.local_encoder, model.model.local_decoder, model.model.patcher)
                   for layer in component.layers]
        with torch.inference_mode():
            model(torch.tensor([[1, 15, 25, 35, 2, 55, 65]]), use_cache=False)
        for h in handles:
            h.remove()
        self.assertEqual(len(captured), 5)
        for mask in captured:
            self.assertEqual(mask[0, 0, 4, 3], 0)
            self.assertLess(mask[0, 0, 5, 4], -1e10)
            self.assertLess(mask[0, 0, 3, 0], -1e10)

    def test_equal_length_unpadded_batch_matches_individual_rows(self):
        import torch
        from blt_hf.patched.modeling_blt import BltForCausalLM
        model = BltForCausalLM(self.config("osc", window=3)).eval()
        ids = torch.tensor([[1, 15, 25, 35, 2, 55, 65], [1, 99, 77, 66, 55, 44, 33]])
        with torch.inference_mode():
            batched = model(ids, use_cache=False).logits
            singles = torch.cat([model(row[None], use_cache=False).logits for row in ids])
        torch.testing.assert_close(batched, singles, rtol=1e-5, atol=1e-6)
