"""Smoke the experimental cache with the project's real OSC model wiring."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'prepared HF environment required')
class GlobalReuseTests(unittest.TestCase):
    def test_consecutive_cpu_prefixes_and_reset(self):
        import torch
        from blt_hf.cache.global_reuse import GlobalPrefixReuse
        from blt_hf.patched.modeling_blt import BltForCausalLM
        from tests.test_hf_model import ModelTests

        torch.manual_seed(17)
        model = BltForCausalLM(ModelTests().config('osc')).eval()
        cached = GlobalPrefixReuse(model, reuse_decoder=True)
        ids = [1, 15, 25, 35, 45, 55, 65, 75]
        prior_starts = None
        with torch.inference_mode():
            for length in range(2, len(ids) + 1):
                tokens = torch.tensor([ids[:length]])
                reference = model(input_ids=tokens, use_cache=False).logits[:, -1]
                candidate, _, _, _ = cached.run(tokens)
                self.assertEqual(candidate.logits[:, -1].shape, reference.shape)
                self.assertTrue(torch.isfinite(candidate.logits).all())
                self.assertEqual(int(candidate.logits[:, -1].argmax()),
                                 int(reference.argmax()), f'next ID changed at prefix {length}')
                if prior_starts is not None and cached.starts != prior_starts:
                    torch.testing.assert_close(candidate.logits[:, -1], reference,
                                               rtol=0, atol=0)
                prior_starts = list(cached.starts)
            _, _, skipped, decoder_reused = cached.run(torch.tensor([ids + [2]]))
            self.assertFalse(skipped)
            self.assertFalse(decoder_reused)
            single_logits, _, _, _ = cached.run(torch.tensor([ids + [2]]), logits_to_keep=1)
            self.assertEqual(single_logits.logits.shape[1], 1)
            with self.assertRaises(ValueError):
                cached.run(torch.tensor([ids + [2]]), logits_to_keep=2)
            cached.run(torch.tensor([[1, 99, 98]]))
            self.assertEqual(cached.previous_ids, (1, 99, 98))
        with self.assertRaises(ValueError):
            cached.run(torch.tensor([[1, 15]]))


if __name__ == '__main__':
    unittest.main()
