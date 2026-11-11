import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
class AttentionTests(unittest.TestCase):
    def test_window_eos_and_batch_isolation_against_scalar_definition(self):
        import torch
        from blt_hf.attention import block_causal_mask
        tokens = torch.tensor([[1, 6, 2, 7, 8, 2, 9], [1, 2, 3, 4, 5, 6, 7]])
        mask = block_causal_mask(tokens, torch.float32, window=3)
        for b, row in enumerate(tokens.tolist()):
            for q in range(len(row)):
                for k in range(len(row)):
                    expected = k <= q and q - k < 3 and row[:q].count(2) == row[:k].count(2)
                    self.assertEqual(mask[b, 0, q, k].item() == 0, expected)

    def test_global_eos_patch_closes_segment(self):
        import torch
        from blt_hf.attention import global_block_mask
        mask = global_block_mask(torch.tensor([[1, 7, 2, 8, 9]]),
                                 torch.tensor([[0, 1, 1, 2, 3]]), 4, torch.float32)
        self.assertEqual(mask[0, 0, 1, 0], 0)
        self.assertLess(mask[0, 0, 2, 1], -1e10)
        self.assertEqual(mask[0, 0, 3, 2], 0)

    def test_padded_or_cached_execution_fails_explicitly(self):
        import torch
        from blt_hf.attention import require_unpadded
        tokens = torch.tensor([[1, 8, 2]])
        with self.assertRaisesRegex(ValueError, "unpadded"):
            require_unpadded(tokens, torch.tensor([[1, 1, 0]]), None, False)
        with self.assertRaisesRegex(ValueError, "cache"):
            require_unpadded(tokens, None, None, True)
