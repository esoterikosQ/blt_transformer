import importlib.util
import unittest
from blt_hf_checks.plain_patcher import patch_lengths

class PlainPatcherTests(unittest.TestCase):
    def test_threshold_equality_and_next_token(self):
        self.assertEqual(patch_lengths([9, 0, 1, 2], 1), [1, 3, 1])
        self.assertEqual(patch_lengths([9, 0, 1, 2], 1, include_next_token=False), [1, 3])
        self.assertEqual(patch_lengths([9, 0, 0, 0], 1), [1, 4])

    def test_max_patch_splitting(self):
        self.assertEqual(patch_lengths([0] * 8, 1, max_patch_length=3), [1, 3, 3, 2])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
    def test_tensor_implementation_matches_scalar_for_mixed_rows(self):
        import torch
        from blt_hf.patching import entropy_patch_lengths
        torch.manual_seed(24)
        data = torch.rand(8, 24)
        data[0] = 0
        data[1] = 1
        result = entropy_patch_lengths(data, .5)
        self.assertTrue(torch.equal(result.sum(-1), torch.full((8,), 25)))
        for row, lengths in zip(data.tolist(), result.tolist()):
            self.assertEqual([x for x in lengths if x], patch_lengths(row, .5))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
    def test_bf16_threshold_boundary_uses_source_comparison_dtype(self):
        import torch
        from blt_hf.patching import entropy_patch_lengths
        threshold = 1.335442066192627
        effective = torch.tensor(threshold, dtype=torch.bfloat16).item()
        entropy = torch.tensor([[0, effective, 0]], dtype=torch.bfloat16)
        observed = entropy_patch_lengths(entropy, threshold)[0].tolist()
        self.assertEqual(observed, patch_lengths(entropy[0].tolist(), effective))
        self.assertNotEqual(observed, patch_lengths(entropy[0].tolist(), threshold))
