import importlib.util
import unittest

from blt_hf_checks.check_attention_mask import check_patch_causality, membership


class CausalTopologyTests(unittest.TestCase):
    def test_shift_prevents_future_patch_leakage_including_empty_patches(self):
        report = check_patch_causality([[1, 10, 20, 30]], [[1, 1, 1, 1, 1]])
        self.assertEqual(report["future_dependency_violations"], 0)
        self.assertEqual(report["conservative_empty_encoder_patches"], 1)

    def test_invalid_prediction_shift_is_rejected(self):
        with self.assertRaises(AssertionError):
            check_patch_causality([[1, 10, 20, 30]], [[2, 3]])
        with self.assertRaises(ValueError):
            membership([1, 1], 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "prepared HF environment required")
class MaskAuditTests(unittest.TestCase):
    def test_future_window_eos_and_small_negative_sentinel_corruption_detected(self):
        import torch
        from blt_hf_checks.check_attention_mask import causal_reference, check_mask
        expected = causal_reference([[1, 40, 50, 2, 60]], "cpu", window=3)
        mask = torch.zeros(expected.shape).masked_fill(~expected, torch.finfo(torch.float32).min)
        check_mask(mask, expected)
        for q, k, value in ((0, 1, 0), (3, 0, 0), (4, 3, 0), (0, 1, -1)):
            bad = mask.clone()
            bad[0, 0, q, k] = value
            with self.assertRaises(AssertionError):
                check_mask(bad, expected)

    def test_actual_tiny_model_all_attention_layers(self):
        import torch
        from test_hf_model import ModelTests
        from blt_hf.patched.modeling_blt import BltForCausalLM
        from blt_hf_checks.check_attention_mask import MaskAudit
        torch.manual_seed(42)
        model = BltForCausalLM(ModelTests().config("osc", window=3)).eval()
        audit = MaskAudit(model)
        try:
            result = audit.run([[1, 40, 2, 60, 70, 80], [1, 40, 50, 2, 2, 80]])
            self.assertEqual(result["modules_checked"], len(audit.expected_names))
            result = audit.run([[1, 40, 50, 60, 70, 80]], lengths=[[1] * 7])
            self.assertEqual(result["conservative_empty_encoder_patches"], 1)
        finally:
            audit.close()
