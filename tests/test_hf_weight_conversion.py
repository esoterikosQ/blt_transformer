import importlib.util
import unittest
from blt_hf_checks.check_weight_conversion import target_for, compare_tensors

class MappingTests(unittest.TestCase):
    def test_semantic_component_mapping(self):
        self.assertEqual(target_for("entropy", "layers.0.attention.wq.weight", 10),
                         ("model.patcher.layers.0.self_attn.q_proj.weight", None))
        self.assertEqual(target_for("main", "local_decoder.output.weight", 10), ("lm_head.weight", None))
        self.assertEqual(target_for("main", "local_encoder.cross_attn_layers.0.cross_attn_norm_q.weight", 10),
                         ("model.local_encoder.cross_attn_layers.0.q_norm.weight", None))

    def test_hash_slices_are_disjoint_and_ordered(self):
        self.assertEqual([target_for("main", f"encoder_hash_tok_embedding.{i}.weight", 10)[1]
                          for i in range(3)], [[0, 10], [10, 20], [20, 30]])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
    def test_equal_shapes_do_not_hide_value_or_dtype_corruption(self):
        import torch
        a = torch.tensor([1, 2], dtype=torch.bfloat16)
        self.assertTrue(compare_tensors(a, a.clone()))
        self.assertFalse(compare_tensors(a, a + 1))
        self.assertFalse(compare_tensors(a, a.float()))
