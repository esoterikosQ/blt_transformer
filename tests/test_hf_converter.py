"""Small CPU tensor tests; run in the prepared HF environment on itcerdo."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers", "safetensors"))


@unittest.skipUnless(AVAILABLE, "requires prepared HF environment (not a GPU)")
class ConverterTests(unittest.TestCase):
    def setUp(self):
        import torch
        from blt_hf_checks.vendor import convert_blt_weights_to_hf as converter
        self.torch = torch
        self.converter = converter

    def config(self):
        return {"encoder_hash_byte_group_nb_functions": 1, "encoder_hash_byte_group_size": [3, 4],
                "encoder_hash_byte_group_vocab": 5, "encoder_config": {"hidden_size": 4}}

    def test_fused_hash_preserves_dtype_and_values(self):
        torch = self.torch
        blocks = [torch.arange(20, dtype=torch.bfloat16).reshape(5, 4) + i for i in range(2)]
        state = {f"model.encoder_hash_tok_embedding.{i}.weight": block for i, block in enumerate(blocks)}
        out = self.converter.convert_hash_embeddings_to_fused(state, self.config())
        fused = out["model.encoder_hash_tok_embedding.weight"]
        self.assertEqual(fused.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(fused[:5], blocks[0]))
        self.assertTrue(torch.equal(fused[5:], blocks[1]))

    def test_missing_hash_slice_is_error(self):
        with self.assertRaisesRegex(ValueError, "hash"):
            self.converter.convert_hash_embeddings_to_fused(
                {"model.encoder_hash_tok_embedding.0.weight": self.torch.zeros(5, 4)}, self.config())

    def test_weight_mapping_collision_fails(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            self.converter.apply_weight_mapping({"x.attention.wq.weight": self.torch.zeros(1),
                                                 "x.self_attn.q_proj.weight": self.torch.zeros(1)})

    def test_tokenizer_contract_with_actual_fast_tokenizer(self):
        from transformers import AutoTokenizer
        from blt_hf.data_adapter import encode_pair, encode_prompt
        with tempfile.TemporaryDirectory() as d:
            config = {"vocab_size": 260, "max_position_embeddings": 4096}
            self.converter.create_tokenizer_json(d, config)
            self.converter.create_tokenizer_config(d, config)
            tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
            self.assertEqual(tok.model_max_length, 4096)
            ex = encode_pair(tok, "안뇽 hello 😀", "안녕 hello 😀")
            self.assertEqual(ex.input_ids.count(tok.bos_token_id), 1)
            self.assertEqual(ex.input_ids.count(tok.eos_token_id), 1)
            self.assertEqual(ex.labels[-1], tok.eos_token_id)
            self.assertEqual(ex.input_ids[:ex.source_len], encode_prompt(tok, "안뇽 hello 😀"))

    def test_revision_must_be_sha(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            self.converter.require_revision("main")

    def test_original_config_semantics_survive_hf_roundtrip(self):
        import json
        from transformers import BltConfig
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "blt_hf_checks/manifests/authenticated_source_configs_2026-09-15.json").read_text())
        main = report["sources"]["facebook/blt-1b"]["files"]
        with tempfile.TemporaryDirectory() as d:
            config_file, entropy_file = Path(d) / "main.json", Path(d) / "entropy.json"
            config_file.write_text(json.dumps(main["config.json"]["config"]))
            entropy_file.write_text(json.dumps(main["entropy_model/params.json"]["config"]))
            merged = self.converter.merge_configurations(config_file, entropy_file)
            loaded = BltConfig(**merged)
        for name in ("encoder_config", "decoder_config", "global_config"):
            config = getattr(loaded, name)
            self.assertEqual(config.rope_parameters["rope_theta"], 500000.0)
            self.assertEqual(config.rms_norm_eps, 1e-5)
        self.assertEqual(loaded.patcher_config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(loaded.original_spec["entropy_sliding_window"], 512)
        self.assertEqual(loaded.original_spec["local_attention_window_len"], 512)
        self.assertEqual(loaded.original_spec["global_attn_bias_type"], "block_causal")
        self.assertEqual(loaded.original_spec["patch_size"], 4.5)
        self.assertEqual((loaded.bos_token_id, loaded.eos_token_id, loaded.pad_token_id), (1, 2, 3))
        self.assertFalse(loaded.use_cache)


if __name__ == "__main__":
    unittest.main()
