import copy
import unittest

from blt_hf_checks.compare_cache_eval import compare_records, validate_manifests
from blt_hf.generation import (HF_BACKEND, GLOBAL_PREFIX_BACKEND, GLOBAL_PREFIX_BACKEND_V2,
                               GLOBAL_PREFIX_GUARDED_BACKEND, GLOBAL_PREFIX_BACKENDS)


class CacheFullComparisonContracts(unittest.TestCase):
    def manifest(self, backend):
        return {'generation_backend': backend, 'dataset': 'native', 'split': 'val',
                'sample_count': 2, 'checkpoint_hash': 'weights', 'checkpoint_step': 1155,
                'source_hash': 'sources', 'target_hash': 'targets', 'm2_hash': 'gold',
                'tsv_hash': 'tsv', 'max_new_bytes': 768, 'max_sequence_bytes': 2048,
                'num_beams': 1, 'batch_size': 1, 'length_penalty': 1.,
                'model_config_hash': 'config', 'tokenizer_hash': 'tokenizer',
                'training_run_signature': 'train', 'training_run_id': 'native-main-01',
                'training_checks': 'passed', 'conversion_checks': {'weight_checks': 'passed'},
                'model_id': 'facebook/blt-1b', 'model_revision': 'revision',
                'transformers_version': '5.16.1', 'torch_version': '2.11.0',
                'attn_implementation': 'eager', 'attention_mode': 'osc',
                'inference_dtype': 'bfloat16', 'decode_policy': 'utf8',
                'scorer_hash': 'scorer', 'shard_count': 1,
                'code_hash': 'eval',
                'code_files': {'blt_hf/generation.py': 'generation-v1',
                               'blt_hf/model.py': 'model-v1'},
                'use_cache': False,
                'prefix_reuse': backend in GLOBAL_PREFIX_BACKENDS,
                'decoder_kv_reuse': False}

    def test_manifest_must_match_except_backend_and_reuse(self):
        reference = self.manifest(HF_BACKEND)
        candidate = self.manifest(GLOBAL_PREFIX_BACKEND)
        validate_manifests(reference, candidate)
        validate_manifests(reference, self.manifest(GLOBAL_PREFIX_BACKEND_V2))
        validate_manifests(reference, self.manifest(GLOBAL_PREFIX_GUARDED_BACKEND))
        new_generation = self.manifest(GLOBAL_PREFIX_GUARDED_BACKEND)
        new_generation['code_hash'] = 'new-eval'
        new_generation['code_files'] = dict(new_generation['code_files'],
                                            **{'blt_hf/generation.py': 'generation-v2'})
        validate_manifests(reference, new_generation)
        changed_model = copy.deepcopy(new_generation)
        changed_model['code_files']['blt_hf/model.py'] = 'model-v2'
        with self.assertRaisesRegex(ValueError, 'non-generation code file'):
            validate_manifests(reference, changed_model)
        wrong = dict(candidate, checkpoint_hash='other')
        with self.assertRaisesRegex(ValueError, 'checkpoint_hash'):
            validate_manifests(reference, wrong)
        wrong = dict(candidate, decoder_kv_reuse=True)
        with self.assertRaisesRegex(ValueError, 'decoder_kv_reuse'):
            validate_manifests(reference, wrong)

    def test_detects_token_and_eos_mismatch_even_if_text_matches(self):
        record = {'index': 0, 'source': 'x', 'text': 'x', 'raw_text': 'x',
                  'token_ids': [124], 'eos_reached': True, 'budget_exhausted': False,
                  'valid_utf8': True, 'valid_token_ids': True}
        reference = [record]
        candidate = [copy.deepcopy(record)]
        self.assertEqual(compare_records(reference, candidate), [])
        candidate[0]['token_ids'] = [125]
        candidate[0]['eos_reached'] = False
        self.assertEqual(compare_records(reference, candidate),
                         [{'index': 0, 'fields': ['token_ids', 'eos_reached']}])


if __name__ == '__main__':
    unittest.main()
