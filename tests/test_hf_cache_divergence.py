import unittest

from blt_hf_checks.diagnose_cache_divergence import (
    classify_case, continuation, first_difference,
)


class CacheDivergenceContracts(unittest.TestCase):
    def test_first_difference_includes_eos(self):
        reference = continuation({'token_ids': [21, 22], 'eos_reached': True})
        candidate = continuation({'token_ids': [21, 23], 'eos_reached': True})
        self.assertEqual(first_difference(reference, candidate), (1, 22, 23))
        self.assertEqual(first_difference([21, 2], [21, 22, 2]), (1, 2, 22))
        with self.assertRaisesRegex(ValueError, 'No token difference'):
            first_difference(reference, reference)

    def test_classification_requires_reproduced_candidate_and_full_reference(self):
        common = {'replayed': True, 'skipped_global': True,
                  'cached_id': 22, 'recorded_candidate_id': 22,
                  'full_id': 23, 'recorded_reference_id': 23,
                  'repeat_full_id': 23}
        self.assertEqual(classify_case(**common), 'global_skip_changes_greedy_choice')
        self.assertEqual(classify_case(**dict(common, cached_id=24)), 'candidate_not_reproduced')
        self.assertEqual(classify_case(**dict(common, skipped_global=False)),
                         'divergence_without_global_skip')
        self.assertEqual(classify_case(**dict(common, full_id=24, repeat_full_id=24)),
                         'full_forward_differs_from_hf_reference')


if __name__ == '__main__':
    unittest.main()
