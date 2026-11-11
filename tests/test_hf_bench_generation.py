import unittest

from blt_hf.data_adapter import GecExample
from blt_hf_checks.bench_generation import select_groups


class ByteTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens, truncation):
        return [byte + 4 for byte in text.encode('utf-8')]


class GenerationBenchmarkContracts(unittest.TestCase):
    def test_selects_disjoint_exact_prompt_length_groups(self):
        sources = ['a', 'b', 'cc', 'dd', 'eee', 'fff']
        rows = [GecExample(source, source, index + 1, f'row-{index}')
                for index, source in enumerate(sources)]
        groups = select_groups(rows, ByteTokenizer(), batch_size=2, group_count=3,
                               max_new_bytes=8)
        self.assertEqual(groups, [[0, 1], [2, 3], [4, 5]])
        with self.assertRaisesRegex(ValueError, 'Only 3'):
            select_groups(rows, ByteTokenizer(), batch_size=2, group_count=4,
                          max_new_bytes=8)


if __name__ == '__main__':
    unittest.main()
