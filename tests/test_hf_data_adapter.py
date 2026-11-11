"""Contract tests, executable without torch or Transformers on the Mac."""
import tempfile
import unittest
from pathlib import Path

from blt_hf.data_adapter import (
    GecDataset, SEPARATOR, canonical_split_paths, encode_pair, encode_prompt,
    read_tsv, prepare_unpadded_batch, dataset_split_path,
)


class ByteTokenizer:
    """Test double for the verified byte+4 vocabulary; not a parity oracle."""
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens, truncation):
        if add_special_tokens or truncation:
            raise AssertionError("adapter must disable automatic specials/truncation")
        return [b + 4 for b in text.encode("utf-8")]


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()

    def test_target_eos_supervision_and_prompt_boundary(self):
        ex = encode_pair(self.tok, "안뇽", "안녕")
        prompt = encode_prompt(self.tok, "안뇽")
        self.assertEqual(ex.input_ids[:ex.source_len], prompt)
        self.assertEqual(ex.labels[:ex.source_len], [-100] * ex.source_len)
        self.assertEqual(ex.labels[ex.source_len:], [b + 4 for b in "안녕".encode()] + [2])
        self.assertEqual(ex.input_ids.count(1), 1)
        self.assertEqual(ex.input_ids.count(2), 1)
        self.assertNotIn(2, prompt)
        self.assertEqual(ex.labels[1:][ex.source_len - 1], ex.input_ids[ex.source_len])

    def test_empty_target_still_learns_eos(self):
        ex = encode_pair(self.tok, "그대로", "")
        self.assertEqual([x for x in ex.labels if x != -100], [2])

    def test_exact_limit_and_overflow(self):
        self.assertEqual(len(encode_pair(self.tok, "a" * 2031, "").input_ids), 2048)
        with self.assertRaisesRegex(ValueError, "row-9.*2049"):
            encode_pair(self.tok, "a" * 2032, "", sample_id="row-9")

    def test_no_normalization(self):
        src = "  안녕\u00a0세상  "
        ids = encode_prompt(self.tok, src)
        self.assertEqual(bytes(x - 4 for x in ids[1:]).decode(), src + SEPARATOR)

    def test_reject_non_byte_tokenizer(self):
        class Bad(ByteTokenizer):
            def encode(self, text, **kwargs):
                return [42]
        with self.assertRaisesRegex(ValueError, "byte"):
            encode_pair(Bad(), "hello", "world")

    def test_generation_budget(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            encode_prompt(self.tok, "a" * 1400, max_new_bytes=768)

    def test_padding_not_silently_introduced(self):
        a, b = encode_pair(self.tok, "a", "a"), encode_pair(self.tok, "long", "b")
        with self.assertRaisesRegex(ValueError, "same length"):
            prepare_unpadded_batch([a, b])
        batch = prepare_unpadded_batch([a, a])
        self.assertEqual(batch["attention_mask"], [[1] * len(a.input_ids)] * 2)


class DatasetTests(unittest.TestCase):
    def test_blank_only_lines_and_whitespace_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_bytes("\n source \t target \r\n\n".encode())
            rows = read_tsv(p)
            self.assertEqual([(r.source, r.target, r.line_number) for r in rows], [(" source ", " target ", 2)])

    def test_bad_tsv_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_text("ok\tok\nbad\textra\tcolumn\n")
            with self.assertRaisesRegex(ValueError, ":2:"):
                read_tsv(p)

    def test_canonical_nine_splits_only(self):
        paths = canonical_split_paths(Path("data/Preprocessed"))
        self.assertEqual(len(paths), 9)
        self.assertEqual(paths["korean_learner/test"].name, "korean_learner_test.txt")
        self.assertEqual(dataset_split_path(Path('data/Preprocessed'),'lang8','test'),
                         Path('data/Preprocessed/lang8/lang8_test.txt'))

    @unittest.skipUnless(any(p.is_file() for p in canonical_split_paths(Path('data/Preprocessed')).values()),
                         'private dataset is not distributed with the repository')
    def test_canonical_counts_and_real_adapter_lengths(self):
        total = 0
        maximum = 0
        over1024 = 0
        for p in canonical_split_paths(Path("data/Preprocessed")).values():
            dataset = GecDataset(p, ByteTokenizer())
            total += len(dataset)
            for ex in dataset:
                n = len(ex.input_ids)
                maximum = max(maximum, n)
                over1024 += n > 1024
        self.assertEqual((total, maximum, over1024), (201534, 1380, 9))


if __name__ == "__main__":
    unittest.main()
