import tempfile
import unittest
from pathlib import Path

from blt_hf.manifest import (
    FINGERPRINT_KEYS, ManifestMismatch, assert_compatible, fingerprint,
    split_identity, verify_file_hashes, sha256_file, write_json,
)


class ManifestTests(unittest.TestCase):
    def manifest(self):
        result = {key: "fixed" for key in FINGERPRINT_KEYS}
        result.update(attention_mode="osc", use_cache=False, num_beams=4,
                      batch_size=8, length_penalty=1.0, max_sequence_bytes=2048,
                      max_new_bytes=768)
        return result

    def test_same_execution_different_shard_ranges(self):
        a = dict(self.manifest(), shard_start=0)
        b = dict(self.manifest(), shard_start=100)
        self.assertEqual(fingerprint(a), fingerprint(b))
        assert_compatible([a, b])

    def test_mode_and_data_changes_are_rejected(self):
        for key in ("attention_mode", "source_hash", "target_hash", "m2_hash", "checkpoint_hash"):
            with self.subTest(key=key), self.assertRaisesRegex(ManifestMismatch, key):
                assert_compatible([self.manifest(), dict(self.manifest(), **{key: "different"})])

    def test_missing_and_null_fields_fail_even_when_both_missing(self):
        for missing in (True, False):
            m = self.manifest()
            if missing:
                del m["attention_mode"]
            else:
                m["attention_mode"] = None
            with self.assertRaisesRegex(ManifestMismatch, "attention_mode"):
                assert_compatible([m, m])

    def test_boolean_and_number_are_not_interchangeable(self):
        with self.assertRaises(ManifestMismatch):
            assert_compatible([self.manifest(), dict(self.manifest(), use_cache=0)])

    def test_no_shards_is_error(self):
        with self.assertRaises(ManifestMismatch):
            assert_compatible([])

    def test_split_identity_preserves_order_and_m2_checks_source(self):
        with tempfile.TemporaryDirectory() as d:
            tsv, m2 = Path(d) / "x.txt", Path(d) / "x.m2"
            tsv.write_text("a\tb\nc\td\n")
            m2.write_text("S a\n\nS c\n")
            a = split_identity(tsv, m2, dataset="native", split="test")
            tsv.write_text("c\td\na\tb\n")
            with self.assertRaisesRegex(ValueError, "source"):
                split_identity(tsv, m2, dataset="native", split="test")
            m2.write_text("S c\n\nS a\n")
            b = split_identity(tsv, m2, dataset="native", split="test")
            self.assertNotEqual(a["source_hash"], b["source_hash"])

    def test_file_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "weights"
            p.write_bytes(b"first")
            expected = {"weights": sha256_file(p)}
            verify_file_hashes(Path(d), expected)
            p.write_bytes(b"other")
            with self.assertRaisesRegex(ManifestMismatch, "weights"):
                verify_file_hashes(Path(d), expected)

    def test_reject_hash_paths_outside_artifact_root(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            verify_file_hashes(Path(d), {"../escape": "0" * 64})

    def test_report_does_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"
            write_json(p, {"state": "first"})
            with self.assertRaises(FileExistsError):
                write_json(p, {"state": "second"})
            self.assertIn("first", p.read_text())


if __name__ == "__main__":
    unittest.main()
