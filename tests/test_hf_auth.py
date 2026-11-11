import tempfile
import unittest
from pathlib import Path

from blt_hf_checks.auth import read_project_token


class AuthTests(unittest.TestCase):
    def test_plain_env_and_json_token_formats(self):
        token = "hf_abcdefghijklmnopqrstuvwxyz"
        for content in (token, f"HF_TOKEN={token}\n", '{"token":"' + token + '"}'):
            with self.subTest(content_type=content[:1]), tempfile.TemporaryDirectory() as d:
                path = Path(d) / ".hf_access"
                path.write_text(content)
                self.assertEqual(read_project_token(path, Path(d)), token)

    def test_login_password_is_not_a_token(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".hf_access"
            path.write_text("username=somebody\npassword=never-print-this\n")
            with self.assertRaises(ValueError) as err:
                read_project_token(path, Path(d))
            self.assertNotIn("never-print-this", str(err.exception))

    def test_ambiguous_tokens_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".hf_access"
            path.write_text("hf_abcdefghijklmnop hf_zyxwvutsrqponmlk")
            with self.assertRaisesRegex(ValueError, "one"):
                read_project_token(path, Path(d))

    def test_external_file_is_not_read(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "project"):
                read_project_token(Path(d).parent / "outside", Path(d))
