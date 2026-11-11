import json
import unittest
from blt_hf_checks.check_load_generate import json_safe

class LoadingReportTests(unittest.TestCase):
    def test_hf_sets_remain_machine_readable(self):
        data = json_safe({"missing_keys": set(), "unexpected_keys": {"b", "a"}, "nested": ({"x"},)})
        self.assertEqual(json.loads(json.dumps(data)),
                         {"missing_keys": [], "unexpected_keys": ["a", "b"], "nested": [["x"]]})
