import unittest

from blt_hf_checks.check_env import validate_gpu_report


class EnvironmentTests(unittest.TestCase):
    def report(self):
        return {"python": "3.11.9", "torch": "2.11.0+cu128", "transformers": "5.16.1",
                "cuda_runtime": "12.8", "cuda_available": True, "blt_import": True,
                "cuda_smoke": True, "xformers": None,
                "devices": [{"name": "NVIDIA RTX 5090", "capability": [12, 0],
                             "bf16_supported": True, "bf16_smoke": True}],
                "errors": []}

    def test_supported_gpu_environment(self):
        self.assertEqual(validate_gpu_report(self.report()), [])

    def test_h200_environment_is_supported(self):
        report = dict(self.report(), devices=[{"name": "NVIDIA H200", "capability": [9, 0],
                                               "bf16_supported": True, "bf16_smoke": True}])
        self.assertEqual(validate_gpu_report(report), [])

    def test_supported_capability_without_working_bf16_fails(self):
        report = dict(self.report(), devices=[{"name": "NVIDIA H200", "capability": [9, 0],
                                               "bf16_supported": True, "bf16_smoke": False}])
        self.assertTrue(any("BF16 CUDA matmul" in error for error in validate_gpu_report(report)))

    def test_wrong_versions_fail(self):
        for field, value in (("torch", "2.14.0"), ("transformers", "5.15.0"), ("cuda_runtime", "13.0")):
            with self.subTest(field=field):
                errors = validate_gpu_report(dict(self.report(), **{field: value}))
                self.assertTrue(any(field in error for error in errors))

    def test_import_success_does_not_replace_cuda_execution(self):
        self.assertTrue(validate_gpu_report(dict(self.report(), cuda_smoke=False)))

    def test_v100_cannot_silently_enter_bf16_path(self):
        report = dict(self.report(), devices=[{"name": "Tesla V100", "capability": [7, 0],
                                               "bf16_supported": False, "bf16_smoke": False}])
        self.assertTrue(validate_gpu_report(report))

    def test_xformers_free_environment(self):
        self.assertTrue(validate_gpu_report(dict(self.report(), xformers="0.0.33")))


if __name__ == "__main__":
    unittest.main()
