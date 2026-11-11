import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from blt_hf import eval as evaluation
from blt_hf.manifest import sha256_file, sha256_json


class ScoreResumeTests(unittest.TestCase):
    def test_cached_inputs_skip_expensive_batch_scan_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tsv = root / 'korean_learner_test.txt'
            m2 = tsv.with_suffix('.m2')
            tsv.write_text('wrong\tcorrect\n', encoding='utf-8')
            m2.write_text('S wrong\nA 0 1|||R|||correct|||REQUIRED|||-NONE-|||0\n', encoding='utf-8')
            output = root / 'eval'
            scored = output / 'scored'
            (scored / 'm2').mkdir(parents=True)
            for name, value in [('source.txt', 'wrong\n'), ('reference.txt', 'correct\n'),
                                ('hypothesis.txt', 'correct\n')]:
                (scored / name).write_text(value, encoding='utf-8')
            manifest = {'dataset': 'korean_learner', 'split': 'test', 'sample_count': 1,
                        'scorer_hash': 'stub',
                        'training_checks': 'passed', 'conversion_checks': {}}
            manifest['fingerprint'] = sha256_json({'required': 'stub', 'complete': manifest})
            (output / 'run.json').write_text(json.dumps(manifest), encoding='utf-8')
            (scored / 'gleu.json').write_text(json.dumps({'fingerprint': manifest['fingerprint'],
                                                         'samples': 1, 'gleu': 100.0}), encoding='utf-8')
            (scored / 'm2/run_config.json').write_text(json.dumps({
                'hypothesis_path': str((scored / 'hypothesis.txt').resolve()),
                'hypothesis_sha256': sha256_file(scored / 'hypothesis.txt'),
                'source_gold_path': str(m2.resolve()), 'source_gold_sha256': sha256_file(m2),
                'examples': 1}), encoding='utf-8')
            args = SimpleNamespace(dataset='korean_learner', split='test', output_dir=output,
                                   max_seconds=10, m2_workers=1, m2_timeout=30, m2_passes=1)
            partial = SimpleNamespace(status='partial', to_dict=lambda: {'completed': 0, 'total': 1})
            with patch.object(evaluation, 'ROOT', root), \
                    patch.object(evaluation, 'local_path', side_effect=lambda path: Path(path)), \
                    patch.object(evaluation, 'fingerprint', return_value='stub'), \
                    patch.object(evaluation, 'scorer_identity', return_value='stub'), \
                    patch.object(evaluation, 'dataset_split_path', return_value=tsv), \
                    patch.object(evaluation, 'split_identity', return_value={}), \
                    patch.object(evaluation, 'collect_records', side_effect=AssertionError('batch rescan')), \
                    patch.object(evaluation, 'compute_gleu', side_effect=AssertionError('GLEU recompute')), \
                    patch.object(evaluation, 'compute_m2_with_checkpoints', return_value=partial):
                self.assertEqual(evaluation.aggregate(args), 75)
                (scored / 'hypothesis.txt').write_text('tampered\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Cached M2 input identity mismatch'):
                    evaluation.aggregate(args)
                (scored / 'hypothesis.txt').write_text('correct\n', encoding='utf-8')
                complete = SimpleNamespace(status='complete', to_dict=lambda: {
                    'status': 'complete', 'completed': 1, 'total': 1, 'unresolved': 0,
                    'precision': 1.0, 'recall': 1.0, 'f0.5': 1.0})
                with patch.object(evaluation, 'compute_m2_with_checkpoints', return_value=complete):
                    self.assertEqual(evaluation.aggregate(args), 0)
                self.assertEqual(json.loads((scored / 'progress.json').read_text())['status'], 'complete')
                (scored / 'progress.json').write_text(json.dumps({'status': 'partial'}), encoding='utf-8')
                self.assertEqual(evaluation.aggregate(args), 0)
                self.assertEqual(json.loads((scored / 'progress.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
