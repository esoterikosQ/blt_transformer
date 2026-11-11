import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from blt_hf import eval as evaluation
from blt_hf import score_local
from blt_hf.manifest import FINGERPRINT_KEYS, sha256_json, split_identity
from blt_hf.m2_resumable import _validate_or_create_manifest
from blt_hf.metrics import compute_m2_with_checkpoints, scorer_identity
from blt_hf.runtime import ROOT


class LocalM2ResumeTests(unittest.TestCase):
    def test_snapshot_never_publishes_partial_completion_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local, remote = root / 'local', root / 'remote'
            local.mkdir()
            (local / 'completed.jsonl').write_bytes(b'{"sentence_id": 0}\n{"sentence_id":')
            score_local.snapshot_journal(local, remote)
            self.assertEqual((remote / 'completed.jsonl').read_bytes(), b'{"sentence_id": 0}\n')

    def test_existing_score_finishes_from_node_local_journal(self):
        fixture = Path(__file__).parent / 'fixtures/m2'
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            base = Path(folder)
            tsv = base / 'korean_learner_test.txt'
            gold = tsv.with_suffix('.m2')
            gold.write_bytes((fixture / 'source_gold').read_bytes())
            tsv.write_text('The cat sat at mat .\tA cat sat on the mat .\n'
                           'The dog .\tThe dog .\n'
                           'Giant otters is an apex predator .\tGiant otters are apex predator .\n',
                           encoding='utf-8')
            root = base / 'eval'
            scored = root / 'scored'
            journal = scored / 'm2'
            journal.mkdir(parents=True)
            (scored / 'source.txt').write_text('The cat sat at mat .\nThe dog .\n'
                                               'Giant otters is an apex predator .\n', encoding='utf-8')
            (scored / 'reference.txt').write_text('A cat sat on the mat .\nThe dog .\n'
                                                  'Giant otters are apex predator .\n', encoding='utf-8')
            (scored / 'hypothesis.txt').write_bytes((fixture / 'system').read_bytes())
            manifest = {key: 'fixture' for key in FINGERPRINT_KEYS}
            manifest.update(split_identity(tsv, gold, dataset='korean_learner', split='test'))
            manifest.update({'scorer_hash': scorer_identity(), 'training_checks': 'passed',
                             'conversion_checks': {}, 'shard_count': 1})
            manifest['fingerprint'] = sha256_json({
                'required': evaluation.fingerprint(manifest), 'complete': manifest})
            (root / 'run.json').write_text(json.dumps(manifest), encoding='utf-8')
            (scored / 'gleu.json').write_text(json.dumps({
                'fingerprint': manifest['fingerprint'], 'samples': 3, 'gleu': 0.0}), encoding='utf-8')
            _validate_or_create_manifest(journal / 'run_config.json', scored / 'hypothesis.txt',
                                         gold, 3, {'beta': 0.5, 'max_unchanged_words': 2,
                                                   'ignore_whitespace_casing': False})
            compute_m2_with_checkpoints(scored / 'hypothesis.txt', gold, journal,
                                        workers=2, max_passes=1)
            first = (journal / 'completed.jsonl').read_text().splitlines()[0]
            (journal / 'completed.jsonl').write_text(first + '\n', encoding='utf-8')
            args = SimpleNamespace(output_dir=str(root), dataset='korean_learner', split='test',
                                   m2_workers=2, m2_timeout=30., m2_passes=1,
                                   max_seconds=30, snapshot_seconds=1)
            with patch.object(score_local, 'ROOT', base), \
                    patch.object(score_local, 'dataset_split_path', return_value=tsv), \
                    patch.object(evaluation, 'ROOT', base), \
                    patch.object(evaluation, 'dataset_split_path', return_value=tsv):
                self.assertEqual(score_local.run(args), 0)
            self.assertEqual(json.loads((scored / 'metrics.json').read_text())['m2']['completed'], 3)
            self.assertEqual(len((journal / 'completed.jsonl').read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main()
