import tempfile
import unittest
from pathlib import Path
from blt_hf.evaluation import shard_bounds, collect_records, publish_lines
from blt_hf.manifest import write_json, sha256_file

class EvaluationContracts(unittest.TestCase):
    def test_exact_full_coverage_and_no_overlap(self):
        parts=[shard_bounds(17, 4, i) for i in range(4)]
        self.assertEqual([j for a,b in parts for j in range(a,b)], list(range(17)))
        with self.assertRaises(ValueError): shard_bounds(2, 3, 0)

    def test_missing_duplicate_or_mixed_shard_cannot_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            m={'fingerprint':'abc','sample_count':2,'shard_count':2}
            for i in range(2):
                d=root/'shards'/f'{i:04d}'
                write_json(d/'batches'/'00000000.json',{'fingerprint':'abc','records':[{'index':i,'text':str(i)}]})
                write_json(d/'complete.json',dict(m,shard_id=i,start=i,end=i+1,batch_hashes={'00000000.json':sha256_file(d/'batches/00000000.json')}))
            self.assertEqual([r['index'] for r in collect_records(root,m)], [0,1])
            (root/'shards/0001/complete.json').unlink()
            with self.assertRaises(ValueError): collect_records(root,m)

    def test_output_text_never_overwrites_other_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hyp.txt'
            publish_lines(p,['a',''])
            publish_lines(p,['a',''])
            with self.assertRaises(ValueError): publish_lines(p,['b',''])
