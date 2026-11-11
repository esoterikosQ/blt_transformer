import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from blt_hf.metrics import compute_gleu
from blt_hf.m2_resumable import evaluate_m2_resumable, _load_completed, load_m2_annotations

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/m2'

class MetricContracts(unittest.TestCase):
    def test_new_s_line_starts_record_without_blank_separator(self):
        text=("S first row\n"
              "A 0 1|||R|||fixed|||REQUIRED|||-NONE-|||0\n"
              "S second row\n"
              "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'joined.m2';path.write_text(text,encoding='utf-8')
            sources,edits=load_m2_annotations(path)
            self.assertEqual(sources,['first row','second row'])
            self.assertEqual(len(edits),2)
            self.assertEqual(edits[0][0][0][:3],(0,1,'first'))
            self.assertEqual(edits[1][0],[])

    @unittest.skipUnless((FIXTURE/'source_gold').is_file() and (ROOT/'blt_hf/vendor/m2/levenshtein.py').is_file(),
                         'local M2 source/fixtures are excluded from publication; see THIRD_PARTY_NOTICES.md')
    def test_m2_matches_original_cli_and_resume_does_not_duplicate(self):
        expected=subprocess.check_output([sys.executable,'-m','blt_hf.vendor.m2.m2scorer',str(FIXTURE/'system'),str(FIXTURE/'source_gold')],cwd=ROOT,text=True)
        self.assertIn('F_0.5       : 0.8000',expected)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'m2'
            result=evaluate_m2_resumable(FIXTURE/'system',FIXTURE/'source_gold',out,workers=2,timeout_seconds=5,max_passes=1)
            self.assertEqual(result.status,'complete');self.assertAlmostEqual(result.f_score,.8)
            before=(out/'completed.jsonl').read_bytes()
            again=evaluate_m2_resumable(FIXTURE/'system',FIXTURE/'source_gold',out,workers=2,timeout_seconds=5,max_passes=1)
            self.assertEqual(again.f_score,result.f_score)
            self.assertEqual(before,(out/'completed.jsonl').read_bytes())
            with self.assertRaises(RuntimeError):
                evaluate_m2_resumable(FIXTURE/'system',FIXTURE/'source_gold',out,beta=1)

    def test_torn_tail_is_preserved_and_not_appended_to(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'completed.jsonl'
            p.write_bytes(b'{"sentence_id":0}\n{"sentence_id":')
            self.assertEqual(list(_load_completed(p)),[0])
            self.assertTrue(p.read_bytes().endswith(b'\n'))
            self.assertEqual(len(list(Path(d).glob('*.torn-*'))),1)
            p.write_text('{broken}\n')
            with self.assertRaises(json.JSONDecodeError): _load_completed(p)

    @unittest.skipUnless((FIXTURE/'source_gold').is_file() and (ROOT/'blt_hf/vendor/m2/levenshtein.py').is_file(),
                         'local M2 source/fixtures are excluded from publication; see THIRD_PARTY_NOTICES.md')
    def test_incomplete_m2_never_exposes_subset_score(self):
        def timeout(ids,*args):
            for i in ids: yield {'status':'timeout','sentence_id':i,'elapsed_seconds':.01}
        with tempfile.TemporaryDirectory() as d, patch('blt_hf.m2_resumable._run_sentence_processes',timeout):
            result=evaluate_m2_resumable(FIXTURE/'system',FIXTURE/'source_gold',d,max_passes=1)
            self.assertEqual(result.status,'partial');self.assertIsNone(result.f_score)

    def test_gleu_corpus_not_sentence_average_and_no_row_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            source=['I has a nice red apple today .','This is a short sentence .']
            ref=['I have a nice red apple today .','This is a short sentence .']
            for name,rows in [('s',source),('r',ref),('h',ref)]: (d/name).write_text('\n'.join(rows)+'\n')
            self.assertEqual(compute_gleu(d/'r',d/'s',d/'h'),100)
            (d/'h').write_text('\n\n')
            self.assertEqual(compute_gleu(d/'r',d/'s',d/'h'),0)
            (d/'h').write_text('one line\n')
            with self.assertRaises(ValueError):compute_gleu(d/'r',d/'s',d/'h')

    @unittest.skipUnless((ROOT/'tests/fixtures/gleu_legacy.json').is_file(),
                         'corpus-derived fixture is not redistributed')
    def test_previous_gleu_wrapper_fixture_exact_scores(self):
        fixture=json.loads((ROOT/'tests/fixtures/gleu_legacy.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            for case in fixture['cases']:
                for name in ('sources','references','hypotheses'):
                    (d/name).write_text('\n'.join(case[name])+'\n',encoding='utf-8')
                self.assertEqual(compute_gleu(d/'references',d/'sources',d/'hypotheses'),case['expected_gleu'])
