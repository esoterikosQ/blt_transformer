import json
import tempfile
import unittest
from pathlib import Path

from blt_hf.integrated_validation import (
    gleu_early_stopping_update, gleu_improved, order_predictions,
    score_validation_epoch, validation_groups,
)
from blt_hf.data_adapter import GecEncoding
from blt_hf.generation import GLOBAL_PREFIX_GUARDED_BACKEND, HF_BACKEND
from blt_hf.train import parser as train_parser, validation_generation_config


class IntegratedValidationContracts(unittest.TestCase):
    def test_guarded_backend_is_explicit_and_requires_gleu_beam1_batch1(self):
        args = train_parser().parse_args(['--run-dir', 'outputs/blt_hf/native/new-run'])
        self.assertEqual(validation_generation_config(args).backend, HF_BACKEND)
        guarded = ['--run-dir', 'outputs/blt_hf/native/new-run',
                   '--selection-metric', 'val_gleu',
                   '--validation-generation-backend', GLOBAL_PREFIX_GUARDED_BACKEND]
        self.assertEqual(validation_generation_config(train_parser().parse_args(guarded)).backend,
                         GLOBAL_PREFIX_GUARDED_BACKEND)
        with self.assertRaises(ValueError):
            validation_generation_config(train_parser().parse_args(guarded + ['--validation-beams', '4']))
        with self.assertRaises(ValueError):
            validation_generation_config(train_parser().parse_args(guarded + ['--validation-batch-size', '4']))
        with self.assertRaises(ValueError):
            validation_generation_config(train_parser().parse_args(
                ['--run-dir', 'outputs/blt_hf/native/new-run',
                 '--validation-generation-backend', GLOBAL_PREFIX_GUARDED_BACKEND]))

    def test_equal_length_validation_batches_cover_each_rank_once(self):
        lengths = [4, 5, 4, 6, 4, 5, 4, 6, 4, 5, 4, 6]
        examples = [GecEncoding([1] * length + [2], [-100] * length + [2], length)
                    for length in lengths]
        groups = [validation_groups(examples, rank=rank, world_size=4, batch_size=2)
                  for rank in range(4)]
        self.assertEqual(sorted(index for rank_groups in groups for group in rank_groups
                                for index in group), list(range(len(examples))))
        self.assertTrue(all(len(group) <= 2 and len({lengths[i] for i in group}) == 1
                            for rank_groups in groups for group in rank_groups))
        self.assertEqual(groups[0], [[0, 4], [8]])

    def test_gleu_not_teacher_forced_loss_selects_best_epoch(self):
        self.assertTrue(gleu_improved(40., None))
        self.assertFalse(gleu_improved(39., 40.))
        self.assertFalse(gleu_improved(40., 40.))
        self.assertTrue(gleu_improved(41., 40.))

    def test_early_stopping_counts_complete_ties_and_survives_resume(self):
        args = train_parser().parse_args([
            '--run-dir', 'outputs/blt_hf/native/new-run', '--epochs', '10',
            '--selection-metric', 'val_gleu', '--early-stopping-patience', '3'])
        self.assertEqual((args.epochs, args.early_stopping_patience), (10, 3))
        best, stale = None, 0
        for score, expected_stale, expected_stop in [
            (52., 0, False), (52., 1, False), (51., 2, False),
            (53., 0, False), (53., 1, False), (52., 2, False), (52., 3, True),
        ]:
            improved, stale, stop = gleu_early_stopping_update(
                score, best, stale, args.early_stopping_patience)
            if improved:
                best = score
            self.assertEqual((stale, stop), (expected_stale, expected_stop))
            # The count and best score are the values restored from an epoch checkpoint.
            stale, best = json.loads(json.dumps([stale, best]))
        self.assertEqual(best, 53.)
        self.assertEqual(gleu_early_stopping_update(52., 53., 2, 0), (False, 3, False))

    def test_four_rank_predictions_score_full_corpus_in_original_order(self):
        sources = ['I has a nice red apple today .', 'This is a short sentence .',
                   'She walk to school each day .', 'Another sentence is correct .']
        references = ['I have a nice red apple today .', 'This is a short sentence .',
                      'She walks to school each day .', 'Another sentence is correct .']
        parts = [[{'index': i, 'text': references[i]}] for i in range(4)]
        self.assertEqual(order_predictions(parts, 4), references)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = score_validation_epoch(root, epoch=1, global_step=2,
                                            sources=sources, references=references,
                                            predictions=order_predictions(parts, 4),
                                            num_beams=1, max_new_bytes=768,
                                            validation_file_hash='fixture')
            self.assertEqual(result['gleu'], 100.)
            self.assertEqual(result['generation_backend'], HF_BACKEND)
            self.assertEqual(json.loads((root/'validation/epoch-0001/metrics.json').read_text())['gleu'], 100.)
            self.assertEqual((root/'validation/epoch-0001/hypothesis.txt').read_text().splitlines(), references)
            self.assertEqual(score_validation_epoch(root, epoch=1, global_step=2,
                                                   sources=sources, references=references,
                                                   predictions=references, num_beams=1,
                                                   max_new_bytes=768,
                                                   validation_file_hash='fixture'), result)

    def test_epoch_report_records_guarded_backend_and_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = score_validation_epoch(
                directory, epoch=1, global_step=2, sources=['I has apples .'],
                references=['I have apples .'], predictions=['I have apples .'],
                num_beams=1, max_new_bytes=768, validation_file_hash='fixture',
                generation_backend=GLOBAL_PREFIX_GUARDED_BACKEND,
                cache_preliminary_skips=12, cache_guard_refreshes=2)
            self.assertEqual(result['generation_backend'], GLOBAL_PREFIX_GUARDED_BACKEND)
            self.assertEqual(result['cache_preliminary_skips'], 12)
            self.assertEqual(result['cache_guard_refreshes'], 2)
            with self.assertRaisesRegex(ValueError, 'Invalid cache refresh counts'):
                score_validation_epoch(directory, epoch=2, global_step=3,
                                       sources=['I has apples .'], references=['I have apples .'],
                                       predictions=['I have apples .'], num_beams=1,
                                       max_new_bytes=768, validation_file_hash='fixture',
                                       cache_preliminary_skips=1, cache_guard_refreshes=2)

    def test_missing_duplicate_and_changed_predictions_are_rejected(self):
        with self.assertRaises(ValueError): order_predictions([[{'index': 0, 'text': 'a'}]], 2)
        with self.assertRaises(ValueError): order_predictions([[{'index': 0, 'text': 'a'}],
                                                               [{'index': 0, 'text': 'b'}]], 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(epoch=1, global_step=2, sources=['I has apples .'],
                        references=['I have apples .'], num_beams=1,
                        max_new_bytes=768, validation_file_hash='fixture')
            score_validation_epoch(root, predictions=['I have apples .'], **args)
            with self.assertRaises(ValueError):
                score_validation_epoch(root, predictions=['I has apples .'], **args)
