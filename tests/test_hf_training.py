import unittest
from blt_hf.training import epoch_batches, rank_work, lr_factor, normalize_gradient_scale

class TrainingContracts(unittest.TestCase):
    def test_no_dropped_or_duplicate_examples_in_ddp_tail(self):
        batches = epoch_batches(19, 8, seed=7, epoch=2)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(19)))
        for batch in batches:
            seen = []
            work = [rank_work(batch, rank, 4) for rank in range(4)]
            self.assertEqual(len({len(v) for v in work}), 1)
            for values in work:
                seen.extend(i for i in values if i is not None)
            self.assertEqual(sorted(seen), sorted(batch))
        self.assertEqual(batches, epoch_batches(19, 8, seed=7, epoch=2))
        self.assertNotEqual(batches, epoch_batches(19, 8, seed=7, epoch=3))

    def test_token_loss_normalization_accounts_for_ddp_average(self):
        # Global summed loss / actual supervised tokens, including uneven final steps.
        tokens, world = 17, 4
        self.assertEqual(sum(x * normalize_gradient_scale(world, tokens) for x in [3, 4, 0, 10]) / world, 1)

    def test_lr_schedule_can_resume_at_exact_step(self):
        self.assertEqual(lr_factor(0, 100, 10), .1)
        self.assertEqual(lr_factor(9, 100, 10), 1)
        self.assertEqual(lr_factor(100, 100, 10), 0)
        self.assertGreater(lr_factor(30, 100, 10), lr_factor(31, 100, 10))
