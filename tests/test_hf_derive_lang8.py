"""Lang-8 extraction must preserve the supplied union and component splits."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blt_hf import derive_lang8
from blt_hf.data_adapter import dataset_split_path


class DeriveLang8Tests(unittest.TestCase):
    def test_split_files_sidecars_and_aggregate_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data' / 'Preprocessed'
            supplied = []
            for split in ('train', 'val', 'test'):
                rows = [('L8' + split, 'fixed' + split), ('learner' + split, 'learner-ok' + split),
                        ('native' + split, 'native-ok' + split)]
                for dataset, subset in (('union', rows), ('korean_learner', rows[1:2]),
                                        ('native', rows[2:])):
                    folder = root / dataset
                    folder.mkdir(parents=True, exist_ok=True)
                    tsv = folder / f'{dataset}_{split}.txt'
                    m2 = folder / f'{dataset}_{split}.m2'
                    tsv.write_text(''.join(f'{source}\t{target}\n' for source, target in subset), encoding='utf-8')
                    m2.write_text(''.join(f'S {source}\n\n' for source, _ in subset), encoding='utf-8')
                    supplied.extend([tsv, m2])
                if split != 'train':
                    hanspell = root / 'union' / f'hanspell_{split}.txt'
                    hanspell.write_text(''.join(f'{source}-spell\n' for source, _ in rows), encoding='utf-8')
                    supplied.append(hanspell)
            before = {path: path.read_bytes() for path in supplied}
            with patch.dict(derive_lang8.LANG8_COUNTS, {'train': 1, 'val': 1, 'test': 1}):
                derive_lang8.derive(root)
                derive_lang8.derive(root)  # Repeat must verify rather than overwrite.

            folder = root / 'lang8'
            for split in ('train', 'val', 'test'):
                prefix = 'L8' + split
                self.assertEqual(dataset_split_path(root, 'lang8', split), folder / f'lang8_{split}.txt')
                self.assertEqual((folder / f'lang8_{split}.txt').read_text(),
                                 f'{prefix}\tfixed{split}\n')
                self.assertEqual((folder / f'lang8_{split}.m2').read_text(), f'S {prefix}\n')
                self.assertEqual((folder / f'lang8_{split}_original.txt').read_text(), f'{prefix}\n')
                self.assertEqual((folder / f'lang8_{split}_corrected.txt').read_text(),
                                 f'fixed{split}\n')
                if split != 'train':
                    self.assertEqual((folder / f'hanspell_{split}.txt').read_text(), f'{prefix}-spell\n')
            self.assertEqual((folder / 'lang8.txt').read_text(),
                             'L8train\tfixedtrain\nL8test\tfixedtest\nL8val\tfixedval\n')
            self.assertEqual((folder / 'lang8.m2').read_text(),
                             'S L8train\n\nS L8test\n\nS L8val\n')
            self.assertEqual((folder / 'lang8_original.txt').read_text(),
                             'L8train\nL8test\nL8val\n')
            self.assertEqual((folder / 'lang8_corrected.txt').read_text(),
                             'fixedtrain\nfixedtest\nfixedval\n')
            self.assertEqual(before, {path: path.read_bytes() for path in supplied})


if __name__ == '__main__':
    unittest.main()
