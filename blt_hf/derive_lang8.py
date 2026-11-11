"""Derive data/Preprocessed/lang8 from union without changing supplied files."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from .manifest import sha256_file, sha256_json

LANG8_COUNTS = {'train': 76692, 'val': 16434, 'test': 16434}


def _rows(path):
    with Path(path).open(encoding='utf-8', newline=None) as stream:
        return [line.rstrip('\r\n') for line in stream if line.strip()]


def _m2_records(path):
    records, current = [], []
    with Path(path).open(encoding='utf-8') as stream:
        for raw in stream:
            line = raw.rstrip('\r\n')
            if not line:
                if current: records.append(current); current = []
                continue
            if line.startswith('S ') and current:
                records.append(current); current = []
            current.append(line)
    if current: records.append(current)
    if any(sum(line.startswith('S ') for line in record) != 1 for record in records):
        raise ValueError(f'Invalid logical M2 record in {path}')
    return records


def _publish(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError(f'Existing derived artifact differs: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _pair_columns(rows, path):
    pairs = [row.split('\t') for row in rows]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f'Expected two TSV columns in {path}')
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _lines(rows):
    return '\n'.join(rows) + '\n'


def _m2_text(records):
    return '\n\n'.join('\n'.join(record) for record in records) + '\n'


def derive(data_root, output_root=None):
    data_root = Path(data_root)
    output_root = Path(output_root) if output_root else data_root/'lang8'
    report = {'schema_version':2, 'derivation':'union-prefix-before-korean_learner-and-native',
              'aggregate_order':['train','test','val'], 'splits':{}}
    extracted = {}
    for split, expected in LANG8_COUNTS.items():
        union_path=data_root/'union'/f'union_{split}.txt'
        learner_path=data_root/'korean_learner'/f'korean_learner_{split}.txt'
        native_path=data_root/'native'/f'native_{split}.txt'
        union, learner, native = _rows(union_path), _rows(learner_path), _rows(native_path)
        if len(union)-len(learner)-len(native) != expected:
            raise ValueError(f'Unexpected Lang-8 count for {split}')
        if union[expected:] != learner + native:
            raise ValueError(f'Union suffix is not learner+native for {split}')
        sources, targets = _pair_columns(union[:expected], union_path)
        output=output_root/f'lang8_{split}.txt'

        union_m2, learner_m2, native_m2 = (_m2_records(path.with_suffix('.m2'))
                                            for path in (union_path, learner_path, native_path))
        union_sources=[next(line[2:].strip() for line in record if line.startswith('S ')) for record in union_m2]
        suffix_sources=[next(line[2:].strip() for line in record if line.startswith('S '))
                        for record in learner_m2+native_m2]
        if (len(union_m2) != len(union) or union_sources[expected:] != suffix_sources
                or union_m2[expected:] != learner_m2 + native_m2):
            raise ValueError(f'Union M2 suffix is not learner+native for {split}')
        all_sources, _ = _pair_columns(union, union_path)
        if union_sources != all_sources:
            raise ValueError(f'Union M2 sources differ from TSV for {split}')
        if len(learner_m2) != len(learner) or len(native_m2) != len(native):
            raise ValueError(f'Component M2 count differs from TSV for {split}')

        _publish(output, _lines(union[:expected]))
        output_m2=output.with_suffix('.m2')
        _publish(output_m2, _m2_text(union_m2[:expected]))
        _publish(output_root/f'lang8_{split}_original.txt', _lines(sources))
        _publish(output_root/f'lang8_{split}_corrected.txt', _lines(targets))
        if split in ('val', 'test'):
            union_hanspell = _rows(data_root/'union'/f'hanspell_{split}.txt')
            if len(union_hanspell) != len(union):
                raise ValueError(f'Union hanspell count differs from TSV for {split}')
            _publish(output_root/f'hanspell_{split}.txt', _lines(union_hanspell[:expected]))

        extracted[split] = {'pairs':union[:expected], 'm2':union_m2[:expected],
                            'sources':sources, 'targets':targets}
        report['splits'][split]={'rows':expected,'tsv_sha256':sha256_file(output),
                                 'm2_sha256':sha256_file(output_m2),
                                 'source_hash':sha256_json(union_sources[:expected])}
    order = ('train', 'test', 'val')  # Mirrors the supplied union.txt/union.m2 layout.
    _publish(output_root/'lang8.txt', _lines(sum((extracted[s]['pairs'] for s in order), [])))
    _publish(output_root/'lang8.m2', _m2_text(sum((extracted[s]['m2'] for s in order), [])))
    _publish(output_root/'lang8_original.txt', _lines(sum((extracted[s]['sources'] for s in order), [])))
    _publish(output_root/'lang8_corrected.txt', _lines(sum((extracted[s]['targets'] for s in order), [])))
    _publish(output_root/'derivation.json',json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/Preprocessed'))
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args(); report=derive(args.data_root,args.output_root)
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
