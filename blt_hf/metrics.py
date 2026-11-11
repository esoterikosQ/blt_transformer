"""Previous experiment's single-reference corpus GLEU and resumable NUS M2."""
from pathlib import Path
from .vendor.gleu import GLEU
from .m2_resumable import evaluate_m2_resumable
from .manifest import sha256_file, sha256_json


def scorer_identity():
    root = Path(__file__).parent
    names = ['metrics.py', 'm2_resumable.py', 'vendor/gleu.py', 'vendor/m2/levenshtein.py', 'vendor/m2/util.py']
    return sha256_json({name: sha256_file(root/name) for name in names})


def compute_gleu(reference, source, hypothesis):
    paths = [Path(v) for v in (reference, source, hypothesis)]
    lines = [p.read_text(encoding='utf-8').splitlines() for p in paths]
    if not lines[0] or len({len(v) for v in lines}) != 1:
        raise ValueError('GLEU requires nonempty aligned full source/reference/hypothesis files')
    calc = GLEU(4)
    calc.load_sources(str(source)); calc.load_references([str(reference)])
    stats = [0] * 10
    for i, hyp in enumerate(lines[2]):
        calc.load_hypothesis_sentence(hyp.split())
        stats = [x+y for x,y in zip(stats, calc.gleu_stats(i, r_ind=0))]
    # Preserve the old gleumodule.get_gleu_stats formatting before multiplication.
    return float(format(calc.gleu(stats), '.6f')) * 100


def compute_m2_with_checkpoints(hypothesis_path, source_gold_path, output_dir, **kwargs):
    return evaluate_m2_resumable(hypothesis_path, source_gold_path, output_dir, **kwargs)
