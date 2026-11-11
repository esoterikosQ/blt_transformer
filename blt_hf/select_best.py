"""Select a completed epoch checkpoint by full validation-set GLEU."""
import argparse
import json
from pathlib import Path

from .runtime import ROOT, atomic_json, local_path


def select_best(run_dir, evaluation_dirs):
    run_dir = Path(run_dir).resolve()
    training_manifest=json.loads((run_dir/'run.json').read_text())
    epochs = json.loads((run_dir/'epoch_checkpoints.json').read_text())
    by_step = {item['global_step']:item for item in epochs}
    if len(by_step) != len(epochs): raise ValueError('Duplicate epoch checkpoint step')
    candidates=[]; generation_contract=None
    for directory in map(Path,evaluation_dirs):
        manifest=json.loads((directory/'run.json').read_text())
        metrics=json.loads((directory/'scored/metrics.json').read_text())
        if manifest['split']!='val' or metrics.get('status')!='complete':
            raise ValueError(f'Incomplete or non-validation evaluation: {directory}')
        expected_run=training_manifest['run_id']
        if manifest['training_run_id']!=expected_run or manifest['checkpoint_step'] not in by_step:
            raise ValueError(f'Evaluation does not identify an epoch checkpoint: {directory}')
        if metrics['fingerprint']!=manifest['fingerprint']:
            raise ValueError(f'Evaluation fingerprint mismatch: {directory}')
        contract={key:manifest[key] for key in ('dataset','split','num_beams','length_penalty','max_new_bytes')}
        if generation_contract is None: generation_contract=contract
        elif contract!=generation_contract: raise ValueError('Validation generation settings differ')
        epoch=by_step[manifest['checkpoint_step']]
        candidates.append({**epoch,'val_gleu':metrics['gleu'],'val_m2_f0.5':metrics['m2']['f0.5'],
                           'evaluation_dir':str(directory),'evaluation_fingerprint':manifest['fingerprint']})
    if {item['global_step'] for item in candidates} != set(by_step):
        raise ValueError('Every epoch checkpoint must have one validation evaluation')
    best=max(candidates,key=lambda item:(item['val_gleu'],item['val_m2_f0.5'],-item['epoch']))
    pointer={key:best[key] for key in ('checkpoint','global_step','epoch','val_gleu','val_m2_f0.5',
                                      'evaluation_dir','evaluation_fingerprint')}
    atomic_json(run_dir/'best_gleu.json',pointer)
    return pointer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--evaluation-dir',action='append',required=True)
    args=parser.parse_args()
    result=select_best(local_path(args.run_dir),[local_path(path) for path in args.evaluation_dir])
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__': raise SystemExit(main())
