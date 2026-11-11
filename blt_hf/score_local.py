"""Resume an existing M2 score with a node-local journal and periodic snapshots."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .data_adapter import dataset_split_path, read_tsv
from .eval import aggregate, parser as eval_parser
from .manifest import sha256_file, sha256_json, split_identity, fingerprint
from .metrics import compute_m2_with_checkpoints, scorer_identity
from .runtime import ROOT, NEURON_ROOT, atomic_json, exclusive_lock, local_path, require_neuron_job, StopRequest


def completed_count(path):
    if not path.exists(): return 0
    with path.open(encoding='utf-8') as stream:
        return sum(1 for _ in stream)


def verify_existing_score(root, dataset, split):
    manifest=json.loads((root/'run.json').read_text())
    payload={key:value for key,value in manifest.items() if key!='fingerprint'}
    if manifest['fingerprint']!=sha256_json({'required':fingerprint(payload),'complete':payload}):
        raise ValueError('Run manifest fingerprint corrupted')
    if (manifest['dataset'],manifest['split'])!=(dataset,split) or manifest['scorer_hash']!=scorer_identity():
        raise ValueError('Evaluation dataset, split, or scorer identity changed')
    tsv=dataset_split_path(ROOT/'data/Preprocessed',dataset,split)
    gold=tsv.with_suffix('.m2')
    if any(manifest[key]!=value for key,value in split_identity(tsv,gold,dataset=dataset,split=split).items()):
        raise ValueError('Evaluation data changed')
    scored=root/'scored'
    common=json.loads((scored/'gleu.json').read_text())
    journal=json.loads((scored/'m2/run_config.json').read_text())
    rows=read_tsv(tsv)
    if common.get('fingerprint')!=manifest['fingerprint'] or common.get('samples')!=len(rows):
        raise ValueError('Cached GLEU identity mismatch')
    if (scored/'source.txt').read_text().splitlines()!=[row.source for row in rows]:
        raise ValueError('Cached scorer source changed')
    if (scored/'reference.txt').read_text().splitlines()!=[row.target for row in rows]:
        raise ValueError('Cached scorer reference changed')
    hypothesis=scored/'hypothesis.txt'
    if (journal.get('hypothesis_path')!=str(hypothesis.resolve())
            or journal.get('hypothesis_sha256')!=sha256_file(hypothesis)
            or journal.get('source_gold_path')!=str(gold.resolve())
            or journal.get('source_gold_sha256')!=sha256_file(gold)
            or journal.get('examples')!=len(rows)):
        raise ValueError('Cached M2 input identity mismatch')
    return hypothesis,gold,scored/'m2'


def snapshot_journal(local, remote):
    """Publish each local file by atomic replacement, leaving prior snapshots intact."""
    began=time.monotonic()
    for source in sorted(local.rglob('*')):
        if not source.is_file() or source.name.endswith('.tmp'):
            continue
        destination=remote/source.relative_to(local)
        destination.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent)
        try:
            try:
                with os.fdopen(fd,'wb') as out, source.open('rb') as inp:
                    if source.name=='completed.jsonl':
                        # The scorer appends while snapshots run. Publish only whole
                        # records, so an interrupted copy cannot create a torn line.
                        data=inp.read()
                        out.write(data[:data.rfind(b'\n')+1])
                    else:
                        shutil.copyfileobj(inp,out)
                    out.flush()
                    if source.name=='completed.jsonl': os.fsync(out.fileno())
            except FileNotFoundError:
                continue  # A worker atomically replaced this mutable progress file.
            os.replace(temporary,destination)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    completed=remote/'completed.jsonl'
    print(json.dumps({'stage':'m2_snapshot','completed':completed_count(completed),
                      'elapsed_seconds':round(time.monotonic()-began,2)}),flush=True)


def worker(args):
    stop=StopRequest(args.max_seconds)
    result=compute_m2_with_checkpoints(args.hypothesis,args.gold,args.journal,
                                       workers=args.m2_workers,timeout_seconds=args.m2_timeout,
                                       max_passes=args.m2_passes,should_stop=stop)
    return 0 if result.status=='complete' else 75


def run(args):
    if ROOT.resolve()==NEURON_ROOT.resolve(): require_neuron_job(gpu=False)
    root=local_path(args.output_dir)
    hypothesis,gold,remote=verify_existing_score(root,args.dataset,args.split)
    with exclusive_lock(root):
        local_base=Path(os.environ.get('SLURM_TMPDIR') or '/tmp')
        local=Path(tempfile.mkdtemp(prefix='blt-m2-',dir=local_base))
        synced=False
        try:
            shutil.copytree(remote,local,dirs_exist_ok=True)
            completed_path=local/'completed.jsonl'
            print(json.dumps({'stage':'m2_local_journal_ready','path':str(local),
                              'completed_before':completed_count(completed_path)}),flush=True)
            command=[sys.executable,'-m','blt_hf.score_local','--worker',
                     '--hypothesis',str(hypothesis),'--gold',str(gold),'--journal',str(local),
                     '--m2-workers',str(args.m2_workers),'--m2-timeout',str(args.m2_timeout),
                     '--m2-passes',str(args.m2_passes),'--max-seconds',str(args.max_seconds)]
            process=subprocess.Popen(command)
            def forward_stop(*_):
                if process.poll() is None: process.send_signal(signal.SIGUSR1)
            signal.signal(signal.SIGUSR1,forward_stop)
            signal.signal(signal.SIGTERM,forward_stop)
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=args.snapshot_seconds)
                    except subprocess.TimeoutExpired:
                        pass
                    snapshot_journal(local,remote)
                result=process.wait()
            except BaseException:
                if process.poll() is None:
                    process.send_signal(signal.SIGUSR1)
                    try: process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.terminate();process.wait(timeout=10)
                snapshot_journal(local,remote)
                raise
            snapshot_journal(local,remote)
            synced=True
            if result not in (0,75): return result
            if result==75:
                progress=json.loads((remote/'progress.json').read_text())
                atomic_json(root/'scored/progress.json',{'status':'partial','m2':{
                    key:progress.get(key) for key in ('completed','total','unresolved','precision','recall','f0.5')}})
        finally:
            if synced: shutil.rmtree(local)
    if result==75: return 75
    final_args=eval_parser().parse_args(['--aggregate','--dataset',args.dataset,'--split',args.split,
                                          '--output-dir',str(root),'--m2-workers',str(args.m2_workers)])
    return aggregate(final_args)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['native','korean_learner','union','lang8'],default='korean_learner')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output-dir',required=False)
    p.add_argument('--m2-workers',type=int,default=8)
    p.add_argument('--m2-timeout',type=float,default=30.)
    p.add_argument('--m2-passes',type=int,default=4)
    p.add_argument('--max-seconds',type=int,default=6300)
    p.add_argument('--snapshot-seconds',type=int,default=300)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--hypothesis');p.add_argument('--gold');p.add_argument('--journal')
    return p


def main():
    args=parser().parse_args()
    if args.worker: return worker(args)
    if not args.output_dir: raise ValueError('--output-dir required')
    if args.snapshot_seconds<1 or args.m2_workers<1: raise ValueError('Invalid worker/snapshot setting')
    return run(args)


if __name__=='__main__': raise SystemExit(main())
