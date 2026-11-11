"""Resumable, unpadded HF generation shards and strict full-split GLEU/M2 scoring."""
import argparse
import json
import os
import time
from pathlib import Path
from .runtime import (ROOT, NEURON_ROOT, MODEL, CONVERSION, EVAL_RUNTIME_FILES,
                      local_path, require_neuron_job, code_identity, record_invocation,
                      tokenizer_identity, model_config_identity, ensure_json,
                      exclusive_lock, atomic_json, StopRequest)
from .manifest import sha256_file, sha256_json, split_identity, fingerprint, write_json
from .data_adapter import dataset_split_path, read_tsv, encode_prompt
from .generation import (GenerationConfig, generate_batch, HF_BACKEND,
                         GLOBAL_PREFIX_BACKENDS)
from .evaluation import shard_bounds, collect_records, publish_lines
from .metrics import compute_gleu, compute_m2_with_checkpoints, scorer_identity


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['native','korean_learner','union','lang8'],default='native')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--checkpoint',help='Immutable checkpoint directory or best/latest.json pointer')
    p.add_argument('--model-path',default=str(MODEL));p.add_argument('--conversion-report',default=str(CONVERSION))
    p.add_argument('--output-dir',required=True)
    p.add_argument('--shard-id',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--num-beams',type=int,choices=[1,4],default=1)
    p.add_argument('--generation-backend',choices=[HF_BACKEND,*GLOBAL_PREFIX_BACKENDS],default=HF_BACKEND)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--max-new-bytes',type=int,default=768)
    p.add_argument('--length-penalty',type=float,default=1.)
    p.add_argument('--aggregate',action='store_true',help='CPU-only full-corpus scoring; no model loaded')
    p.add_argument('--m2-workers',type=int,default=1)
    p.add_argument('--m2-timeout',type=float,default=30.)
    p.add_argument('--m2-passes',type=int,default=4)
    p.add_argument('--max-seconds',type=int,default=6300)
    return p


def generate(args):
    require_neuron_job()
    import torch, transformers
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from .model import load_model, configured_model
    from .checkpoint import resolve_checkpoint
    if not args.checkpoint: raise ValueError('--checkpoint required')
    if torch.cuda.device_count()!=1: raise ValueError('Evaluation shard uses exactly one allocated GPU')
    cp,meta=resolve_checkpoint(local_path(args.checkpoint),verify_optimizer=False)
    trained=meta['run_manifest']
    if trained['mode']!='train': raise ValueError('Smoke/overfit checkpoint cannot be used for main evaluation')
    if trained['dataset']!=args.dataset: raise ValueError('Cross-dataset evaluation requires a separate experimental plan')
    identity=code_identity((*EVAL_RUNTIME_FILES, 'blt_hf/cache/global_reuse.py',
                            'blt_hf/cache/frontier.py'))
    for name in ('blt_hf/model.py','blt_hf/patched/modeling_blt.py','blt_hf/attention.py','blt_hf/patching.py'):
        if trained['code_files'][name]!=identity['code_files'][name]: raise ValueError(f'Training/model implementation mismatch: {name}')
    model_path,conversion=local_path(args.model_path),local_path(args.conversion_report)
    if trained['conversion_hash']!=sha256_file(conversion): raise ValueError('Checkpoint base artifact mismatch')
    config=configured_model(model_path,attention_mode='osc')
    if trained['model_config_hash']!=model_config_identity(config,loader_dtype="bfloat16"): raise ValueError('Checkpoint runtime config mismatch')
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    cfg=GenerationConfig(num_beams=args.num_beams,batch_size=args.batch_size,max_new_bytes=args.max_new_bytes,
                         length_penalty=args.length_penalty,backend=args.generation_backend)
    tsv=dataset_split_path(ROOT/'data/Preprocessed',args.dataset,args.split);m2=tsv.with_suffix('.m2')
    rows=read_tsv(tsv)
    for row in rows: encode_prompt(tok,row.source,max_new_bytes=cfg.max_new_bytes,sample_id=row.sample_id)
    if trained['tokenizer_hash']!=tokenizer_identity(model_path): raise ValueError('Checkpoint tokenizer mismatch')
    manifest={**split_identity(tsv,m2,dataset=args.dataset,split=args.split),**identity,**cfg.to_dict(),
              'model_id':'facebook/blt-1b','model_revision':'8134b32f0b1d25d1248c30e8c7bdfd442d3bb380',
              'checkpoint_hash':meta['files']['model.safetensors'],'checkpoint_step':meta['global_step'],
              'conversion_checks':trained['conversion_checks'],'training_checks':meta['training_checks'],
              'evaluation_checks':'not_run', 'training_run_signature':meta['run_signature'],'training_run_id':trained['run_id'],'transformers_version':transformers.__version__,
              'torch_version':torch.__version__,'attn_implementation':'eager','attention_mode':'osc',
              'model_config_hash':model_config_identity(config,loader_dtype="bfloat16"),'tokenizer_hash':tokenizer_identity(model_path),
              'generation_backend':cfg.backend,
              'prefix_reuse':cfg.backend in GLOBAL_PREFIX_BACKENDS,
              'decoder_kv_reuse':False,
              'inference_dtype':'bfloat16','decode_policy':'utf8-replace-whitespace-collapse-v1',
              'scorer_hash':scorer_identity(),'shard_count':args.shard_count}
    # Include every extra execution/scorer field alongside the required standard fields.
    manifest['fingerprint']=sha256_json({'required':fingerprint(manifest),'complete':manifest})
    root=local_path(args.output_dir)
    with exclusive_lock(root):
        ensure_json(root/'run.json',manifest)
        record_invocation(root,stage='generate',identity=identity,
                          details={'checkpoint_step':meta['global_step'],'shard_id':args.shard_id})
    start,end=shard_bounds(len(rows),args.shard_count,args.shard_id)
    directory=root/'shards'/f'{args.shard_id:04d}'
    stop=StopRequest(args.max_seconds)
    with exclusive_lock(directory):
        if (directory/'complete.json').exists():
            # Aggregator rechecks complete coverage; no model reload needed for finished work.
            return 0
        model=load_model(model_path,conversion,attention_mode='osc',device='cpu')
        model.load_state_dict(load_file(str(cp/'model.safetensors')),strict=True)
        model.eval().requires_grad_(False).to('cuda')
        for offset in range(start,end,cfg.batch_size):
            ids=list(range(offset,min(offset+cfg.batch_size,end)))
            path=directory/'batches'/f'{offset:08d}.json'
            if path.exists():
                prior=json.loads(path.read_text())
                if prior['fingerprint']!=manifest['fingerprint'] or [r['index'] for r in prior['records']]!=ids:
                    raise ValueError('Partial generation identity/range mismatch')
                continue
            if stop:
                atomic_json(directory/'progress.json',{'next_index':offset,'end':end,'status':'paused'})
                return 75
            began=time.monotonic()
            predictions=generate_batch(model,tok,[rows[i].source for i in ids],cfg)
            records=[{'index':i,'source':rows[i].source,**prediction} for i,prediction in zip(ids,predictions)]
            write_json(path,{'fingerprint':manifest['fingerprint'],'records':records,'seconds':time.monotonic()-began})
            atomic_json(directory/'progress.json',{'next_index':ids[-1]+1,'end':end,'status':'running'})
            print(json.dumps({'shard':args.shard_id,'completed':ids[-1]+1-start,'total':end-start}),flush=True)
        write_json(directory/'complete.json',{'fingerprint':manifest['fingerprint'],'shard_id':args.shard_id,
                                               'start':start,'end':end,'sample_count':len(rows),'shard_count':args.shard_count,
                                               'batch_hashes':{p.name:sha256_file(p) for p in sorted((directory/'batches').glob('*.json'))}})
        atomic_json(directory/'progress.json',{'next_index':end,'end':end,'status':'complete'})
    return 0


def aggregate(args):
    # Pure CPU path. Scheduler launcher enforces Neuron CPU resources; also usable locally for fixtures.
    prepared_at=time.monotonic()
    if ROOT.resolve() == NEURON_ROOT.resolve():
        require_neuron_job(gpu=False)
    root=local_path(args.output_dir)
    with exclusive_lock(root):
        manifest=json.loads((root/'run.json').read_text())
        payload={k:v for k,v in manifest.items() if k!='fingerprint'}
        if manifest['fingerprint']!=sha256_json({'required':fingerprint(payload),'complete':payload}):
            raise ValueError('Run manifest fingerprint corrupted')
        if manifest['dataset']!=args.dataset or manifest['split']!=args.split: raise ValueError('Dataset/split mismatch')
        if manifest['scorer_hash']!=scorer_identity(): raise ValueError('Scoring implementation changed')
        tsv=dataset_split_path(ROOT/'data/Preprocessed',args.dataset,args.split);m2=tsv.with_suffix('.m2')
        actual=split_identity(tsv,m2,dataset=args.dataset,split=args.split)
        if any(manifest[k]!=v for k,v in actual.items()): raise ValueError('Evaluation data changed')
        rows=read_tsv(tsv)
        scored=root/'scored'
        common_file=scored/'gleu.json'
        m2_manifest_file=scored/'m2/run_config.json'
        cached=common_file.exists() and m2_manifest_file.exists()
        if cached:
            # A prior invocation already verified every generation batch and published
            # these immutable scorer inputs. M2's own manifest binds its journal to
            # the exact hypothesis and gold files. Avoid re-reading thousands of
            # Lustre batch files and recomputing GLEU on every timed continuation.
            common=json.loads(common_file.read_text())
            m2_manifest=json.loads(m2_manifest_file.read_text())
            if common.get('fingerprint')!=manifest['fingerprint'] or common.get('samples')!=len(rows):
                raise ValueError('Cached GLEU identity mismatch')
            publish_lines(scored/'source.txt',[row.source for row in rows])
            publish_lines(scored/'reference.txt',[row.target for row in rows])
            hypothesis=scored/'hypothesis.txt'
            if (m2_manifest.get('hypothesis_path')!=str(hypothesis.resolve())
                    or m2_manifest.get('hypothesis_sha256')!=sha256_file(hypothesis)
                    or m2_manifest.get('source_gold_path')!=str(m2.resolve())
                    or m2_manifest.get('source_gold_sha256')!=sha256_file(m2)
                    or m2_manifest.get('examples')!=len(rows)):
                raise ValueError('Cached M2 input identity mismatch')
        else:
            records=collect_records(root,manifest)
            if any(r['source']!=row.source for r,row in zip(records,rows)): raise ValueError('Prediction source/order mismatch')
            publish_lines(scored/'source.txt',[r.source for r in rows])
            publish_lines(scored/'reference.txt',[r.target for r in rows])
            publish_lines(scored/'hypothesis.txt',[r['text'] for r in records])
            gleu=compute_gleu(scored/'reference.txt',scored/'source.txt',scored/'hypothesis.txt')
            common={'fingerprint':manifest['fingerprint'],'samples':len(rows),'gleu':gleu,
                    'eos_rate':sum(r['eos_reached'] for r in records)/len(records),
                    'copy_rate':sum(r['text'].split()==row.source.split() for r,row in zip(records,rows))/len(records),
                    'invalid_utf8':sum(not r['valid_utf8'] for r in records),
                    'invalid_token_ids':sum(not r['valid_token_ids'] for r in records),
                    'budget_exhausted':sum(r['budget_exhausted'] for r in records)}
            ensure_json(common_file,common)
        print(json.dumps({'stage':'score_inputs_ready','cached':cached,
                          'elapsed_seconds':round(time.monotonic()-prepared_at,2)}),flush=True)
        if (scored/'metrics.json').exists():
            final=json.loads((scored/'metrics.json').read_text())
            if final.get('status')!='complete' or final.get('fingerprint')!=manifest['fingerprint']:
                raise ValueError('Existing final metrics do not match this completed evaluation')
            atomic_json(scored/'progress.json',{'status':'complete','m2':final['m2']})
            print(json.dumps(final,ensure_ascii=False,indent=2),flush=True);return 0
        stop=StopRequest(args.max_seconds)
        workers=min(args.m2_workers,int(os.environ.get('SLURM_CPUS_PER_TASK',str(args.m2_workers))))
        result=compute_m2_with_checkpoints(scored/'hypothesis.txt',m2,scored/'m2',workers=workers,
                                          timeout_seconds=args.m2_timeout,timeout_multiplier=4.,max_passes=args.m2_passes,should_stop=stop)
        if result.status!='complete':
            atomic_json(scored/'progress.json',{'status':'partial','m2':result.to_dict()})
            return 75
        final={**common,'m2':result.to_dict(),'status':'complete','scorer_hash':scorer_identity(),
               'evaluation_checks':'passed','training_checks':manifest['training_checks'],
               'conversion_checks':manifest['conversion_checks']}
        write_json(scored/'metrics.json',final)
        atomic_json(scored/'progress.json',{'status':'complete','m2':result.to_dict()})
        print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)
    return 0


def main():
    args=parser().parse_args()
    return aggregate(args) if args.aggregate else generate(args)
if __name__=='__main__': raise SystemExit(main())
