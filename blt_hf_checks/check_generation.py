"""Forward-only real-B generation check; safe for itcerdo, no optimizer/backward."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from blt_hf.manifest import sha256_file, sha256_json, write_json

def main():
    import torch
    from transformers import AutoTokenizer
    from blt_hf.model import load_model, configured_model
    from blt_hf.runtime import model_config_identity
    from blt_hf.generation import GenerationConfig, generate_batch
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    path=ROOT/'artifacts/converted/blt-1b-hf-own'
    report={'scope':'inference_only','checks':[],'generation_hash':sha256_file(ROOT/'blt_hf/generation.py')}
    try:
        config=configured_model(path,attention_mode='osc')
        model=load_model(path,ROOT/'blt_hf_checks/manifests/conversion_B_20260915.json',attention_mode='osc').eval()
        # Training/checkpoint metadata uses this same pre-load configuration identity.
        report['raw_config_differences']={k:[config.to_dict().get(k),model.config.to_dict().get(k)] for k in set(config.to_dict())|set(model.config.to_dict()) if config.to_dict().get(k)!=model.config.to_dict().get(k)}
        report['factory_config_hash']=model_config_identity(config,loader_dtype='bfloat16')
        report['loaded_config_hash']=model_config_identity(model.config)
        if report['factory_config_hash']!=report['loaded_config_hash']:
            raise AssertionError('Factory and loaded runtime config differ')
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        sources=['가','나','오늘 날씨가 조아요.']
        for beams in (1,4):
            singles=[generate_batch(model,tok,[s],GenerationConfig(num_beams=beams,max_new_bytes=8))[0] for s in sources]
            batch=generate_batch(model,tok,sources,GenerationConfig(num_beams=beams,batch_size=3,max_new_bytes=8))
            report['checks'].append({'beams':beams,'same_token_ids':all(a['token_ids']==b['token_ids'] for a,b in zip(singles,batch)),
                                     'outputs':batch})
            if singles!=batch:raise AssertionError('Batch grouping changed generation output')
        # Validate the unchanged BF16 parameter/compute path without creating gradients.
        model.model.patcher.bfloat16().requires_grad_(False).eval()
        wrong_dtypes=[(name,str(parameter.dtype)) for name,parameter in model.named_parameters()
                      if parameter.dtype != torch.bfloat16]
        if wrong_dtypes:raise AssertionError(f'Non-BF16 model parameters: {wrong_dtypes[:8]}')
        from blt_hf.data_adapter import encode_pair,collate
        batch={k:v.cuda() for k,v in collate([encode_pair(tok,'가','나')]).items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(**batch,use_cache=False).loss
        if not torch.isfinite(loss):raise AssertionError('Nonfinite mixed precision forward')
        report.update(status='passed',parameter_dtype='bfloat16',compute_dtype='bfloat16',
                      bf16_forward_loss=loss.item(),backward_executed=False,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    write_json(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return int(report['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
