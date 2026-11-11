"""USER-RUN ON NEURON ONLY: tiny BLT backward, gradient and optimizer/RNG resume check."""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from blt_hf.runtime import require_neuron_job
from blt_hf.manifest import write_json,sha256_file


def tiny_config():
    from transformers import BltConfig
    part=dict(vocab_size=260,hidden_size=8,hidden_size_global=16,num_attention_heads=2,
              num_hidden_layers=2,intermediate_size=24,max_position_embeddings=128)
    cfg=BltConfig(encoder_config=dict(part,num_hidden_layers=1),decoder_config=part.copy(),
                  global_config=dict(part,hidden_size=16,intermediate_size=48),patcher_config=part.copy(),
                  cross_attn_k=2,encoder_hash_byte_group_size=[2,3],encoder_hash_byte_group_vocab=17,
                  encoder_hash_byte_group_nb_functions=1,use_cache=False,bos_token_id=1,eos_token_id=2,pad_token_id=3)
    spec=dict(local_attention_window_len=512,entropy_sliding_window=512,eos_id=2,
              global_attn_bias_type='block_causal',entropy_attn_bias_type='local_block_causal')
    for c in (cfg,cfg.encoder_config,cfg.decoder_config,cfg.global_config,cfg.patcher_config):
        c.attention_mode='osc';c.original_spec=spec;c._attn_implementation='eager'
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    require_neuron_job()
    import torch
    from safetensors.torch import load_file
    from blt_hf.patched.modeling_blt import BltForCausalLM
    from blt_hf.checkpoint import save_checkpoint,resolve_checkpoint
    from blt_hf.data_adapter import encode_pair,collate
    class ByteTokenizer:
        bos_token_id=1;eos_token_id=2
        def encode(self,text,**kwargs):return [b+4 for b in text.encode()]
    report={'scope':'tiny_model_training_and_resume','checker_hash':sha256_file(__file__),'real_B_training':False}
    try:
        torch.manual_seed(31);random.seed(31)
        def build():
            m=BltForCausalLM(tiny_config()).bfloat16().cuda()
            m.model.patcher.bfloat16().requires_grad_(False).eval()
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.95),fused=True)
            return m,o
        model,opt=build()
        batch={k:v.cuda() for k,v in collate([encode_pair(ByteTokenizer(),'오늘 조아요.','오늘 좋아요.')]).items()}
        def step(m,o):
            m.train();m.model.patcher.eval();o.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=m(**batch,use_cache=False)
            manual=torch.nn.functional.cross_entropy(out.logits[:,:-1].float().reshape(-1,260),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
            torch.testing.assert_close(out.loss,manual,rtol=0,atol=0)
            out.loss.backward()
            norms={name:sum(p.grad.float().norm().item() for p in part.parameters() if p.grad is not None)
                   for name,part in [('encoder',m.model.local_encoder),('global',m.model.global_transformer),('decoder',m.model.local_decoder)]}
            if any(v<=0 for v in norms.values()):raise AssertionError(f'Missing gradients: {norms}')
            if any(p.grad is not None for p in m.model.patcher.parameters()):raise AssertionError('Entropy patcher must remain frozen')
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step()
            for parameter,state in o.state.items():
                if parameter.dtype != torch.bfloat16:raise AssertionError(f'Non-BF16 parameter: {parameter.dtype}')
                for name in ('exp_avg','exp_avg_sq'):
                    if state[name].dtype != torch.bfloat16:raise AssertionError(f'{name} is {state[name].dtype}')
            return out.loss.item(),norms
        first,norms=step(model,opt)
        rng={'python':random.getstate(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/tmp') as d:
            save_checkpoint(d,model,opt,{'global_step':1,'run_signature':'tiny-training-resume'},[rng])
            second,_=step(model,opt)
            expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            restored,other_opt=build()
            cp,_=resolve_checkpoint(Path(d)/'latest.json')
            restored.load_state_dict(load_file(str(cp/'model.safetensors')),strict=True)
            saved=torch.load(cp/'training.pt',map_location='cpu',weights_only=True)
            other_opt.load_state_dict(saved['optimizer'])
            random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
            resumed,_=step(restored,other_opt)
            for key,value in restored.state_dict().items():
                torch.testing.assert_close(value.cpu(),expected[key],rtol=1e-6,atol=1e-7)
        report.update(status='passed',first_loss=first,second_loss=second,resumed_loss=resumed,
                      parameter_dtype='bfloat16',gradient_dtype='bfloat16',optimizer_state_dtype='bfloat16',
                      gradient_norms=norms,resume_rtol=1e-6,resume_atol=1e-7)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    write_json(args.output,report);print(json.dumps(report,indent=2),flush=True)
    return int(report['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
