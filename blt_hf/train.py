"""Neuron-only BF16 full-main-model fine-tuning with a frozen entropy patcher."""
import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from .training import epoch_batches, rank_work, lr_factor, normalize_gradient_scale
from .runtime import (ROOT, MODEL, CONVERSION, TRAIN_RUNTIME_FILES, local_path,
                      require_neuron_job, code_identity, record_invocation,
                      tokenizer_identity, model_config_identity, conversion_verification,
                      ensure_json, exclusive_lock, StopRequest, atomic_json)
from .manifest import sha256_file, sha256_json, write_json
from .generation import GenerationConfig, HF_BACKEND, GLOBAL_PREFIX_GUARDED_BACKEND


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['native', 'korean_learner', 'union', 'lang8'], default='native')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--model-path', default=str(MODEL))
    p.add_argument('--conversion-report', default=str(CONVERSION))
    p.add_argument('--resume', help='Checkpoint directory or latest.json; exact run contract required')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--effective-batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--warmup-ratio', type=float, default=.05)
    p.add_argument('--selection-metric', choices=['val_loss', 'val_gleu'], default='val_loss')
    p.add_argument('--early-stopping-patience', type=int, default=0,
                   help='Stop after this many complete epochs without a strictly higher validation GLEU; 0 disables')
    p.add_argument('--validation-beams', type=int, choices=[1, 4], default=1)
    p.add_argument('--validation-batch-size', type=int, default=1)
    p.add_argument('--validation-max-new-bytes', type=int, default=768)
    p.add_argument('--validation-generation-backend',
                   choices=[HF_BACKEND, GLOBAL_PREFIX_GUARDED_BACKEND], default=HF_BACKEND)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--clip-grad-norm', type=float, default=1.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--max-steps', type=int, default=0, help='Stop this invocation after N updates, without changing schedule')
    p.add_argument('--max-seconds', type=int, default=21000)
    p.add_argument('--mode', choices=['train', 'smoke', 'overfit'], default='train')
    p.add_argument('--overfit-steps', type=int, default=200)
    return p


def validation_generation_config(args):
    if args.validation_generation_backend == GLOBAL_PREFIX_GUARDED_BACKEND and args.selection_metric != 'val_gleu':
        raise ValueError('Guarded validation requires GLEU checkpoint selection')
    return GenerationConfig(num_beams=args.validation_beams,
                            max_new_bytes=args.validation_max_new_bytes,
                            batch_size=args.validation_batch_size,
                            backend=args.validation_generation_backend)


def run(args):
    require_neuron_job()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from .model import load_model
    from .data_adapter import GecDataset, collate, dataset_split_path
    from .checkpoint import resolve_checkpoint, save_checkpoint
    if args.selection_metric == 'val_gleu':
        from .data_adapter import read_tsv
        from .generation import generate_batch
        from .integrated_validation import (
            gleu_early_stopping_update, order_predictions, score_validation_epoch, validation_groups,
        )

    rank, world, local_rank = (int(os.environ.get(k, d)) for k, d in
                               [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    def stage(name, **details):
        print(json.dumps({'event':'train_stage','stage':name,'rank':rank,
                          'local_rank':local_rank,'world_size':world,**details}),flush=True)
    if world != torch.cuda.device_count() or local_rank >= world:
        raise ValueError('torchrun world size must equal allocated visible GPU count')
    if min(args.epochs, args.effective_batch, args.save_every, args.validation_batch_size,
           args.validation_max_new_bytes) < 1 or args.lr <= 0 or args.clip_grad_norm <= 0:
        raise ValueError('Invalid positive training setting')
    if min(args.max_steps, args.max_seconds, args.weight_decay, args.early_stopping_patience) < 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError('Negative training setting')
    if args.early_stopping_patience and (args.mode != 'train' or args.selection_metric != 'val_gleu'):
        raise ValueError('GLEU early stopping requires train mode and val_gleu selection')
    validation_cfg = validation_generation_config(args)
    if args.mode == 'overfit' and (world != 1 or args.overfit_steps < 1):
        raise ValueError('Overfit diagnostic requires one GPU and positive steps')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) // world))
    if world > 1: dist.init_process_group('nccl', device_id=device)
    stage('distributed_initialized',device=str(device))
    stopper = StopRequest(args.max_seconds)
    run_dir = local_path(args.run_dir)
    guard = exclusive_lock(run_dir) if rank == 0 else contextlib.nullcontext()
    try:
        with guard:
            random.seed(args.seed + rank); torch.manual_seed(args.seed + rank); torch.cuda.manual_seed(args.seed + rank)
            model_path, conversion = local_path(args.model_path), local_path(args.conversion_report)
            tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            data_root = ROOT / 'data/Preprocessed'
            paths = {f'{args.dataset}/{split}':dataset_split_path(data_root,args.dataset,split)
                     for split in ('train','val')}
            # Audit ALL provided splits without altering or truncating any records.
            # Dataset encoding below verifies actual tokenizer IDs for train and validation.
            train = GecDataset(paths[f'{args.dataset}/train'], tok)
            val = GecDataset(paths[f'{args.dataset}/val'], tok)
            train_items = [train[i] for i in range(len(train))]
            val_items = [val[i] for i in range(len(val))]
            val_rows = read_tsv(paths[f'{args.dataset}/val']) if args.selection_metric == 'val_gleu' else None
            if val_rows is not None and len(val_rows) != len(val_items):
                raise ValueError('Validation text and encoded row counts differ')
            if args.mode == 'smoke':
                # Stress longest TRAIN rows, in a diagnostic run that evaluation rejects.
                train_items = sorted(train_items, key=lambda item: len(item.input_ids), reverse=True)[:max(4, 2*world)]
            if args.mode == 'overfit':
                # Dedicated diagnostic only, never substituted for the full experiment/evaluation.
                train_items = train_items[:4]
            count = len(train_items)
            batch_size = 4 if args.mode == 'overfit' else args.effective_batch
            steps_per_epoch = math.ceil(count / batch_size)
            epochs = args.overfit_steps if args.mode == 'overfit' else args.epochs
            total_steps = steps_per_epoch * epochs
            warmup = round(total_steps * args.warmup_ratio) if args.mode != 'overfit' else 0
            model = load_model(model_path, conversion, attention_mode='osc', device='cpu')
            model.model.patcher.bfloat16().requires_grad_(False).eval()
            wrong_dtypes = [(name, str(param.dtype)) for name, param in model.named_parameters()
                            if param.dtype != torch.bfloat16]
            if wrong_dtypes:
                raise RuntimeError(f'BF16 checkpoint/runtime contract violated: {wrong_dtypes[:8]}')
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # BF16 parameters, gradients and two Adam moments; activations/workspaces need extra room.
            minimum = trainable * 8 + sum(p.numel()*p.element_size() for p in model.parameters() if not p.requires_grad)
            free, _ = torch.cuda.mem_get_info(device)
            if minimum > free * .95:
                raise RuntimeError(f'BF16 Adam/DDP minimum {minimum} exceeds available {free}; plan sharding')
            gleu_files = ('blt_hf/generation.py', 'blt_hf/integrated_validation.py',
                          'blt_hf/metrics.py', 'blt_hf/evaluation.py', 'blt_hf/vendor/gleu.py')
            if args.validation_generation_backend == GLOBAL_PREFIX_GUARDED_BACKEND:
                gleu_files += ('blt_hf/cache/global_reuse.py', 'blt_hf/cache/frontier.py')
            identity = code_identity(TRAIN_RUNTIME_FILES + (gleu_files if args.selection_metric == 'val_gleu' else ()))
            manifest = {'schema_version': 2, 'run_id': str(run_dir.relative_to(ROOT)), 'dataset': args.dataset, 'mode': args.mode,
                        'train_file_hash': sha256_file(paths[f'{args.dataset}/train']),
                        'val_file_hash': sha256_file(paths[f'{args.dataset}/val']),
                        'train_count': len(train), 'val_count': len(val), 'diagnostic_count': count,
                        'diagnostic_selection': 'longest_train_rows' if args.mode == 'smoke' else ('first_four_train_rows' if args.mode == 'overfit' else 'all_train_rows'),
                        'conversion_hash': sha256_file(conversion), 'model_config_hash': model_config_identity(model.config),
                        'tokenizer_hash': tokenizer_identity(model_path), 'environment_lock_hash': sha256_file(ROOT/'blt_hf/requirements.lock.txt'),
                        'torch_version': torch.__version__, 'attention_mode': 'osc', 'attn_implementation': 'eager',
                        'parameter_dtype': 'bfloat16', 'gradient_dtype': 'bfloat16',
                        'optimizer_state_dtype': 'bfloat16', 'compute_dtype': 'bfloat16', 'entropy_dtype': 'bfloat16',
                        'entropy_trainable': False, 'gradient_checkpointing': True, 'use_cache': False,
                        'world_size': world, 'ddp_bucket_priming': world > 1, 'micro_batch_size': 1, 'effective_batch': batch_size,
                        'loss_normalization': 'global_supervised_token_mean', 'epochs': epochs,
                        'total_steps': total_steps, 'warmup_ratio': args.warmup_ratio,
                        'warmup_steps': warmup, 'seed': args.seed,
                        'selection_metric': args.selection_metric,
                        'early_stopping_patience': args.early_stopping_patience,
                        'validation_num_beams': args.validation_beams if args.selection_metric == 'val_gleu' else None,
                        'validation_batch_size': args.validation_batch_size if args.selection_metric == 'val_gleu' else None,
                        'validation_max_new_bytes': args.validation_max_new_bytes if args.selection_metric == 'val_gleu' else None,
                        'validation_generation_backend': args.validation_generation_backend if args.selection_metric == 'val_gleu' else None,
                        'lr': args.lr, 'weight_decay': args.weight_decay, 'clip_grad_norm': args.clip_grad_norm,
                        'optimizer': 'AdamW-fused-bfloat16-state', 'betas': [.9, .95], 'eps': 1e-8,
                        'conversion_checks': conversion_verification(conversion), **identity}
            signature = sha256_json(manifest)
            manifest_status=[None]
            if rank == 0:
                try:
                    if (run_dir/'run.json').exists() and not args.resume:
                        raise ValueError('Existing run requires explicit --resume, or choose a fresh run directory')
                    ensure_json(run_dir/'run.json', manifest)
                    record_invocation(run_dir, stage='train', identity=identity,
                                      details={'resume': bool(args.resume), 'world_size': world})
                    manifest_status[0]={'error':None}
                except Exception as exc:
                    manifest_status[0]={'error':f'{type(exc).__name__}: {exc}'}
            if world > 1: dist.broadcast_object_list(manifest_status,src=0)
            if manifest_status[0]['error'] is not None:
                raise ValueError(f"Run manifest verification failed: {manifest_status[0]['error']}")
            stage('manifest_verified',resume=bool(args.resume),code_hash=identity['code_hash'])
            state = {'epoch': 0, 'next_batch': 0, 'global_step': 0, 'best_val_loss': None,
                     'last_val_loss': None, 'best_val_gleu': None, 'last_val_gleu': None,
                     'epochs_without_gleu_improvement': 0,
                     'last_training_loss': None, 'run_signature': signature,
                     'training_checks': 'not_run', 'evaluation_checks': 'not_run'}
            restore = None
            if args.resume:
                payload = [None]
                if rank == 0:
                    try:
                        cp, meta = resolve_checkpoint(local_path(args.resume))
                        payload[0] = {'path':str(cp),'metadata':meta,'error':None}
                    except Exception as exc:
                        payload[0] = {'path':None,'metadata':None,
                                      'error':f'{type(exc).__name__}: {exc}'}
                if world > 1: dist.broadcast_object_list(payload,src=0)
                if payload[0]['error'] is not None:
                    raise ValueError(f"Rank-0 checkpoint verification failed: {payload[0]['error']}")
                cp, meta = Path(payload[0]['path']), payload[0]['metadata']
                stage('checkpoint_hashes_verified',checkpoint=cp.name)
                if meta['run_signature'] != signature: raise ValueError('Resume contract mismatch (code/data/world/schedule/model)')
                model.load_state_dict(load_file(str(cp/'model.safetensors')), strict=True)
                stage('model_state_loaded',checkpoint=cp.name)
                restore = torch.load(cp/'training.pt', map_location='cpu', weights_only=True)
                stage('optimizer_state_file_loaded',checkpoint=cp.name)
                state.update({k: meta[k] for k in state})
                if (run_dir/'completed.json').exists():
                    done = json.loads((run_dir/'completed.json').read_text())
                    if done.get('status') == 'complete' and (state['epoch'] >= epochs or done.get('stop_reason') == 'gleu_patience'):
                        return 0 if done.get('overfit_passed', True) else 1
            model.to(device)
            stage('model_moved_to_device')
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and p.ndim >= 2], 'weight_decay': args.weight_decay},
                      {'params': [p for p in model.parameters() if p.requires_grad and p.ndim < 2], 'weight_decay': 0.}]
            optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(.9, .95), eps=1e-8, fused=True)
            wrapper = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else model
            if world > 1:
                # Build/rebuild gradient bucket views BEFORE allocating Adam moments. No parameter update.
                rng_before = (random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state(device))
                probe = {k:v.to(device) for k,v in collate([train_items[0]]).items()}
                model.train(); model.model.patcher.eval()
                for _ in range(2):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        probe_loss = wrapper(**probe, use_cache=False).loss
                    probe_loss.backward()
                    optimizer.zero_grad(set_to_none=False)
                del probe, probe_loss
                random.setstate(rng_before[0]); torch.set_rng_state(rng_before[1]); torch.cuda.set_rng_state(rng_before[2], device)
            if restore:
                optimizer.load_state_dict(restore['optimizer'])
                rng = restore['rng_states'][rank]
                random.setstate(rng['python']); torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'], device)
                del restore
                stage('optimizer_and_rng_restored',global_step=state['global_step'])
            def stopping():
                flag = torch.tensor(int(bool(stopper)), device=device)
                if world > 1: dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                return bool(flag.item())
            def save(best=False):
                rng = {'python': random.getstate(), 'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}
                all_rng = [None] * world
                if world > 1: dist.all_gather_object(all_rng, rng)
                else: all_rng[0] = rng
                if rank == 0:
                    checkpoint_name=save_checkpoint(run_dir, model, optimizer, {**state, 'run_manifest': manifest}, all_rng, best=best)
                    if best and args.selection_metric == 'val_gleu':
                        atomic_json(run_dir/'best_gleu.json',
                                    {'checkpoint': checkpoint_name, 'global_step': state['global_step'],
                                     'epoch': state['epoch'], 'val_gleu': state['best_val_gleu'],
                                     'num_beams': args.validation_beams,
                                     'generation_backend': args.validation_generation_backend})
                    stage('checkpoint_published',checkpoint=checkpoint_name,best=best)
                if world > 1: dist.barrier()
            def validate():
                model.eval()
                loss_sum, ntokens, seen = 0., 0, 0
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for i in range(rank, len(val_items), world):
                        if stopper: break
                        item = val_items[i]
                        batch = {k:v.to(device) for k,v in collate([item]).items()}
                        loss = model(**batch, use_cache=False).loss
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite validation loss')
                        n = sum(v != -100 for v in item.labels[1:])
                        loss_sum += loss.item()*n; ntokens += n; seen += 1
                stats = torch.tensor([loss_sum, ntokens, seen], device=device, dtype=torch.float64)
                if world > 1: dist.all_reduce(stats)
                if int(stats[2]) != len(val_items): return None
                return (stats[0]/stats[1]).item()
            def validate_gleu(epoch_number):
                model.eval()
                config = validation_cfg
                started = time.monotonic()
                local = []
                for indices in validation_groups(val_items, rank=rank, world_size=world,
                                                 batch_size=args.validation_batch_size):
                    if stopper: break
                    predictions = generate_batch(model, tok, [val_rows[index].source for index in indices], config)
                    local.extend({'index': index, 'text': prediction['text'],
                                  'invalid_utf8': not prediction['valid_utf8'],
                                  'budget_exhausted': prediction['budget_exhausted'],
                                  'cache_preliminary_skips': prediction.get('cache_preliminary_skips', 0),
                                  'cache_guard_refreshes': prediction.get('cache_guard_refreshes', 0)}
                                 for index, prediction in zip(indices, predictions))
                part = {'stopped': bool(stopper), 'predictions': local,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
                if world > 1:
                    gathered = [None] * world
                    dist.all_gather_object(gathered, part)
                else:
                    gathered = [part]
                outcome = [None]
                if rank == 0:
                    try:
                        if any(item['stopped'] for item in gathered):
                            outcome[0] = {'gleu': None}
                        else:
                            predictions = order_predictions([item['predictions'] for item in gathered], len(val_rows))
                            report = score_validation_epoch(
                                run_dir, epoch=epoch_number, global_step=state['global_step'],
                                sources=[row.source for row in val_rows],
                                references=[row.target for row in val_rows], predictions=predictions,
                                num_beams=args.validation_beams,
                                batch_size=args.validation_batch_size,
                                max_new_bytes=args.validation_max_new_bytes,
                                generation_backend=args.validation_generation_backend,
                                cache_preliminary_skips=sum(record['cache_preliminary_skips'] for item in gathered for record in item['predictions']),
                                cache_guard_refreshes=sum(record['cache_guard_refreshes'] for item in gathered for record in item['predictions']),
                                validation_file_hash=manifest['val_file_hash'])
                            outcome[0] = {'gleu': report['gleu'],
                                          'invalid_utf8': sum(record['invalid_utf8'] for item in gathered for record in item['predictions']),
                                          'budget_exhausted': sum(record['budget_exhausted'] for item in gathered for record in item['predictions']),
                                          'generation_backend': args.validation_generation_backend,
                                          'cache_preliminary_skips': report['cache_preliminary_skips'],
                                          'cache_guard_refreshes': report['cache_guard_refreshes'],
                                          'max_rank_peak_allocated_bytes': max(item['peak_allocated_bytes'] for item in gathered)}
                            stage('validation_gleu_complete', epoch=epoch_number,
                                  elapsed_seconds=round(time.monotonic()-started, 2), **outcome[0])
                    except Exception as exc:
                        outcome[0] = {'error': f'{type(exc).__name__}: {exc}'}
                if world > 1: dist.broadcast_object_list(outcome, src=0)
                if 'error' in outcome[0]:
                    raise RuntimeError(f"Validation GLEU failed: {outcome[0]['error']}")
                return outcome[0]['gleu']
            invocation_steps = 0
            grad_report = None
            stage('training_loop_entered',global_step=state['global_step'],epoch=state['epoch'],next_batch=state['next_batch'])
            stop_reason = ('gleu_patience' if args.early_stopping_patience and
                           state['epochs_without_gleu_improvement'] >= args.early_stopping_patience and
                           state['epoch'] < epochs else 'max_epochs')
            for epoch in range(state['epoch'], epochs):
                if stop_reason == 'gleu_patience': break
                epoch_started = time.monotonic()
                batches = epoch_batches(count, batch_size, seed=args.seed, epoch=epoch)
                start = state['next_batch'] if epoch == state['epoch'] else 0
                for batch_index in range(start, len(batches)):
                    if stopping():
                        save(); return 75
                    model.train(); model.model.patcher.eval()
                    indices = batches[batch_index]
                    denominator = sum(sum(v != -100 for v in train_items[i].labels[1:]) for i in indices)
                    work = rank_work(indices, rank, world)
                    optimizer.zero_grad(set_to_none=world == 1)
                    rate = args.lr * lr_factor(state['global_step'], total_steps, warmup)
                    for group in optimizer.param_groups: group['lr'] = rate
                    losses = 0.
                    for micro, index in enumerate(work):
                        item = train_items[index if index is not None else indices[0]]
                        batch = {k:v.to(device) for k,v in collate([item]).items()}
                        sync = wrapper.no_sync() if world > 1 and micro+1 < len(work) else contextlib.nullcontext()
                        with sync:
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                loss = wrapper(**batch, use_cache=False).loss
                                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                                tokens = sum(v != -100 for v in item.labels[1:]) if index is not None else 0
                                weighted = loss * tokens * normalize_gradient_scale(world, denominator)
                            weighted.backward()
                        losses += loss.item() * tokens
                    wrong_grad_dtypes = [(name, str(param.grad.dtype)) for name, param in model.named_parameters()
                                         if param.grad is not None and param.grad.dtype != torch.bfloat16]
                    if wrong_grad_dtypes:
                        raise RuntimeError(f'BF16 gradient contract violated: {wrong_grad_dtypes[:8]}')
                    if grad_report is None:
                        grad_report = {}
                        for name, component in [('encoder', model.model.local_encoder), ('global', model.model.global_transformer),
                                                ('decoder', model.model.local_decoder), ('hash', model.model.encoder_hash_tok_embedding)]:
                            # Check one parameter at a time to avoid an extra full-model allocation.
                            norms = [torch.linalg.vector_norm(p.grad.detach()).item() for p in component.parameters() if p.grad is not None]
                            grad_report[name] = math.sqrt(sum(n*n for n in norms))
                        if any(not math.isfinite(v) or v <= 0 for v in grad_report.values()):
                            raise RuntimeError(f'Missing/nonfinite gradient flow: {grad_report}')
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    for parameter, optimizer_state in optimizer.state.items():
                        if parameter.dtype != torch.bfloat16:
                            raise RuntimeError(f'Optimizer received non-BF16 parameter: {parameter.dtype}')
                        for state_name in ('exp_avg', 'exp_avg_sq'):
                            value = optimizer_state.get(state_name)
                            if value is None or value.dtype != torch.bfloat16:
                                dtype = None if value is None else value.dtype
                                raise RuntimeError(f'Adam {state_name} must remain BF16, got {dtype}')
                    optimizer.zero_grad(set_to_none=world == 1)
                    state.update(epoch=epoch, next_batch=batch_index+1, global_step=state['global_step']+1)
                    invocation_steps += 1
                    value = torch.tensor(losses, device=device, dtype=torch.float64)
                    if world > 1: dist.all_reduce(value)
                    state['last_training_loss'] = value.item()/denominator
                    if rank == 0:
                        print(json.dumps({'step': state['global_step'], 'epoch': epoch, 'loss': value.item()/denominator,
                                          'lr': rate, 'grad_norm': float(norm), 'peak_allocated': torch.cuda.max_memory_allocated(device)}), flush=True)
                    preempted = stopping()
                    stop = preempted or (args.max_steps > 0 and invocation_steps >= args.max_steps) or (args.mode == 'smoke' and invocation_steps >= 2)
                    if stop:
                        save()
                        if rank == 0:
                            atomic_json(run_dir/'progress.json', {**state, 'status': 'paused', 'gradient_norms': grad_report,
                                                                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)})
                        return 75 if preempted else 0
                    if state['global_step'] % args.save_every == 0: save()
                # Validation belongs to this epoch; a resumed end-of-epoch checkpoint repeats it safely.
                training_seconds = time.monotonic() - epoch_started
                validation_started = time.monotonic()
                val_loss = validate() if args.mode == 'train' else None
                if args.mode == 'train' and val_loss is None:
                    save(); return 75
                if rank == 0 and val_loss is not None:
                    stage('validation_loss_complete', epoch=epoch+1, val_loss=val_loss,
                          training_seconds=round(training_seconds, 2),
                          validation_loss_seconds=round(time.monotonic()-validation_started, 2))
                val_gleu = validate_gleu(epoch+1) if args.mode == 'train' and args.selection_metric == 'val_gleu' else None
                if args.mode == 'train' and args.selection_metric == 'val_gleu' and val_gleu is None:
                    save(); return 75
                loss_improved = val_loss is not None and (state['best_val_loss'] is None or val_loss < state['best_val_loss'])
                if loss_improved: state['best_val_loss'] = val_loss
                state['last_val_loss'] = val_loss
                if args.mode == 'train' and args.selection_metric == 'val_gleu':
                    gleu_is_better, stale_epochs, should_stop = gleu_early_stopping_update(
                        val_gleu, state['best_val_gleu'],
                        state['epochs_without_gleu_improvement'], args.early_stopping_patience)
                    state['epochs_without_gleu_improvement'] = stale_epochs
                    if gleu_is_better: state['best_val_gleu'] = val_gleu
                else:
                    gleu_is_better, should_stop = False, False
                state['last_val_gleu'] = val_gleu
                best = gleu_is_better if args.selection_metric == 'val_gleu' else loss_improved
                state.update(epoch=epoch+1, next_batch=0)
                if args.mode == 'train':
                    state['training_checks'] = 'passed'
                elif epoch+1 == epochs:
                    state['training_checks'] = 'passed' if state['last_training_loss'] < .05 else 'failed'
                if args.mode != 'overfit' or epoch+1 == epochs: save(best=best)
                if rank == 0:
                    atomic_json(run_dir/'progress.json', {**state, 'status': 'running',
                                                          'gradient_norms': grad_report})
                    stage('epoch_complete', epoch=epoch+1, global_step=state['global_step'],
                          selection_metric=args.selection_metric, best=best,
                          val_loss=val_loss, val_gleu=val_gleu,
                          elapsed_seconds=round(time.monotonic()-epoch_started, 2))
                if should_stop and epoch + 1 < epochs:
                    stop_reason = 'gleu_patience'
                    stage('early_stopping', epoch=epoch+1, patience=args.early_stopping_patience,
                          best_val_gleu=state['best_val_gleu'])
                    break
            if rank == 0:
                result = {**state, 'status': 'complete', 'mode': args.mode, 'gradient_norms': grad_report,
                          'stop_reason': stop_reason, 'planned_epochs': epochs,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
                if args.mode == 'overfit':
                    result['final_training_loss'] = state['last_training_loss']
                    result['overfit_passed'] = result['final_training_loss'] < .05
                if args.mode == 'train' and args.selection_metric == 'val_gleu':
                    best_pointer = json.loads((run_dir/'best_gleu.json').read_text())
                    best_meta = json.loads((run_dir/best_pointer['checkpoint']/'checkpoint.json').read_text())
                    ensure_json(ROOT/'blt_hf_checks/results'/f'{run_dir.name}_integrated_gleu.json',
                                {'status': 'complete', 'run_id': manifest['run_id'],
                                 'run_signature': signature, 'code_hash': identity['code_hash'],
                                 'dataset': args.dataset, 'epochs': epochs,
                                 'completed_epochs': state['epoch'], 'stop_reason': stop_reason,
                                 'early_stopping_patience': args.early_stopping_patience,
                                 'selection_metric': args.selection_metric,
                                 'validation_num_beams': args.validation_beams,
                                 'validation_max_new_bytes': args.validation_max_new_bytes,
                                 'validation_generation_backend': args.validation_generation_backend,
                                 'epoch_checkpoints': json.loads((run_dir/'epoch_checkpoints.json').read_text()),
                                 'best': {**best_pointer,
                                          'model_sha256': best_meta['files']['model.safetensors']}})
                write_json(run_dir/'completed.json', result)
                atomic_json(run_dir/'progress.json', result)
            if args.mode == 'overfit' and state['last_training_loss'] >= .05:
                return 1
            return 0
    finally:
        if world > 1 and dist.is_initialized(): dist.destroy_process_group()


def main(): return run(parser().parse_args())
if __name__ == '__main__': raise SystemExit(main())
