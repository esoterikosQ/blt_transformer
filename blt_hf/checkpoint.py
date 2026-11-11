"""Immutable checkpoint directories with mutable latest/best pointers."""
import json
import os
import uuid
from pathlib import Path
from .manifest import sha256_file, write_json
from .runtime import atomic_json


def resolve_checkpoint(path, *, verify_optimizer=True, verify_files=True):
    path = Path(path).resolve()
    if path.is_file() and path.suffix == '.json':
        pointer = json.loads(path.read_text())
        target = (path.parent / pointer['checkpoint']).resolve()
        if not target.is_relative_to(path.parent):
            raise ValueError('Checkpoint pointer escapes run directory')
        path = target
    metadata = json.loads((path / 'checkpoint.json').read_text())
    if set(metadata['files']) != {'model.safetensors', 'training.pt'}:
        raise ValueError('Incomplete checkpoint metadata')
    if verify_files:
        for name, value in metadata['files'].items():
            if name == 'training.pt' and not verify_optimizer:
                continue
            if name not in ('model.safetensors', 'training.pt') or sha256_file(path / name) != value:
                raise ValueError(f'Checkpoint file/hash mismatch: {name}')
    return path, metadata


def save_checkpoint(run_dir, model, optimizer, state, rng_states, *, best=False):
    import torch
    from safetensors.torch import save_file
    run_dir = Path(run_dir)
    name = f"step-{state['global_step']:08d}-{uuid.uuid4().hex[:8]}"
    staging = run_dir / ('.'+name)
    staging.mkdir()
    # CPU materialization avoids allocating a second checkpoint copy on the GPU.
    tensors = {key: tensor.detach().cpu().contiguous() for key, tensor in model.state_dict().items()}
    save_file(tensors, str(staging / 'model.safetensors'))
    del tensors
    torch.save({'optimizer': optimizer.state_dict(), 'rng_states': rng_states}, staging / 'training.pt')
    files = {n: sha256_file(staging / n) for n in ('model.safetensors', 'training.pt')}
    write_json(staging / 'checkpoint.json', {**state, 'files': files, 'format_version': 1})
    for p in staging.iterdir():
        with p.open('rb') as stream: os.fsync(stream.fileno())
    os.rename(staging, run_dir / name)
    pointer = {'checkpoint': name, 'global_step': state['global_step']}
    atomic_json(run_dir / 'latest.json', pointer)
    if best: atomic_json(run_dir / 'best.json', pointer)
    if state.get('next_batch') == 0 and state.get('epoch', 0) > 0:
        epochs_path = run_dir / 'epoch_checkpoints.json'
        epochs = json.loads(epochs_path.read_text()) if epochs_path.exists() else []
        entry = {'epoch':state['epoch'],'checkpoint':name,'global_step':state['global_step'],
                 'val_loss':state.get('last_val_loss')}
        if state.get('run_manifest', {}).get('selection_metric') == 'val_gleu':
            entry['val_gleu'] = state.get('last_val_gleu')
        prior = [item for item in epochs if item['epoch'] == entry['epoch']]
        if prior and prior[0] != entry:
            raise ValueError(f"Epoch checkpoint already recorded differently: {entry['epoch']}")
        if not prior:
            atomic_json(epochs_path, epochs + [entry])
    return name
