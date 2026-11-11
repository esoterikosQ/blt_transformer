"""Pure sharding and complete-corpus publication contracts."""
import json
import os
import tempfile
from pathlib import Path
from .manifest import sha256_file


def shard_bounds(count, shards, shard):
    if count < 1 or not 1 <= shards <= count or not 0 <= shard < shards:
        raise ValueError('Invalid shard count/id')
    return count*shard//shards, count*(shard+1)//shards


def collect_records(root, manifest):
    root = Path(root)
    records = {}
    expected_dirs = {f'{i:04d}' for i in range(manifest['shard_count'])}
    if {p.name for p in (root/'shards').iterdir() if p.is_dir()} != expected_dirs:
        raise ValueError('Missing or extra shard directories')
    for shard in range(manifest['shard_count']):
        directory = root/'shards'/f'{shard:04d}'
        a,b=shard_bounds(manifest['sample_count'],manifest['shard_count'],shard)
        if not (directory/'complete.json').exists(): raise ValueError(f'Incomplete shard {shard}')
        status=json.loads((directory/'complete.json').read_text())
        if any(status.get(k)!=v for k,v in dict(fingerprint=manifest['fingerprint'],start=a,end=b,shard_id=shard).items()):
            raise ValueError('Shard identity/range mismatch')
        paths=sorted((directory/'batches').glob('*.json'))
        if status.get('batch_hashes')!={p.name:sha256_file(p) for p in paths}:
            raise ValueError('Completed shard prediction files changed')
        seen=set()
        for path in sorted((directory/'batches').glob('*.json')):
            batch=json.loads(path.read_text())
            if batch['fingerprint']!=manifest['fingerprint']: raise ValueError('Mixed execution fingerprint')
            for record in batch['records']:
                index=record['index']
                if not a<=index<b or index in records: raise ValueError('Duplicate/out-of-range prediction')
                records[index]=record; seen.add(index)
        if seen!=set(range(a,b)): raise ValueError('Prediction holes in completed shard')
    if set(records)!=set(range(manifest['sample_count'])): raise ValueError('Incomplete corpus')
    return [records[i] for i in range(manifest['sample_count'])]


def publish_lines(path, lines):
    path=Path(path); text=''.join(line+'\n' for line in lines)
    if any('\n' in line or '\r' in line for line in lines): raise ValueError('Multiline scorer input')
    if path.exists():
        if path.read_text(encoding='utf-8')!=text: raise ValueError('Existing text output differs')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as out:
            out.write(text);out.flush();os.fsync(out.fileno())
        os.link(tmp,path)
    finally: os.unlink(tmp)
