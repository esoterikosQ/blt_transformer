"""Audit complete native validation outputs from HF and global-prefix generation."""

import argparse
import json
import math
from pathlib import Path

from blt_hf.evaluation import collect_records
from blt_hf.generation import HF_BACKEND, GLOBAL_PREFIX_BACKENDS, GLOBAL_PREFIX_GUARDED_BACKEND
from blt_hf.manifest import fingerprint, sha256_file, sha256_json, write_json
from blt_hf.runtime import ROOT, NEURON_ROOT, local_path, require_neuron_job


MATCH_FIELDS = (
    'dataset', 'split', 'sample_count', 'checkpoint_hash', 'checkpoint_step',
    'source_hash', 'target_hash', 'm2_hash', 'tsv_hash', 'max_new_bytes',
    'max_sequence_bytes', 'num_beams', 'batch_size', 'length_penalty',
    'model_config_hash', 'tokenizer_hash', 'training_run_signature',
    'training_run_id', 'training_checks', 'conversion_checks',
    'model_id', 'model_revision', 'transformers_version', 'torch_version',
    'attn_implementation', 'attention_mode', 'inference_dtype', 'decode_policy',
    'scorer_hash', 'shard_count', 'use_cache',
)
GENERATION_CODE_PATHS = frozenset({
    'blt_hf/generation.py', 'blt_hf/eval.py', 'blt_hf/cache/global_reuse.py',
})
OUTPUT_FIELDS = (
    'source', 'token_ids', 'eos_reached', 'budget_exhausted',
    'valid_utf8', 'valid_token_ids', 'raw_text', 'text',
)


def validate_manifests(reference, candidate):
    if reference.get('generation_backend') != HF_BACKEND:
        raise ValueError('Reference must use the HF no-cache backend')
    if candidate.get('generation_backend') not in GLOBAL_PREFIX_BACKENDS:
        raise ValueError('Candidate must use the global-prefix backend')
    if reference.get('prefix_reuse') is not False or candidate.get('prefix_reuse') is not True:
        raise ValueError('prefix_reuse identity does not match backend')
    if reference.get('decoder_kv_reuse') is not False or candidate.get('decoder_kv_reuse') is not False:
        raise ValueError('decoder_kv_reuse must be false for both runs')
    for field in MATCH_FIELDS:
        if field not in reference or field not in candidate:
            raise ValueError(f'Missing comparison field: {field}')
        if reference.get(field) != candidate.get(field):
            raise ValueError(f'Runs differ in {field}')
    reference_files, candidate_files = reference.get('code_files'), candidate.get('code_files')
    if not isinstance(reference_files, dict) or not isinstance(candidate_files, dict):
        raise ValueError('Both runs must record code_files')
    if reference_files.keys() != candidate_files.keys():
        raise ValueError('Runs differ in code_files paths')
    for path in reference_files:
        if path not in GENERATION_CODE_PATHS and reference_files[path] != candidate_files[path]:
            raise ValueError(f'Runs differ in non-generation code file: {path}')
    if (reference['dataset'], reference['split'], reference['num_beams'],
            reference['batch_size'], reference['use_cache']) != ('native', 'val', 1, 1, False):
        raise ValueError('Expected native validation, beam 1, batch 1, HF use_cache=False')


def compare_records(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError('Different numbers of completed records')
    mismatches = []
    for left, right in zip(reference, candidate):
        if left['index'] != right['index']:
            raise ValueError('Prediction order or index differs')
        fields = [field for field in OUTPUT_FIELDS if left.get(field) != right.get(field)]
        if fields:
            mismatches.append({'index': left['index'], 'fields': fields})
    return mismatches


def verified_manifest(directory):
    manifest = json.loads((directory / 'run.json').read_text())
    payload = {key: value for key, value in manifest.items() if key != 'fingerprint'}
    if manifest['fingerprint'] != sha256_json({'required': fingerprint(payload), 'complete': payload}):
        raise ValueError(f'Corrupted run manifest: {directory}')
    return manifest


def batch_seconds(directory, manifest):
    durations = []
    for shard in range(manifest['shard_count']):
        for path in sorted((directory / 'shards' / f'{shard:04d}' / 'batches').glob('*.json')):
            batch = json.loads(path.read_text())
            if batch['fingerprint'] != manifest['fingerprint'] or len(batch['records']) != 1:
                raise ValueError(f'Mixed fingerprint or batch size: {path}')
            seconds = batch['seconds']
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f'Invalid generation time: {path}')
            durations.append(seconds)
    if len(durations) != manifest['sample_count']:
        raise ValueError(f'Incomplete batch timing: {directory}')
    return durations


def percentile(values, percent):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--candidate-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if ROOT.resolve() == NEURON_ROOT.resolve():
        require_neuron_job(gpu=False)
    reference_dir = local_path(args.reference_dir)
    candidate_dir = local_path(args.candidate_dir)
    output = local_path(args.output)
    if reference_dir == candidate_dir or output.exists():
        raise ValueError('Distinct evaluation directories and a new report path are required')
    reference_manifest = verified_manifest(reference_dir)
    candidate_manifest = verified_manifest(candidate_dir)
    validate_manifests(reference_manifest, candidate_manifest)
    for path, expected in candidate_manifest['code_files'].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f'Candidate code file changed since generation: {path}')
    reference_records = collect_records(reference_dir, reference_manifest)
    candidate_records = collect_records(candidate_dir, candidate_manifest)
    mismatches = compare_records(reference_records, candidate_records)
    reference_times = batch_seconds(reference_dir, reference_manifest)
    candidate_times = batch_seconds(candidate_dir, candidate_manifest)
    reference_total, candidate_total = sum(reference_times), sum(candidate_times)
    report = {
        'status': 'complete' if not mismatches else 'mismatch',
        'scope': 'all native validation beam-1 generated records, no scoring',
        'reference_dir': str(reference_dir.relative_to(ROOT)),
        'candidate_dir': str(candidate_dir.relative_to(ROOT)),
        'reference_fingerprint': reference_manifest['fingerprint'],
        'candidate_fingerprint': candidate_manifest['fingerprint'],
        'checkpoint_hash': reference_manifest['checkpoint_hash'],
        'reference_code_hash': reference_manifest['code_hash'],
        'candidate_code_hash': candidate_manifest['code_hash'],
        'sample_count': len(reference_records),
        'mismatches': mismatches,
        'exact_output_parity': 'passed' if not mismatches else 'failed',
        'reference_generation_seconds': reference_total,
        'cached_generation_seconds': candidate_total,
        'generation_speedup': reference_total / candidate_total if candidate_total else None,
        'reference_sentence_p50_seconds': percentile(reference_times, .50),
        'reference_sentence_p95_seconds': percentile(reference_times, .95),
        'cached_sentence_p50_seconds': percentile(candidate_times, .50),
        'cached_sentence_p95_seconds': percentile(candidate_times, .95),
    }
    if candidate_manifest['generation_backend'] == GLOBAL_PREFIX_GUARDED_BACKEND:
        report['cache_preliminary_skips'] = sum(record['cache_preliminary_skips'] for record in candidate_records)
        report['cache_guard_refreshes'] = sum(record['cache_guard_refreshes'] for record in candidate_records)
    write_json(output, report)
    print(json.dumps({key: report[key] for key in
                      ('status', 'sample_count', 'exact_output_parity', 'generation_speedup',
                       'reference_generation_seconds', 'cached_generation_seconds')},
                     ensure_ascii=False), flush=True)
    return 0 if not mismatches else 1


if __name__ == '__main__':
    raise SystemExit(main())
