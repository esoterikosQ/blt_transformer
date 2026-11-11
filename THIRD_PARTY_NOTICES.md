# Third-party source and publication scope

Publication review: 2026-09-16. Local excluded files are retained; exclusions apply to Git.
This document records the source notices and the chosen publication scope, not an assertion
that every upstream component has the same license.

## Included source

### Hugging Face Transformers — Apache-2.0

- `blt_hf/patched/modeling_blt.py`: modified from Transformers 5.16.1; original Hugging Face
  copyright and Apache notice remain in the file. Project modifications from 2026-09-15
  implement the documented attention masks, entropy arithmetic and patch alignment.
  The change record is `blt_hf_checks/manifests/modeling_blt_osc.diff`.
- `blt_hf_checks/vendor/upstream_convert_blt_weights_to_hf.py`: upstream converter at commit
  `2cba19507be799b7bef247ca6c1c4708bf881b5b`.
- `blt_hf_checks/vendor/convert_blt_weights_to_hf.py`: project-modified converter,
  changed 2026-09-15 for pinned local artifacts, mapping validation and conversion reporting.
  The change record is `blt_hf_checks/manifests/vendor.diff`.
- Full license: [LICENSE.transformers](blt_hf_checks/vendor/LICENSE.transformers).
- Upstream: <https://github.com/huggingface/transformers>.

These are modified project copies, not changes to the installed Transformers package.
Recorded source hashes are preserved in `blt_hf_checks/manifests/`.

### GLEU — source-embedded Modified MIT notice

`blt_hf/vendor/gleu.py` and `gleumodule_legacy.py` retain the full source-embedded
Modified MIT permission notice, Copyright (c) 2022 Soyoung Yoon, and the original
Courtney Napoles attribution where present. They were copied from the previous experiment
without changing their bodies. The legacy wrapper is kept for reference only.

Source: <https://github.com/soyoung97/Standard_Korean_GEC/tree/main/metric>.
Exact previous-project source hashes and destination hashes are recorded in
`blt_hf/vendor/provenance.json`. Dataset permissions are separate from these code notices.

## Excluded material and local prerequisites

### Datasets, model artifacts and access information

- `data/` in its entirety, including raw/preprocessed train, validation, test and M2 files.
- Corpus-derived `tests/fixtures/gleu_legacy.json` and the two
  `blt_hf_checks/results/patches_p1_osc_v{4,5}_20260915.json` reports.
- Model weights, converted artifacts, checkpoints, generated outputs, logs, dependency
  installations, caches and release archives under `artifacts/`/`outputs/`.
- `ssh.md`, environment secrets, HF access files and private keys.

Obtain data and Meta weights through their respective authorized distribution channels.
Dataset access and restrictions are described by the dataset authors at
<https://github.com/soyoung97/Standard_Korean_GEC#1-how-to-get-data>.
Model source identities are recorded in `blt_hf_checks/manifests/sources.lock.json`.
Do not use this repository as a redistribution channel for those artifacts.

### Meta BLT reference snapshots

`blt_hf_checks/vendor/model_sources/` is omitted. Its Meta source snapshot carries
CC BY-NC 4.0; the redundant HF inspection copies in that directory are also omitted.
Source references and hashes remain in `model_source_review.json` and `entropy_source_review.json`.
These reference snapshots are not imported by the training or evaluation runtime.

Source/license: <https://github.com/facebookresearch/blt/blob/main/LICENSE>.
The exact reviewed revision is recorded in the manifests; do not substitute current `main`
when reproducing an old review.

### NUS M2 scorer and example fixtures

The local M2 files state GPL version 3 or later in their headers, while the accompanying
`LICENSE.md` contains GPL version 2. The upstream repository has the same discrepancy:
<https://github.com/nusnlp/m2scorer>. We do not resolve that conflict by relabeling the code.
The M2 source, copied examples and source-containing diffs are excluded from this publication.
The independently written `blt_hf/m2_resumable.py` orchestration remains included.

For the existing authorized experiment, restore these **already prepared local** files
from the project workspace to the same relative paths on the execution machine:

```text
blt_hf/vendor/m2/__init__.py
blt_hf/vendor/m2/levenshtein.py
blt_hf/vendor/m2/util.py
blt_hf/vendor/m2/m2scorer.py
blt_hf/vendor/m2/LICENSE.md
```

Verify the restored files against `target_sha256` in `blt_hf/vendor/provenance.json`.
Do not silently replace the prior Python 3 scorer with the upstream Python 2 implementation
or a different scorer: prior scores and resume fingerprints depend on the exact implementation.
New recipients must resolve applicable distribution/use permissions with the upstream authors
before redistributing these files. Optional local regression fixtures go under
`tests/fixtures/m2/`; corpus-derived GLEU fixtures remain private.

Because generation records the scorer hash, restore M2 before generation as well as before
CPU aggregation. The Neuron transfer archive created for the user's existing experiment is
a local artifact; it is not the public Git distribution and is not committed.
