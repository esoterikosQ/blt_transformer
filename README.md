# blt_transformer

Meta의 BLT(Byte Latent Transformer)를 Hugging Face Transformers의 정식 구현
(`BltForCausalLM`) 위에서 실행하고, 한국어 문법·오탈자 교정(GEC)에 파인튜닝·생성·평가하는
연구 코드입니다.

이 프로젝트는 원본 BLT runtime과 hidden/logits 값이 같다고 주장하지 않습니다. 변환 검증은 가중치 보존과
attention mask 검사로 한정하고, 구현상 차이를 기록한 뒤 한국어 GEC 성능을 그대로 보고합니다.

## 1. 개요

| 항목 | 내용 |
|---|---|
| 모델 | `facebook/blt-1b` + 엔트로피 패처를 HF 형식으로 직접 변환한 artifact **B** |
| 참고 artifact | `itazap/blt-1b-hf`(**C**), 비교 대상일 뿐 필수 검증 대상은 아님 |
| 실행 환경 | Python 3.11.16, `torch==2.11.0+cu128`, `transformers==5.16.1`, xFormers 없음 |
| 고정 설정 | `attn_implementation=eager`, BF16, `use_cache=False`, 무패딩 |
| 과제 | 한국어 GEC 4종(native, korean_learner, lang8, union) |
| 지표 | 전체 split 기준 GLEU, M2 F0.5 |

배경: 기존 실험은 `reference_code/blt`(xFormers 커밋과 PyTorch nightly 고정)에 의존해 재현과
신규 GPU(sm_120) 구동이 어려웠습니다. 이 저장소는 그 의존을 제거하고 HF 구현 위에서 데이터 처리,
학습, 생성, 채점까지 다시 구축합니다. 이전 구현(`blt_gec/`)은 동결했으며 이 저장소에서 import하지 않습니다.

## 2. 수행한 작업

### 2.1 가중치 변환과 검증 (2026-09-15)

- 원본 체크포인트와 엔트로피 모델의 revision을 고정하고, upstream 변환 스크립트를 vendoring해
  artifact B를 만들었습니다. 출처와 hash는 [sources.lock.json](blt_hf_checks/manifests/sources.lock.json),
  [conversion_B_20260915.json](blt_hf_checks/manifests/conversion_B_20260915.json)에 있습니다.
- 원본 tensor 515개를 HF tensor 510개에 대응시켰습니다. 차이는 hash embedding 6개를 하나로 합친 데서 생깁니다.
  값·shape·dtype·coverage 일치 검사와 strict loading을 통과했습니다.
  ([weights_p1_validation_20260915.json](blt_hf_checks/results/weights_p1_validation_20260915.json))
- eager와 SDPA의 greedy 출력이 달라 attention backend를 eager로 고정했습니다.

### 2.2 Attention mask와 패칭 경로 (2026-09-15 ~ 16)

HF 기본 구현은 로컬·엔트로피 모델에 일반 full-causal mask를 사용합니다. 원본 설정의 마스크 규칙을
반영하기 위해 `modeling_blt.py`를 [blt_hf/patched/](blt_hf/patched/)로 복사해 프로젝트 안에서 관리합니다.
site-packages의 transformers는 수정하지 않았습니다.

- 로컬 인코더·디코더와 엔트로피 모델: 512 window와 EOS 구간 마스크
- 전역 트랜스포머: patch 인과성과 EOS 구간 분리
- 엔트로피 계산 순서·dtype, BLT의 next-token patch slot 처리
- 원본 대비 diff: [modeling_blt_osc.diff](blt_hf_checks/manifests/modeling_blt_osc.diff)

이 경로를 `attention_mode=osc`, HF 기본 full-causal 경로를 `lre`라고 부릅니다. 둘 다 실행 식별자이며
동등성 인증을 뜻하지 않습니다. 실제 B에서 attention 모듈 59개와 가상·zero-patch 모듈 45개의 마스크를
독립 규칙으로 검사했고, byte→patch 경로를 통한 미래 정보 유입도 확인했습니다.
([p1_masks_20260916.json](blt_hf_checks/results/p1_masks_20260916.json))
검증 범위는 eager·BF16·no-cache·무패딩이며, padding과 cache 입력은 코드에서 거부합니다.

### 2.3 데이터 형식

- 학습열: `[BOS] + source + SEP + target + [EOS]`. 본문은 `add_special_tokens=False`로 인코딩하고
  BOS/EOS는 명시적으로 넣습니다.
- labels: BOS·source·SEP는 `-100`, target과 EOS만 학습합니다. shift는 모델 내부에서 한 번만 합니다.
- 생성 prompt: `[BOS] + source + SEP`
- 절단·padding·문서 packing을 하지 않습니다. 샘플당 한 문장 쌍이며, 길이 한도를 넘으면 실패 처리합니다.
- 제공된 split은 수정·재분할하지 않습니다. Lang-8은 제공된 union split의 앞부분(learner·native 행 이전)을
  결정적으로 추출합니다. ([derive_lang8.py](blt_hf/derive_lang8.py))

| 데이터셋 | train | val | test |
|---|---:|---:|---:|
| native | 12,292 | 2,634 | 2,634 |
| korean_learner | 19,898 | 4,264 | 4,265 |
| lang8 | 76,692 | 16,434 | 16,434 |
| union | 108,882 | 23,332 | 23,333 |

전체 split 201,534행 가운데 2,048바이트를 넘는 시퀀스는 없습니다.
([neuron_923056_0_data.json](blt_hf_checks/results/neuron_923056_0_data.json))

### 2.4 학습 절차

- 학습 대상: 엔트로피 패처는 동결하고 로컬 인코더, hash embedding, 전역 트랜스포머, 로컬 디코더,
  `lm_head`를 모두 학습합니다. hash 함수는 고정이고, hash ID에 대응하는 임베딩 벡터를 학습합니다.
- 정밀도: 파라미터, gradient, Adam 상태와 연산을 모두 BF16으로 처리하고 gradient checkpointing을 씁니다.
  dtype이 맞지 않으면 즉시 실패합니다.
- 배치: rank당 무패딩 1문장, DDP gradient accumulation으로 effective batch 32를 맞춥니다.
  loss는 전역 supervised-token 평균입니다.
- 최적화: AdamW(β=(0.9, 0.95)), 기본 학습률 1e-5, warmup 5%
- 일정: 최대 10 epoch. 매 epoch 끝에 전체 validation으로 GLEU를 계산하고, 3 epoch 연속 개선이
  없으면(동점은 미개선) 종료합니다. 가장 높은 validation GLEU를 낸 checkpoint만 test에 사용합니다.
- 체크포인트는 불변 디렉터리로 저장하고 `latest.json`, `best_gleu.json`, `completed.json` 포인터만 갱신합니다.
  시간 한도에 걸리면 exit 75로 끝나며, 같은 설정으로 재제출하면 이어서 학습합니다.

### 2.5 생성과 증분 계산 실험 (P3a)

| backend | 용도 | 내용 |
|---|---|---|
| `hf-generate-exact-length-unpadded-v1` | test 생성 | HF `generate()`, KV cache 없음, beam 1 |
| `global-prefix-guarded-greedy-v1` | 학습 중 validation | 닫힌 patch의 전역 계산을 재사용하고, 1위와 2위 byte logit 차이가 1.0 이하이면 전체 forward로 다시 계산 |

HF BLT의 `use_cache=True`는 패처 층에 cache를 전달하지 않아 실제로 동작하지 않습니다. 대신 전역 prefix를
재사용하는 guarded 경로를 만들었습니다. native validation 2,634문장에서는 HF 경로와 출력이 모두 일치했고,
생성 시간은 9,074초에서 6,907초로 줄었습니다(1.31배).
([p3a_native_full_guarded_11.json](blt_hf_checks/results/p3a_native_full_guarded_11.json),
[DESIGN.md](blt_hf/cache/DESIGN.md), [P3a 계획](plan/P3a_generation_cache_20260925.md))
이 일치는 native·beam 1·batch 1 조건의 실험 결과입니다. 일반적인 수치 보장이 아니며,
다른 데이터셋이나 beam 4에서는 확인하지 않았습니다. 캐시 개발은 별도 프로젝트로 분리했습니다.

### 2.6 평가와 실행 manifest

- 생성은 shard 단위로 재개할 수 있습니다. 모델·checkpoint hash, transformers 버전, attention 설정,
  config·tokenizer hash, 데이터·M2 hash, generation backend, beam, batch, 길이 정책, 코드 identity로 구성한
  fingerprint가 모두 같아야 shard를 합산합니다. ([manifest.py](blt_hf/manifest.py))
- GLEU: 단일 참조 corpus GLEU(order 4), 기존 실험의 소수점 6자리 반올림 ×100 규칙을 따릅니다.
- M2: β=0.5, max unchanged words 2. 문장 단위로 journal을 남겨 재개할 수 있고, 미완료 상태에서는
  corpus 점수를 발표하지 않습니다. ([m2_resumable.py](blt_hf/m2_resumable.py))
- 기존 scorer는 독립 이식했고, legacy와 신규 구현의 GLEU/M2 parity를 테스트합니다. 출처와 변경 diff는
  [blt_hf/vendor/provenance.json](blt_hf/vendor/provenance.json)에 있습니다.

## 3. 결과 (seed 0, test, beam 1, HF 생성)

| 데이터셋 | 선택 checkpoint | GLEU | M2 P | M2 R | M2 F0.5 |
|---|---|---:|---:|---:|---:|
| native | epoch 6 (step 2,310) | 65.6366 | 0.7601 | 0.5968 | 0.7207 |
| korean_learner | epoch 8 (step 4,976) | 37.4558 | 0.3801 | 0.1846 | 0.3137 |
| lang8 | epoch 8 (step 19,176) | 25.3489 | 0.2803 | 0.1091 | 0.2133 |
| union | epoch 8 (step 27,224) | 31.6857 | – | – | 0.2806 |

- union 점수는 동기화한 생성 결과를 로컬 CPU에서 같은 scorer로 계산한 값입니다. Neuron 공식 채점 기록은 아직 없습니다.
- 앞선 3 epoch 학습 run의 test 점수는 native GLEU 55.3096 / F0.5 0.6604, learner GLEU 33.2126 / F0.5 0.2648입니다.
  최대 10 epoch와 GLEU 기준 checkpoint 선택으로 바꾼 뒤 점수가 올랐습니다.

epoch별 validation GLEU(굵은 값이 선택 checkpoint):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 53.31 | 60.14 | 62.29 | 64.25 | 65.07 | **66.08** | 65.94 | 65.94 | 65.82 | – |
| learner | 33.12 | 35.87 | 37.15 | 38.03 | 38.54 | 38.48 | 38.83 | **38.87** | 38.74 | 38.79 |
| lang8 | 22.84 | 23.71 | 24.44 | 24.11 | 24.53 | 24.74 | 24.78 | **24.89** | 24.88 | 24.86 |
| union | 27.82 | 29.30 | 30.87 | 30.85 | 31.30 | 31.41 | 31.58 | **31.69** | 31.65 | 31.66 |

출처: `blt_hf_checks/results/*_integrated_gleu.json`

### 해석과 한계

- 같은 데이터로 한국어 모델(예: Kanana 2.1B)을 파인튜닝한 별도 실험의 중간 점수는 native 84,
  learner 59, lang8 38, union 48 이상이었습니다. BLT는 네 데이터셋 모두에서 크게 뒤처집니다.
  이 비교는 이 저장소에서 통제한 벤치마크가 아닙니다.
- 후반 epoch의 개선 폭이 작아, 학습량만 늘려서는 격차를 줄이기 어려워 보입니다. 사전학습 데이터에 한국어가
  적은 점이 유력한 원인 후보지만 확정하지는 않았습니다. 다음 실험 후보는 한국어 추가 사전학습입니다.
- 엔트로피 패처의 BF16 엔트로피가 임계값 근처에 있으면 입력 길이에 따라 과거 patch 경계가 바뀔 수 있습니다.
  혼합 입력 접두부 36개 중 5개에서 경계가 바뀌었습니다.
  ([p3a_patch_causality_20260926_v2.json](blt_hf_checks/results/p3a_patch_causality_20260926_v2.json))
- 4GPU validation 결과를 모으는 `all_gather_object`에서 10분 NCCL timeout이 한 번 발생했습니다.
  rank 사이의 생성 시간 차이가 크면 다시 생길 수 있습니다.

### 소요 시간

- native, A100 4GPU: epoch당 약 47분(파라미터 학습 약 16분, validation 생성·GLEU 약 29분)
- union, A100 4GPU: epoch당 약 6.4시간
- 생성, A100 1GPU, native 2,634문장: HF 경로 문장당 3.45초, guarded 경로 2.62초
- 1바이트 생성 단계별 시간(A100, 전체 재계산 6회 평균): 패처 10.95ms, 로컬 인코더 1.52ms,
  전역 트랜스포머 17.41ms, 로컬 디코더 10.69ms, 합계 43.51ms
  ([p3a_native_cache_spike_diagnostic_04.json](blt_hf_checks/results/p3a_native_cache_spike_diagnostic_04.json))

## 4. 디렉터리 구조

```
blt_hf/                  본 패키지
  model.py               모델 로딩·설정 고정
  attention.py           OSC 마스크, 지원하지 않는 입력(padding·cache) 거부
  patching.py            엔트로피 기반 patch 분할
  data_adapter.py        GEC 입력·label 구성
  train.py, training.py  학습 CLI, 일정·DDP·재개
  generation.py          HF 생성과 guarded 생성
  eval.py, evaluation.py shard 생성·합산·채점
  integrated_validation.py, select_best.py   validation GLEU 기반 checkpoint 선택
  manifest.py, checkpoint.py, runtime.py     실행 fingerprint, 불변 checkpoint, 실행 기록
  metrics.py, m2_resumable.py, score_local.py   GLEU·M2 채점(독립 이식)
  derive_lang8.py        Lang-8 split 추출
  patched/               프로젝트 관리 modeling_blt.py
  cache/                 전역 prefix 재사용, patch 경계 판정
  vendor/                GLEU scorer 이식본과 출처 기록
  NEURON.md, COMMANDS.md, NOTES.md   실행 안내, 명령·복구 절차, 결정 기록
blt_hf_checks/           변환·마스크·패칭·생성 검사, 벤치마크, 진단
  manifests/             원본 출처, 변환 기록, diff
  results/               검사·벤치마크 결과 JSON
  vendor/                vendored 변환 스크립트
scripts/                 SLURM 학습·생성·채점, 검사 실행 스크립트
tests/                   단위 테스트(test_hf_*.py)
plan/                    P1 계획, P3a 생성 캐시 계획
```

## 5. 실행

학습·생성·채점은 SLURM 클러스터(A100/H200)에서 실행합니다. 실제 인자와 재개 절차는
[blt_hf/NEURON.md](blt_hf/NEURON.md)를 따릅니다. 구현 결정과 검증 판정은 [blt_hf/NOTES.md](blt_hf/NOTES.md),
작업 경과는 [log.md](log.md), 전체 계획은 [plan/P1.md](plan/P1.md)에 있습니다.

```bash
# 단위 테스트
python -m unittest discover -s tests -p 'test_hf_*.py' -v
```

Torch, 데이터, 비공개 fixture, M2 사본이 없으면 해당 테스트는 skip됩니다. skip을 학습·평가 검증 통과로
해석하면 안 됩니다.

## 6. 저장소에 포함하지 않는 파일

- `data/`, 모델 가중치, 학습 checkpoint, `artifacts/`, 설치 환경, 접속·인증 정보
- 생성·채점 결과(`outputs/`)와 SLURM 로그(`runtimelog/`)
- 데이터셋 발췌 fixture, Meta 원본 코드 검토 사본, 라이선스 표기가 상충하는 M2 scorer 사본

학습에는 원본 TSV split 9개와 val/test M2 6개, 변환본 B가 필요합니다. Git clone만으로는 실행 준비가 끝나지
않습니다. 복원 대상과 hash는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다.
기존 검증 JSON에 적힌 hash는 당시 파일의 기록이며, 지금 공개된 파일 전체가 그 실행을 증명한다는 뜻은 아닙니다.
외부 코드는 각각의 라이선스를 따르며, 저장소 전체에 새로운 일괄 라이선스를 부여하지 않습니다.
