# CLAUDE.md — P1: HF Transformers 기반 BLT-GEC 재구축

> **2026-09-26 사용자 결정:** 생성 속도 개선을 위해
> `plan/P3a_generation_cache_20260925.md`의 증분 생성 작업에 착수한다.
> 이 파일의 P1 전용 no-cache 허용 범위와 P3a 보류 문구는 기존 학습·평가
> 기본 경로에만 적용한다. 새 backend는 승인된 실행안의 단계별 비교를
> 통과하기 전까지 기본 경로로 사용하지 않는다. Neuron 미접속 지시는 유지한다.

> 배치 위치: `PHDQ3/CLAUDE.md` (저장소 전체 P1 작업에 적용)
> 상세 계획: 저장소 루트 기준 `plan/P1.md` — 작업 착수 전 해당 Phase 절과
> 「작업 트랙 구조」·「실행 준비와 결과 채택 판정」을 먼저 읽는다.
> 노드 역할·환경·SLURM 실행 규칙은 `plan/P1.md` A-1과 이 파일을 기준으로 한다.
> `blt_hf_checks/`, `scripts/`, `tests/`를 작업할 때도 이 규칙을 명시적으로 참조한다.

## 프로젝트 개요

`reference_code/blt`(xFormers 커밋 고정) 의존을 제거하고 Hugging Face
Transformers의 정식 BLT 구현(`BltForCausalLM`)과 직접 변환본 B 위에
한국어 GEC 파인튜닝·생성·평가를 재구축한다. `itazap/blt-1b-hf`는 참고 artifact C다.
**2026-09-16 사용자 결정**: 변환 관련 검증은 가중치 보존과 attention mask 검사로
한정한다. 원본 runtime·hidden/logits 수치 동질성은 요구하지 않으며, 미확인을 이유로
학습 결과를 폐기하지 않는다. 구현 변경을 명시하고 한국어 GEC 성능을 평가한다.
학습·평가 유효성 테스트는 별도 필수 작업이며, 검증과 파인튜닝은 병행한다.

## 절대 규칙 (위반 금지)

1. **`blt_gec/`는 동결 상태다. 어떤 파일도 수정하지 않고, `blt_hf/`에서
   import하지도 않는다 (runtime 의존 0건).** metrics/M2 등 재사용이 필요한
   로직은 기존 동작을 fixture로 고정한 뒤 `blt_hf/`로 **독립 이식**하고
   (`blt_hf/metrics.py`, `blt_hf/m2_resumable.py`), 원본 파일 경로와 commit을
   기록하며, 동일 hypothesis fixture에 대한 legacy↔신규 GLEU/M2 parity
   테스트를 둔다.
2. **`blt_hf/` 패키지는 `reference_code/blt`를 import하지 않는다.** 원본 소스는
   mask 규칙·가중치 대응 확인을 위해 열람한다. 원본 실행 환경 복구·legacy dump·
   별도 원본 entropy forward·B↔C 비교는 P1 필수 작업이 아니다.
3. **구동 준비 후 파인튜닝은 별도 변환 검증 완료를 기다리지 않는다.** 변환 검증은
   A-7의 가중치 값/shape/dtype/coverage/strict loading 및 A-4의 실제 self/cross
   attention mask 검사다. 값과 허용 위치는 정확히 비교한다. label·패처·gradient·
   tiny overfit·생성·채점은 일반 실험 유효성 테스트로 유지한다.
   `weight_checks`, `attention_mask_checks`, `training_checks`, `evaluation_checks`를
   각각 `passed|failed|not_run`으로 기록하고 보고서를 해당 artifact·코드·설정에 연결한다.
   전체 동질성의 `pending/confirmed/not_confirmed` 게이트는 폐기했다. 과거 JSON은
   이력으로 보존하고 새 보고서를 추가한다. 실제 구현 결함은 영향 run을 구분해 수정하고
   기존 checkpoint·로그를 보존한다. 원본 수치 미확인만으로 결과를 폐기하지 않으며,
   낮은 GEC 성능도 보고한다. 성능 동등성을 비교 없이 주장하지 않는다.
   정본은 `plan/P1.md`의 「실행 준비와 결과 채택 판정」이다.
4. **site-packages의 transformers 코드를 직접 편집하지 않는다.** HF 구현의
   수정이 필요하면(A-6 얕은 cache 고장, A-4 마스크 미구현 등) `modeling_blt.py`를
   `blt_hf/patched/`로 복사하거나 `BltForCausalLM`을 서브클래싱해 자기
   네임스페이스에서 명시적으로 관리한다. 조건 세 가지가 모두 충족되어야 한다:
   (a) 원본 대비 diff를 NOTES.md에 기록, (b) 무수정 경로와의 parity 테스트
   동반, (c) 재현 스크립트 확보 후 upstream 이슈/PR 지향.
   **attention mask**: main은 local/entropy window=512 및 EOS 구간, global patch
   구간과 byte↔patch 연결을 검사한다. 모든 실제 계층의 마스크를 독립 규칙으로 검사하고
   pooling/global/decoder를 통한 미래 정보 유입도 확인한다. 원본 runtime은 필요 없다.
   현재 `attention_mode=osc`는 프로젝트의 window/EOS·패칭 적용 경로, `lre`는 HF
   기본 full-causal 경로라는 호환용 실행 식별자다. 이름이 동질성 인증을 뜻하지 않는다.
   현재 검증 범위는 eager/bf16/no-cache/무패딩이다. padding·cache는 명시적으로 거부한다.
   샘플당 한 문장 쌍을 사용하며 문서 packing은 하지 않는다. EOS self-attention 검사만으로
   hash n-gram·pooling을 포함한 전체 상태 격리를 주장하지 않는다.
5. **실행 조건이 다른 shard를 aggregate하지 않는다.** `blt_hf/manifest.py`의
   fingerprint(모델 ID/revision/checkpoint hash, transformers 버전, attn_impl,
   attention_mode, 모델 config·tokenizer hash, dataset/split, 전체 split의
   source/target/M2 hash, generation backend, use_cache, beam, batch,
   길이 정책, code commit) 완전 일치가 aggregate의 전제다.
   이는 HF 공식 클래스가 아니라 우리가 구현할 실행 manifest 규약이다.
   필수 필드 누락·불일치는 즉시 실패하도록 구현한다. `attention_mode`는
   실제 실행 설정이며 검사 통과 여부와 구분한다. 정의는 P1 D-1을 따른다.
6. **legacy 결과와 신규 결과를 같은 표에서 직접 비교하지 않는다.** legacy는
   full-causal fallback ablation으로만 병기한다.
7. **제공 데이터와 split은 그대로 사용한다.** 제3자가 제공한 validation/test를
   포함해 중복 제거·재분할·행 제외·정답 수정·평가 subset 대체를 하지 않는다.
   `data/Preprocessed/`와 `data/Raw/`의 원본 TSV/M2를 수정하지 않는다.
   본 평가는 canonical split의 M2를 사용한다. gold 파일과 이를 채점하는
   scorer 코드는 구분하며, scorer 확보·독립 이식은 P1 D-1에 따른다.

## 환경

- **접속·실행 담당 (2026-09-15 사용자 지시)**: 에이전트는 neuron에 접속하거나
  원격 명령·파일 전송·SLURM 제출을 수행하지 않는다. neuron용 코드와 명령은
  로컬에서 준비하고 사용자가 실행한다. itcerdo는 사용자 제공 `ssh.md`에 따라
  `ssh itcerdo`로 접속하며 `/home/itcmaster/projects/phdq3` 안에서만 작업한다.
- conda env: `phdq_blt_hf` — itcerdo 실측 Python 3.11.16,
  `transformers==5.16.1`, `torch==2.11.0+cu128`, xFormers 없음.
  `blt_hf/requirements.lock.txt`의 환경을 고정하며 neuron A100 구동은 사용자가 확인한다.
  neuron job에서 `module load cuda` 없이 wheel 번들 런타임을 사용한다.
  CUDA 마이그레이션은 별도 작업으로 분리한다.
- **노드 역할** (SLURM은 neuron에만 존재):
  - gsm/yellowstone(mac): 코드 작성·정적 테스트·결과 취합. CUDA 미설치.
  - itcerdo(5090, 일반 Linux): Phase A 구동·가중치·마스크 검사와
    2026-09-27 사용자 지시의 P3a 캐시 단건 BF16 추론·프로파일 전용.
    캐시 작업공간·인계 규칙은 `cache.md`를 따른다.
    **일반 shell 명령만 사용 — srun/sbatch 금지. optimizer step·tiny overfit·
    파인튜닝·full tuning 일절 금지.** 여기서 검증한 작동 코드를 neuron에서
    학습에 사용한다.
  - neuron(A100, SLURM): backward smoke, tiny overfit, 본 학습, 본 평가.
- BLT 변환 스크립트는 pip 패키지에 없다 — `blt_hf_checks/vendor/`의 vendored
  사본(main commit SHA 기록)을 사용한다.
- 노드별 환경과 검사 결과를 기록한다. GPU 간 logits/greedy 출력의 완전 일치를
  연구 결과의 채택 조건으로 두지 않는다. 원본 수치 동질성도 주장하지 않는다.
- 버전은 `blt_hf/requirements.lock.txt`로 고정. 환경 변경 시 lock 재생성 +
  manifest의 transformers_version 갱신.
- 모든 job 시작부에서 `python blt_hf_checks/check_env.py` 실행 (스크립트에 포함).
- attn_implementation은 Phase A-2에서 결정된 값으로 고정하고 이후 변경하지
  않는다 (현재 값은 NOTES.md 참조).

## 디렉토리 구조

```
blt_hf/            # 본 패키지: model, data_adapter, train, generation, eval,
                   #   metrics(독립 이식), manifest
blt_hf_checks/     # Phase A 검증 + 벤치 스크립트. fixtures/와 results/ 포함
scripts/train_blt_hf.sh, scripts/eval_blt_hf.sh
tests/test_hf_*.py
outputs/blt_hf/<dataset>/                              # 학습 체크포인트
outputs/blt_hf_eval/<dataset>/<split>/<ckpt>/<cond>/   # 평가 (생성 조건별 격리)
```

`blt_hf_checks/results/*.json`(cache_probe, gen_bench, m2_profile)은 P3 착수
판정의 근거 자료이므로 삭제·덮어쓰기 전에 git commit한다.

## 주요 명령

사용자 실행 순서와 실제 인자는 `blt_hf/NEURON.md`를 정본으로 한다.
현재 `train.py`, `eval.py`, `generation.py`, `metrics.py`, `m2_resumable.py`와
학습/생성/CPU 채점/제출 shell 스크립트가 구현되어 있다.

```bash
# neuron login shell에서 사용자만 실행. job header의 field는 nlp.
cd /scratch/r984a02/phdq3
sbatch --export=ALL,RUN_ID=native-smoke-01,DATASET_TYPE=native,NUM_GPUS=1,TRAIN_MODE=smoke scripts/train_blt_hf.sh
# smoke/overfit/DDP 확인 후 본 학습. 기존 run은 RESUME=<latest.json>을 명시.
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 --export=ALL,RUN_ID=native-v2-s0,DATASET_TYPE=native,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,SEED=0 scripts/train_blt_hf.sh
# 학습 완료 후 CKPT_PATH와 조건별 EVAL_DIR을 지정.
sbatch --export=ALL,CKPT_PATH=<불변-step-directory>,EVAL_DIR=<조건별-output-directory>,DATASET_TYPE=native scripts/eval_blt_hf.sh
sbatch --export=ALL,EVAL_DIR=<동일-output-directory>,DATASET_TYPE=native scripts/score_blt_hf.sh
```

- 스크립트는 `ssh.md`의 root·A100/H200 partition·GPU/CPU 비율·comment 형식을 검사한다.
- 본 학습은 A100 4GPU, 생성은 A100 1GPU가 기본이다. H200은 한 작업당 최대 2GPU만
  사용하며 1GPU 검사·생성을 우선한다. H200 4GPU 이상 작업은 제출하지 않는다.
- 체크포인트는 불변 directory이며 `latest.json`/`best.json`만 포인터로 갱신한다.
- 새 본 학습은 10 epoch와 5% warmup을 사용한다. epoch별 checkpoint를 전체 validation
  GLEU로 비교해 `best_gleu.json`을 만들고 그 checkpoint만 test 비교에 사용한다.
- 먼저 별도 native 2-epoch 통합 시험에서 A100 4GPU rank가 validation 생성을 분담하고
  GLEU로 checkpoint를 고른다. 시험 명령과 재개는 `blt_hf/NEURON.md`를 따른다.
- 실행 identity는 단계별 실행 파일 hash로 고정하고 Git commit/job/node는
  `provenance.jsonl`에 감사 정보로 기록한다. 로그만 추가한 commit은 resume을 막지 않는다.
- Lang-8은 union prefix를 검증해 `data/Preprocessed/lang8`에 별도로 만들며 기존 제공 파일은 변경하지 않는다.
- 원본·변환 artifact와 동일하게 main parameter/gradient/Adam 상태와 연산은 BF16,
  entropy는 BF16 고정, gradient checkpointing. dtype 불일치는 즉시 실패한다.
  rank당 무패딩 1개와 전역 supervised-token loss, DDP accumulation을 사용한다.
- smoke는 작은 모델의 backward/재개 검사 및 실제 B의 긴 train 입력 optimizer 검사다.
  neuron 외 환경에서 이를 실행하지 않는다. 아직 실제 A100 학습을 통과했다고 주장하지 않는다.
- 기존 scorer는 `~/Nextcloud/QLab/phdq`에서 확보했으며 `blt_hf/vendor/`로 독립 이식했다.
  원본 실험 위치를 runtime으로 참조하지 않는다. 파일 hash·출처·라이선스·변경 diff 보존.
- 평가는 전체 split만 합산하며 미완료 M2를 최종 점수로 발표하지 않는다.

## 작업 방식

- **테스트 먼저**: data_adapter, generation, manifest는 대응 테스트 파일이
  존재해야 하며, 수정 시 테스트를 먼저 갱신한다. batch 불변성 테스트
  (동일 문장, batch=1 vs 혼합 batch → 동일 출력)는 generation 변경마다 실행.
- **절단 금지**: source/SEP/target을 자르지 않는다. `truncation=False`,
  입력 한도 초과 시 sample ID·길이를 출력하고 실패한다. separator는 보존한다.
- **BOS/EOS 명시 삽입**: 본문은 `add_special_tokens=False`로 인코딩한다.
  학습열은 `[BOS]+source+SEP+target+[EOS]`, 생성 prompt는 `[BOS]+source+SEP`다.
  BOS/EOS 설정값을 자동 삽입의 증거로 삼지 않고 반환 IDs를 검사한다.
  labels는 BOS+source+SEP만 -100, target EOS 포함, 내부 shift는 1회다.
- **padding 방향 주의**: 생성은 left-padding이 표준이지만 BLT patcher와의
  상호작용이 검증 대상이다. 관련 코드를 만지면
  `tests/test_hf_generation.py`의 batch 불변성 테스트를 반드시 재실행.
- **결정 기록**: 코드 리딩 결과(HF patcher 위치, patch_lengths 입력 경로,
  마스크 의미), 판정 분기 결과, 우회 사유는 전부 `blt_hf/NOTES.md`에 날짜와
  함께 기록. 실행 명령·복구 절차는 `blt_hf/COMMANDS.md`.
- **커밋**: `[P1][mac|slurm] 설명` 형식. 검증 판정·후속 계획 결정 시 저장소 루트 `log.md` 갱신.

## 알려진 함정

- HF BLT는 편입 1년 미만 — `patch_lengths` 학습 경로, left-padding, use_cache
  에 미발견 버그 가능. 이상 동작은 먼저 최소 재현을 만들고 버전을 의심한다.
- `use_cache=False`가 문서 기준 기본 전제. cache 미동작 시 A-6의 깊이 판정을
  따른다: 얕은 고장은 `blt_hf/patched/`로 P1 안에서 수정(규칙 4의 3조건 준수),
  증분 상태 관리의 근본 부재는 P3a 스코프 — 이 프로젝트에서 설계하지 않는다.
- legacy 스택은 5090에서 실행되지 않는다(sm_120 미지원). legacy 관련
  스크립트를 5090에 제출하는 실수 방지 — 스크립트 시작부에 capability 검사.
- backend/dtype 차이로 greedy 출력이 달라질 수 있다. 현재 eager를 기준으로
  고정한다. 생성 batch 변경 시 실제 지원 경로의 동작을 검사하고 차이를 기록한다.
  원본 출력 문자열과의 완전 일치를 완료 조건으로 두지 않는다.
- V100(cas_v100*)은 bf16 미지원 — fp16 경로 검증 없이 V100에 제출하지 않는다.
- 환경 검사는 A100(sm_80), H100/H200(sm_90), itcerdo 검증용 RTX 5090(sm_120)을
  허용한다. Neuron job은 `ssh.md`에 확인된 A100/H200 partition만 사용한다.
