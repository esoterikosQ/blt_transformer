# P1 명령 및 현재 실행 범위

## 2026-09-26 — P3a 캐시 시제품 검사

itcerdo의 추론 전용 단계 0·1 보고서는
`blt_hf_checks/results/p3a_patch_causality_20260926_v2.json`,
`p3a_state_stability_repeat_20260926.json`,
`p3a_forward_profile_20260926.json`,
`p3a_reuse_sensitivity_20260926.json`,
`p3a_global_skip_bench_20260926.json`,
`p3a_global_decoder_nocopy_20260926.json`이다. 같은 출력 경로로 재실행하지
않는다. 해석과 제약은 `blt_hf/cache/DESIGN.md`에 있다.

아래는 사용자가 Neuron A100 1GPU에서 실행을 완료한 첫 native probe의
명령 이력이다. `BENCH_OUTPUT`의 기존 경로로 재실행하지 않는다. 이 probe는
학습이나 본 평가 backend를 변경하지 않았다.

```bash
cd /scratch/r984a02/phdq3
CKPT_PATH=outputs/blt_hf/native/native-main-01/best.json
BENCH_OUTPUT=blt_hf_checks/results/p3a_native_cache_probe_01.json
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=00:30:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",BENCH_OUTPUT="$BENCH_OUTPUT",DATASET_TYPE=native,SAMPLES=12,STEPS=32,REUSE_DECODER=0 \
  scripts/bench_blt_cache.sh
```

결과의 `probe_token_parity`, 각 문장의 `mismatches`, `global_skips`,
`timing_excluding_initial`을 본다. `probe_token_parity=passed`는 선택된
문장·step에만 해당하며 전체 validation 동등성을 뜻하지 않는다. 이 시제품은
batch 1·빔 1 전용이다. 에이전트는 Neuron에 접속·전송·제출하지 않는다.
완료된 `01.json`/`02.json` 및 선택형 decoder KV의 `03_decoder.json`
진단 명령은 `NEURON.md`의 P3a 절에 실행 이력으로 남긴다. 기존 보고서를
덮어쓰지 않는다. 03은 다음 ID 384/384 일치·forward 1.377배였다.
완료한 두 지연의 반복 진단 명령은 `NEURON.md`에 이력으로 남긴다. 적용을 배제한 이전
판단은 철회했다. 선택형 `global-prefix-greedy-v1` backend와 완성 생성 05
시험 명령은 `NEURON.md`의 P3a 절을 따른다. decoder KV는 이 backend에서
꺼져 있고 기존 no-cache가 기본이다.
05 완성 생성은 12/12 출력 일치·1.467배였으며, 전체 native validation의
별도 EVAL_DIR 두 개와 CPU 전수 비교 명령도 같은 절에 있다.
06 전수 비교는 32/2,634건의 출력 불일치·1.371배였다. 첫 분기 07 진단
명령과 기존 06 보고서 보존 규칙은 `NEURON.md`를 따른다.
08은 32/32에서 HF식 `logits_to_keep=1`이 기준 ID를 복구했다.
projection만 바꾼 선택형 v2의 44문장 완성 생성 09 명령도 `NEURON.md`에 있다.
09는 기존 불일치 12건을 해결하고 20건을 남겼다. 별도 guarded 경로의
44문장 10 시험 명령과 한계도 `NEURON.md`에 있다.
10은 44/44에서 HF 출력을 재현했다. 전체 native validation 생성·시간
비교 11의 제출·재개·CPU 비교 명령은 `NEURON.md`를 따른다.
11은 2,634/2,634 출력 일치와 순수 생성 1.314배 개선을 확인했다.
새 guarded 통합 GLEU 최대 10-epoch·patience 3 run·재개 명령은 `NEURON.md`의
"11 전수 비교 결과와 guarded 통합 GLEU 학습" 절을 따른다.

## 2026-09-26 — P3a 패치 경계 인과성 검사

단계 0 실행안은 `plan/P3a_generation_cache_20260925.md`, 상태 수명 가설은
`blt_hf/cache/DESIGN.md`에 기록한다. 실제 1B의 짧은 forward 검사만
itcerdo에서 수행한다. 기존 보고서 경로를 재사용하지 않는다.

```bash
ssh itcerdo
cd /home/itcmaster/projects/phdq3
conda activate phdq_blt_hf
python -m unittest tests.test_hf_cache_causality -v
python -m blt_hf_checks.check_patch_causality \
  --output blt_hf_checks/results/p3a_patch_causality_20260926.json
```

보고서의 `status`, `boundary_failures`, `max_abs_entropy_difference`를 확인한
뒤에만 encoder/global/decoder 상태 재사용 범위를 정한다. 이 명령은 학습이나
Neuron job을 제출하지 않는다.

## 2026-09-24 — native 2-epoch 통합 GLEU 시험

Neuron에서는 사용자가 `NEURON.md`의 "native 2-epoch 통합 GLEU 시험" 명령을
실행한다. `native-gleu2-b1-s0`은 A100 4GPU에서 두 epoch를 학습하고 각 epoch마다
동일 4GPU로 전체 validation 생성·corpus GLEU를 계산한다. 최종 선택 기준은
validation GLEU이며 loss는 진단값이다. 에이전트는 Neuron에 접속하거나 job을
제출하지 않는다.

## 2026-09-23 — 기존 learner 채점 후 새 사이클

새 코드를 Neuron에 반영하기 전에 기존 learner beam1 생성 결과를 기존 scorer로 채점한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
EVAL_DIR=outputs/blt_hf_eval/korean_learner/test/learner-main-01-beam1
sbatch --export=ALL,CONDA_ENV=phdq_blt_hf,EVAL_DIR="$EVAL_DIR",DATASET_TYPE=korean_learner,M2_WORKERS=8 \
  scripts/score_blt_hf.sh
```

새 코드 배포 후 Lang-8 파생 검사를 로그인 노드에서 먼저 실행할 수 있다. 출력은
`data/Preprocessed/lang8`이며 기존 제공 데이터 파일은 변경하지 않는다.

```bash
conda activate phdq_blt_hf
python -m blt_hf.derive_lang8
```

새 seed 0 학습은 A100 4GPU에서 시작한다. 기존 union run을 재사용하지 않는다.

```bash
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 --time=00:30:00 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=union-v2-s0,DATASET_TYPE=union,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,MAX_STEPS=1,MAX_SECONDS=1200 \
  scripts/train_blt_hf.sh
```

최초 1 step이 성공하면 같은 설정과 RUN_ID에 다음 인자를 넣어 다시 1 step을 실행한다.
두 번째 job에서 재개 stage와 global step 2를 확인한 뒤 `MAX_STEPS`를 빼고 계속한다.

```bash
RESUME=outputs/blt_hf/union/union-v2-s0/latest.json
```

전체 명령과 epoch별 validation GLEU 선택 절차는 `NEURON.md`를 정본으로 한다.

## Neuron 학습·성능 평가 — 코드 준비 완료

실제 사용자 실행 명령·SLURM 정책·재개·생성 shard·CPU 채점 절차는 `NEURON.md`를 따른다.
`submit_blt_hf.sh smoke → overfit/2GPU smoke → train → eval → score` 순서다.
Neuron에는 에이전트가 접속하지 않았으며 A100 backward/메모리 실측은 사용자 실행 대기다.
현재 HF 단위 테스트와 itcerdo의 forward-only 확인은 NOTES.md 최신 절에 기록했다.

## 2026-09-16 가중치·마스크 검사 — 완료

itcerdo tmux `phdq3-p1:masks-0916`에서 `p1_masks_20260916` 실행, exit_code=0.
현재 실행 중인 검사가 아니라 완료 후 shell 대기 상태다. 아래 2026-09-15 기록의
전체 동질성 대기 정책은 폐기했고, 가중치·mask와 학습/평가 상태를 분리한다.

```bash
ssh itcerdo
cd /home/itcmaster/projects/phdq3
cat artifacts/logs/p1_masks_20260916.status
tail -n 40 artifacts/logs/p1_masks_20260916.log
# 재실행할 필요가 있을 때만 새 RUN_NAME 지정; 기존 로그/JSON 재사용 금지
RUN_NAME=p1_masks_repeat_001 bash scripts/run_p1_masks.sh
```

이 스크립트는 환경 검사→54개 단위 테스트→B 파일 hash/strict load 확인→실제 B의
전 계층 attention mask 검사를 수행한다. 원본 텐서 비교는 변경되지 않은 기존
`weights_p1_validation_20260915.json` 증거와 연결한다. B/변환 입력/가중치 checker가
바뀌면 `check_weight_conversion.py`를 새 output/mapping 경로로 다시 실행해야 한다.
검사 CLI: `python blt_hf_checks/check_attention_mask.py --output <새 JSON 경로>`.

지원 범위는 eager/bf16/no-cache/무패딩이며 학습·optimizer 작업은 포함하지 않는다.
이 검사 당시 후속 단계였던 neuron용 학습·평가 코드는 현재 구현됐으며 `NEURON.md`를 따른다.

## 2026-09-15 정적·GPU·OSC 검사 — 과거 실행

itcerdo 프로젝트에서 tmux `phdq3-p1`의 별도 window로 실행했다.
`p1_osc_v5_20260915`는 exit_code=0으로 종료했다. 각 window는 종료 뒤 shell 대기다.

```bash
ssh itcerdo
cd /home/itcmaster/projects/phdq3
cat artifacts/logs/p1_osc_v5_20260915.status
tail -n 30 artifacts/logs/p1_osc_v5_20260915.log
```

재현 파일: `scripts/run_p1_validation.sh`(정적+무수정 HF),
`scripts/run_p1_load.sh`(HF loading info 집합 JSON 변환 수정 후 재검사),
`scripts/run_p1_osc.sh`(환경·단위 테스트·실제 후보 forward·20개 patch fixture).
완료 보고서는 덮어쓰지 않는다. **OSC 재검사는 새 RUN_NAME**으로 실행한다.

```bash
# itcerdo에서만 실행. 실제로 아직 사용하지 않은 RUN_NAME을 지정한다.
RUN_NAME=p1_osc_repeat_001 bash scripts/run_p1_osc.sh
```

패치 재현은 `tests/test_hf_model.py`, `tests/test_hf_attention.py`,
`tests/test_hf_patcher.py` 및 `blt_hf_checks/check_patch_parity.py`에 있다.
당시 검사는 optimizer/backward 없이 실행했다. 당시 없었던 neuron 제출 스크립트는 현재 구현됐다.

## 이동 전 인계 — 2026-09-15 당시 상태

itcerdo의 다운로드·CPU 변환은 정상 종료했다. tmux는 shell 대기 상태로 유지된다.
현재 실행 중인 학습·검증 job은 없다. 재접속과 로그 확인:

```bash
ssh itcerdo
cd /home/itcmaster/projects/phdq3
tmux attach -t phdq3-p1
# 세션 밖에서 결과 확인
cat artifacts/logs/p1_artifacts_20260915.status
tail -n 40 artifacts/logs/p1_artifacts_20260915.log
```

`scripts/run_p1_artifacts.sh`가 실행한 명령을 보존한다. 이미 B와 변환 보고서가
있으므로 이 스크립트를 그대로 재실행하면 덮어쓰기를 거부한다. 변환 재시도가
필요하면 기존 산출물을 보존하고 새 output/report 경로를 사용한다.

변환본: `artifacts/converted/blt-1b-hf-own` (itcerdo에만 존재).
정적 parity/strict load/OSC 구동을 다음 단계로 진행한다. neuron 제출은 사용자가
`showque`, `showappl` 확인 후 `--comment="field=<허용 field>;appl=pytorch"` 형식으로 한다.

아래는 초기 준비 명령 기록이다. itcerdo 정보·실제 환경 lock은 현재 확보 완료했으며
현재 상태는 맨 위 2026-09-16 절과 NOTES.md의 같은 날짜 기록을 따른다.

명령의 작업 디렉터리는 해당 노드의 PHDQ3 저장소 루트다.
에이전트는 neuron에 접속·전송·제출하지 않는다. itcerdo 경로는 사용자 제공 파일로 확정한다.

## mac — 지금 실행 가능한 검사

추가 패키지 설치 없이 Python 3.11에서 실행한다.

```bash
python3 -m unittest discover -s tests -p 'test_hf_*.py' -v
python3 blt_hf_checks/analyze_data_lengths.py
python3 blt_hf_checks/check_env.py --cpu-only
```

`analyze_data_lengths.py`는 9개 canonical TSV와 val/test M2만 읽는다.
데이터를 변경하지 않으며, 모델을 실행하지 않는다.

JSON을 보존하려면 `--output <새로운_결과경로>.json`을 지정한다.
기존 결과 파일이 있으면 덮어쓰지 않고 실패한다. 재검사는 새 이름을 사용한다.

## itcerdo — 접속정보 파일 확인 후

이미 설치한 `phdq_blt_hf` 환경에서 실행한다. 아래 명령에는 SLURM 작업이나 학습이 없다.

```bash
conda activate phdq_blt_hf
python blt_hf_checks/check_env.py
python blt_hf_checks/analyze_data_lengths.py
```

환경 보고서가 통과한 뒤 실제 패키지 목록을 기록한다. 기존 lock이 있다면 먼저
비교하고 보존하며, 재생성된 내용을 확인하지 않고 덮어쓰지 않는다.

```bash
python -m pip freeze
```

`requirements.in`은 계획의 의존성 입력 명세다. 아직 노드에서 수집한
`requirements.lock.txt`는 없으며, 환경 설치 완료라는 사용자 설명을 근거로
임의의 lock 내용을 만들지 않는다.

## neuron — 사용자 실행

사용자가 적절한 A100 SLURM allocation과 `phdq_blt_hf` 환경에서 실행할 준비 명령:

```bash
python blt_hf_checks/check_env.py
python blt_hf_checks/analyze_data_lengths.py
```

로그인 노드에서 GPU 검사·학습을 실행하지 않는다. SLURM 정책과 프로젝트 경로는
실제 사용자 환경에 맞춰야 한다. 학습 스크립트·변환본 B·OSC 로더는 아직 준비 중이며,
실행되지 않는 학습 제출 명령을 완료된 단계처럼 제공하지 않는다.
