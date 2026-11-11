# Neuron 학습·성능 평가 실행

작업 위치는 `/scratch/r984a02/phdq3`이며, 접속·파일 이관·제출은 사용자가 수행한다.
에이전트는 neuron에 접속하지 않았다. 아래 스크립트는 기존 `phdq_blt_hf` 환경을
사용하며 패키지 설치나 CUDA module load를 하지 않는다.

## 준비

**코드 정본은 GitHub `main` 하나다.** 전달용
`artifacts/releases/p1-neuron-code-20260916.tar.gz`는 현재 `main`보다 오래됐으므로
최종 실행 코드로 사용하지 않는다. 평소에는 현재 작업을 커밋한 뒤 `main`만
`pull --ff-only`/`push`한다. `outputs/blt_hf_eval/`도 Git 추적 대상이므로
`git reset --hard`는 평가 결과의 로컬 변경을 덮어쓴다. 학습 체크포인트가 있는
`outputs/blt_hf/`, `data/`, `artifacts/`, `ssh.md`는 Git에서 제외한다.
`git clean`은 실행하지 않는다.

기존 Git checkout인 경우:

```bash
cd /scratch/r984a02/phdq3
git fetch origin main
git merge --ff-only origin/main
git branch --set-upstream-to=origin/main main
git status --short
git log -1 --oneline
```

과거 로그 전용 커밋 때문에 로컬 `main`의 이력이 원격 `main`과 갈라졌다면 위
`merge --ff-only`는 안전하게 실패한다. 이 경우 바로 `pull`/`rebase`하지 말고
현재 커밋과 미커밋 변경을 보존한 뒤 한 번만 원격 `main`으로 정렬한다.
진행 중인 작업이 끝나기 전에는 코드를 교체하지 않는다.

2026-09-24 로그 전송 후 Neuron HEAD가 정확히 `089c377`인 기존 checkout의
일회성 정렬 절차는 다음과 같다. 먼저 HEAD와 브랜치를 검사한다. 검사에 실패하면
브랜치를 교체하지 않고 현재 상태를 확인한다. `main`에는 이 로그 브랜치의 파일이
모두 반영되어 있으며, 완료 상태와 과거에 잘린 로그만 수정되어 있다.

```bash
bash <<'SH'
cd /scratch/r984a02/phdq3
git fetch origin main || exit 1
if [ "$(git rev-parse HEAD)" != 089c3779232040815daf2fe61949fdba49cc90b5 ]; then
  echo 'Unexpected HEAD; stop before changing branches' >&2
  exit 1
fi
case "$(git branch --show-current)" in
  ''|main) ;;
  *) echo 'Unexpected branch; stop before changing branches' >&2; exit 1 ;;
esac
git tag neuron-before-single-main-20260924 HEAD || exit 1
git stash push -u -m neuron-before-single-main-20260924 || exit 1
git switch -C main origin/main || exit 1
git branch --set-upstream-to=origin/main main || exit 1
git status --short --branch
SH
```

위 절차의 이전 버전으로 이미 `8877a81`에 도달했지만 `## HEAD (no branch)`가
표시된다면, stash를 적용하거나 다시 reset하지 말고 아래처럼 `main`에 붙인다.
`origin/main`의 최신 문서 수정도 함께 받는다.

```bash
cd /scratch/r984a02/phdq3
git fetch origin main
git switch -C main origin/main
git status --short --branch
git log -1 --oneline
git stash list -1
```

위 stash는 미커밋 변경의 안전 사본이다. 새 `main`에 이미 들어간 코드를 다시
덮어쓰지 않도록 자동으로 `stash pop`하지 않는다. 이후에는 한 브랜치에서
`git pull --ff-only origin main`으로 받고, 변경 파일만 커밋한 뒤
`git -c credential.helper= push origin main`으로 보낸다. GitHub `Password`
프롬프트에는 계정 비밀번호가 아니라 PAT를 입력한다.
브랜치를 합쳐도 기존 run의 코드·scorer fingerprint는 바뀌지 않는다. 새 코드가
다른 fingerprint를 내면 기존 `RUN_ID`/`EVAL_DIR`의 이어하기는 거부되는 것이
정상이며, 새 실험은 새 ID와 디렉터리에서 시작한다.

압축만 풀어서 `.git`이 없는 경우:

```bash
cd /scratch/r984a02/phdq3
git init
git remote add origin https://github.com/esoterikosQ/phdq3.git
git fetch origin main
git reset --hard FETCH_HEAD
git branch -M main
git branch --set-upstream-to=origin/main main
git status --short
git log -1 --oneline
```

압축에는 가중치·전체 데이터·인증 파일이 포함되지 않는다. 로컬 전용 M2 scorer 사본은
공개 Git에서도 제외되므로 기존 실험용 압축에서 남은 `blt_hf/vendor/m2/`는 보존한다.
그 존재 여부와 hash는 `THIRD_PARTY_NOTICES.md` 절차로 확인한다.

코드·`data/Preprocessed`의 제공된 9개 TSV/6개 val·test M2 및 직접 변환본 B가
프로젝트 아래에 있어야 한다. 기존 데이터는 수정하거나 재분할하지 않는다.
B의 경로는 `artifacts/converted/blt-1b-hf-own`이며 현재 itcerdo에 있는 B를 사용자가
이관해야 한다. model loader는 `conversion_B_20260915.json`의 출력 SHA256과 로컬
파일을 대조하므로 원본 소스 weights를 neuron에 중복 복사할 필요는 없다.
HF 토큰은 필요 없다. 모든 모델 로드는 offline/local-files-only다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
showque
showappl
# batch shell에서 conda를 못 찾는 환경만 실제 conda.sh 경로를 지정한다.
# export CONDA_SH=/apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
```

이 프로젝트에서 확인한 field는 `nlp`다. 각 job script 맨 앞의 SBATCH header에
`--comment="field=nlp;appl=pytorch"`를 고정했다. `showque`와 `showappl`은 로그인
shell에서 직접 확인한다. job script를 `bash`로 실행하면 SBATCH 줄은 주석으로 처리되어
자원이 할당되지 않는다. 아래와 같이 반드시 `sbatch`로 직접 제출한다.

지원 GPU partition은 `amd_a100nv_8`(GPU당 CPU ≤8, active ≤4), `amd_a100_4`
(GPU당 CPU ≤16, active ≤2), `amd_h200nv_8`(GPU당 CPU ≤8, active ≤2)이다.
GPU job에는 `--gres=gpu:N`을 지정한다. 기본은
`amd_a100nv_8`, 1 node/1 SLURM task, GPU당 CPU 8개이며 torchrun이 GPU별 rank를 만든다.
본 학습은 A100 4GPU, 생성은 A100 1GPU를 기본으로 한다. H200은 긴 대기와 자원
단편화를 피하기 위해 한 작업당 최대 2GPU만 요청하며, 1GPU 검사·생성을 우선한다.
H200 4GPU 이상 작업은 제출하지 않는다. 같은 run의 재개에서는 GPU 수를 바꾸지 않는다.
일반 환경 검사는 A100(sm_80), H100/H200(sm_90), itcerdo 검증용 RTX 5090(sm_120)을
인식하고 각 GPU의 native BF16 지원 및 실제 BF16 matmul을 검사한다. BF16 미지원
V100(sm_70)은 거부한다. 파티션과 실제 capability가 A100=sm_80, H200=sm_90으로
일치해야 한다. 새 비교 run은 GPU 종류별로 RUN_ID·로그를 분리한다. 중단된 기존 run을
다른 GPU 종류에서 복구해야 할 때는 GPU 수와 BF16 설정을 유지하고 SLURM 로그에 전환
경계를 남긴다.
job array는 사용하지 않으며 running limit은 scheduler가 적용한다.

CPU 채점은 `cpu` partition을 사용한다. GPU를 할당해 CPU 채점을 기다리지 않는다.
본 학습의 기본 SLURM 제한 시간은 6시간이며 Python은 21,000초(5시간 50분)에
재개 가능한 checkpoint를 저장하고 종료한다. 남은 10분은 26GB급 checkpoint 게시와
정리에 사용한다. 생성·CPU 채점은 기존 1시간 55분 제한을 유지한다. 모든 job은
5분 전 TERM 신호도 보조 종료 신호로 사용한다.
반환 코드 75는 **재개 가능한 미완료**이며 완료로 해석하지 않는다. 자동 재제출하지 않는다.

## 1. 사용자 실행 학습 검사

```bash
# 1 GPU: 작은 모델의 실제 backward/optimizer 재개 검사 후, 실제 B의 긴 train 입력 2 update.
sbatch --export=ALL,RUN_ID=native-smoke-01,DATASET_TYPE=native,NUM_GPUS=1,TRAIN_MODE=smoke \
  scripts/train_blt_hf.sh

# 위 작업 완료 및 메모리 확인 후 2 GPU DDP 검사. 같은 RUN_ID를 재사용하지 않는다.
sbatch --gres=gpu:2 --cpus-per-task=8 \
  --export=ALL,RUN_ID=native-ddp-smoke-01,DATASET_TYPE=native,NUM_GPUS=2,TRAIN_MODE=smoke \
  scripts/train_blt_hf.sh

# H200 1 GPU BF16 smoke. 명령줄 -p가 스크립트의 기본 A100 partition을 덮어쓴다.
sbatch -p amd_h200nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --export=ALL,RUN_ID=native-h200-bf16-smoke-01,DATASET_TYPE=native,NUM_GPUS=1,TRAIN_MODE=smoke \
  scripts/train_blt_hf.sh

# H200 1 GPU 결과 확인 후 2 GPU DDP smoke.
sbatch -p amd_h200nv_8 --gres=gpu:2 --cpus-per-task=16 \
  --export=ALL,RUN_ID=native-h200-bf16-ddp-smoke-01,DATASET_TYPE=native,NUM_GPUS=2,TRAIN_MODE=smoke \
  scripts/train_blt_hf.sh

# tiny overfit: train 앞 4개에만 수행하는 별도 진단. 평가용 checkpoint로 사용하지 않는다.
sbatch --export=ALL,RUN_ID=native-overfit-01,DATASET_TYPE=native,NUM_GPUS=1,TRAIN_MODE=overfit,LR=0.0001,OVERFIT_STEPS=200 \
  scripts/train_blt_hf.sh
```

`smoke`는 `check_train_forward.py`에서 label shift, encoder/global/decoder gradient,
고정 entropy, optimizer/RNG 복원 후 연속 학습 결과를 확인한다. 이어 실제 B의 선택한
데이터셋에서 가장 긴 train 문장들을 사용해 optimizer 상태까지 할당한다.
이 진단이 실제 A100 학습 통과의 근거이며, 현재 에이전트가 실행한 상태는 아니다.
`overfit`의 최종 loss<0.05 여부는 `completed.json`에 기록한다.
`smoke/overfit` checkpoint는 본 평가 CLI가 거부한다.

검사 로그: 프로젝트 루트의 `slurm-blt-hf-train-<jobid>.out` 및 `.err`.
공통 batch trap이 로그 끝에 종료 시각, 총 경과 초, exit code를 성공·실패 모두 기록한다.
작업별 환경·9개 split 길이 보고서와 작은 모델의 training report는
`blt_hf_checks/results/neuron_<jobid>_<restart>_*.json`에 남는다.

## 2. 본 학습 및 재개

### native 2-epoch 통합 GLEU 시험 (A100 4GPU)

이 시험은 기존 3-epoch/10-epoch run과 다른 `RUN_ID`를 쓴다. 역전파는 target-token
loss로 수행하지만, 매 epoch 전체 native validation을 네 rank가 나눠 생성하고
corpus GLEU로 `best.json`과 `best_gleu.json`을 갱신한다. validation loss도 진단용으로
남긴다. M2는 checkpoint 선택에 필요하지 않아 이 단계에서 계산하지 않는다.
첫 시험은 기존 생성 조건인 beam 1로 통합 처리량과 GPU 메모리를 측정한다.
논문 조건의 beam 4는 이 결과를 확인한 뒤 **새 RUN_ID**로 실행한다.
2 epoch는 학습률 schedule의 일부이므로 이 run의 `EPOCHS`를 10으로 바꿔 이어갈 수
없다. 본 학습은 별도 10-epoch run을 처음부터 시작한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=native-gleu2-b1-s0,DATASET_TYPE=native,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=2,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1 \
  scripts/train_blt_hf.sh
```

시간 제한으로 exit 75가 나오면 코드와 위 설정을 그대로 유지하고 다음을 제출한다.

```bash
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=native-gleu2-b1-s0,DATASET_TYPE=native,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=2,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1,RESUME=outputs/blt_hf/native/native-gleu2-b1-s0/latest.json \
  scripts/train_blt_hf.sh
```

완료 판정은 `completed.json`의 `status=complete`, `epoch=2`,
`training_checks=passed`다. `validation/epoch-0001/metrics.json`과
`validation/epoch-0002/metrics.json`에 GLEU·입력/출력 hash가 남고,
`epoch_checkpoints.json`에는 각 epoch의 loss와 GLEU가 기록된다.
`best_gleu.json`/`best.json`은 높은 GLEU의 불변 checkpoint를 가리킨다.
동점이면 이른 epoch가 유지된다. `validation_gleu_complete` 로그의
`elapsed_seconds`는 4GPU 전체 validation 생성·GLEU 채점 시간이다.
작은 결과 요약은 Git 추적 대상인
`blt_hf_checks/results/native-gleu2-b1-s0_integrated_gleu.json`에도 남는다.
기존 별도 `select_best.py`는 이 통합 run에 다시 적용하지 않는다.

### 생성 병목 진단 (native 시험 checkpoint, A100 1GPU)

현재 OSC 생성은 `use_cache=False`라 생성 바이트마다 앞부분을 다시 계산한다.
통합 validation은 기본값 `VALIDATION_BATCH_SIZE=1`을 유지한다. 다음 명령은
**학습하지 않고** 실제 native checkpoint의 같은 64개 문장을 단건과 동일 prompt 길이
batch 4로 각각 생성해 시간·출력 token ID 일치·peak GPU 메모리를 기록한다.
beam 1과 4는 다른 job/report로 실행한다. 결과를 확인하기 전에는 batch 4를
본 학습의 기본값으로 사용하지 않는다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-gleu2-b1-s0/best_gleu.json
sbatch --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",BENCH_OUTPUT=blt_hf_checks/results/native-gleu2-beam1-batch4-bench.json,DATASET_TYPE=native,BLT_NUM_BEAMS=1,BATCH_SIZE=4,GROUPS=16 \
  scripts/bench_blt_generation.sh
sbatch --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",BENCH_OUTPUT=blt_hf_checks/results/native-gleu2-beam4-batch4-bench.json,DATASET_TYPE=native,BLT_NUM_BEAMS=4,BATCH_SIZE=4,GROUPS=16 \
  scripts/bench_blt_generation.sh
```

보고서의 `status=passed`, `token_id_mismatches=[]`, `speedup>1`과 메모리 여유를
확인한 뒤 **새 RUN_ID**의 학습 명령에만 `SELECTION_METRIC=val_gleu`,
`VALIDATION_BATCH_SIZE=4`를 함께 지정한다. 기존 native 2-epoch run은 코드
identity가 달라지므로 이 옵션으로 재개하지 않는다. cache 경로는 OSC에서
명시적으로 거부하며 별도 구현·검증 전에는 켜지 않는다.

### P3a 증분 생성 시제품 진단 (2026-09-26, A100 1GPU)

실행 중인 작업이 끝난 뒤 새 Git 코드를 반영한다. 현재 시제품은
`blt_hf/cache/global_reuse.py`의 **batch 1·beam 1 전용 진단**이며 학습,
`eval.py`, 통합 validation의 기본 생성 경로를 변경하지 않는다. itcerdo의
사전학습 1B에서 96 step, Neuron native fine-tuned checkpoint에서
12문장 × 32 step의 다음 byte ID가 일치했다. 이는 완성 문장이나 beam 4의
동등성은 아니다. 아래 `01.json`, `02.json`, `03_decoder.json` 명령은 실행 이력이며 기존
보고서 경로로 **재실행하지 않는다**.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/best.json
export BENCH_OUTPUT=blt_hf_checks/results/p3a_native_cache_probe_01.json
test -f "$CKPT_PATH"
test ! -e "$BENCH_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=00:30:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",BENCH_OUTPUT="$BENCH_OUTPUT",DATASET_TYPE=native,SAMPLES=12,STEPS=32,REUSE_DECODER=0 \
  scripts/bench_blt_cache.sh
```

`probe_token_parity=passed`와 각 사례의 `mismatches=0`, `global_skips`,
`timing_excluding_initial`을 확인한다. 이 검사는 native validation에서
문장 길이별로 뽑은 12개 입력의 최대 32 step만 비교한다. 통과해도 전체
validation이나 beam 4의 결과 동일성은 보증하지 않으며, 시제품을 본 평가에
사용하지 않는다. 실패해도 JSON 보고서는 남고 job은 nonzero로 종료한다.
재검사할 때는 새 `BENCH_OUTPUT` 경로를 지정한다. Neuron 제출은 사용자만 한다.

915640의 첫 probe는 `probe_token_parity=passed`(384/384)였지만 측정 forward
속도는 1.372배였다. 경계 변경에서 KV 꼬리 계산이 느린 문제를 고쳐 해당
step은 기준 global full forward를 사용한다. 다음 `02.json` 명령은 915655에서
이미 실행됐다. 기존 `01.json`/`02.json`을 덮어쓰지 않는다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/best.json
export BENCH_OUTPUT=blt_hf_checks/results/p3a_native_cache_probe_02.json
test -f "$CKPT_PATH"
test ! -e "$BENCH_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=00:30:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",BENCH_OUTPUT="$BENCH_OUTPUT",DATASET_TYPE=native,SAMPLES=12,STEPS=32,REUSE_DECODER=0 \
  scripts/bench_blt_cache.sh
```

915655의 수정 probe도 384/384 다음 ID가 일치했고 forward 합계는
15.659→10.701초, **1.463배**였다. global 재계산 step의 느린 문제가
거의 사라졌지만 2배 목표는 아직 미달이다. 선택형 decoder KV 진단도
915710에서 이미 실행됐다. 아래 명령은 재현을 위한 이력이며 같은 보고서
경로로 다시 제출하지 않는다. 이 검사도 본 평가 backend를 바꾸지 않았다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/best.json
export BENCH_OUTPUT=blt_hf_checks/results/p3a_native_cache_probe_03_decoder.json
test -f "$CKPT_PATH"
test ! -e "$BENCH_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=00:30:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",BENCH_OUTPUT="$BENCH_OUTPUT",DATASET_TYPE=native,SAMPLES=12,STEPS=32,REUSE_DECODER=1 \
  scripts/bench_blt_cache.sh
```

915710은 같은 checkpoint·데이터·12문장과 코드에서 decoder 재사용을 켜
384/384 다음 ID가 일치했지만, forward 합계는 16.405→11.917초(1.377배)로
02의 1.463배보다 낮았다. EOS 생성 사례가 없고 전체 validation·beam4·GLEU
검사도 아니다. 두 큰 지연의 원인은 이 보고서만으로 특정할 수 없으므로
2배 목표만으로 적용을 배제한 이전 판단은 철회했다. 당시에는 통합 backend가
없었고, 현재는 아래 선택형 global 경로를 준비했다. 상세 분석은
`cache/DESIGN.md`에 있다.

아래는 job 915864에서 **완료한 지연 원인 진단 명령 이력**이다. 같은 native
체크포인트와 03 보고서의 두 접두부를 복원해 decoder on/off를 번갈아
3회씩 실행했다. 기존 보고서 경로로 재실행하지 않는다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export DIAG_OUTPUT=blt_hf_checks/results/p3a_native_cache_spike_diagnostic_04.json
test -f "$CKPT_PATH/model.safetensors"
test ! -e "$DIAG_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=00:30:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",DIAG_OUTPUT="$DIAG_OUTPUT",REPEATS=3 \
  scripts/diagnose_blt_cache_spikes.sh
```

04 보고서에서 2290 step 3의 첫 decoder 재사용 지연은 약 569ms로 재현됐으나
후속 on/off 실행은 약 26ms였다. 297 step 27은 patch 경계 변경 때문에
재사용을 멈춘 것이 맞고, 반복 시 약 43ms로 263ms 지연은 재현되지 않았다.
캐시 분기 오류는 발견되지 않았다. 다음 시험은 선택형 beam-1 완성 생성의
출력 일치와 전체 native validation의 실제 처리 시간이다. 상세 분석은
`cache/DESIGN.md`다.

### P3a 선택형 global 캐시 완성 생성 시험

`global-prefix-greedy-v1`은 **decoder KV를 끄고** global 재사용만 사용하는
beam 1·batch 1 생성 backend다. 기존 HF no-cache가 기본이며 학습과
beam 4에는 적용하지 않는다. 먼저 같은 native checkpoint와 validation
길이별 12문장을 양쪽에서 끝까지 생성해 byte ID·EOS·UTF-8·완성 문장과
시간을 비교한다. 아래는 사용자 job 915874의 실행 이력이며 기존 출력
경로로 재실행하지 않는다. 보고서의 `complete_output_parity=passed`는
이 12문장에만 적용된다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export BENCH_OUTPUT=blt_hf_checks/results/p3a_native_complete_global_05.json
test -f "$CKPT_PATH/model.safetensors"
test ! -e "$BENCH_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=01:55:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",BENCH_OUTPUT="$BENCH_OUTPUT",SAMPLES=12,MAX_NEW_BYTES=768 \
  scripts/bench_blt_cache_complete.sh
```

915874는 12/12 완성 출력 일치, 모두 EOS 종료, 기준 43.594초→global 캐시
29.711초(1.467배)였다. 첫 기준 문장을 제외하면 약 1.432배다. 다음은
**전체 native validation 2,634건** 비교다. 아래 두 생성 job은 같은
checkpoint·beam1·batch1·최대 768바이트를 쓰며 backend와 출력 디렉터리만
다르다. 새 Git 코드는 공유 checkout의 다른 작업이 끝난 후 받는다. 두 job은
각각 1GPU이며 독립적으로 제출할 수 있다. exit 75면 같은 명령과 디렉터리로
재개하고, `shards/0000/complete.json`이 생겨야 완료다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export HF_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-hf-p3a
export CACHE_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-global-p3a
test -f "$CKPT_PATH/model.safetensors"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:55:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",EVAL_DIR="$HF_EVAL_DIR",DATASET_TYPE=native,SPLIT=val,BLT_NUM_BEAMS=1,BATCH_SIZE=1,SHARD_COUNT=1,SHARD_ID=0,GENERATION_BACKEND=hf-generate-exact-length-unpadded-v1 \
  scripts/eval_blt_hf.sh
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:55:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",EVAL_DIR="$CACHE_EVAL_DIR",DATASET_TYPE=native,SPLIT=val,BLT_NUM_BEAMS=1,BATCH_SIZE=1,SHARD_COUNT=1,SHARD_ID=0,GENERATION_BACKEND=global-prefix-greedy-v1 \
  scripts/eval_blt_hf.sh
```

두 생성이 모두 완료된 뒤 **CPU**에서 불변 batch 파일·지문을 검증하고 매
문장의 token ID·EOS·UTF-8·문자열, 생성 시간 합계와 p50/p95를 비교한다.
M2/GLEU 채점은 이 출력 동등성·속도 검사에 필요하지 않다. 새 보고서 경로를
사용한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
export HF_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-hf-p3a
export CACHE_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-global-p3a
export COMPARE_OUTPUT=blt_hf_checks/results/p3a_native_full_global_06.json
test -f "$HF_EVAL_DIR/shards/0000/complete.json"
test -f "$CACHE_EVAL_DIR/shards/0000/complete.json"
test ! -e "$COMPARE_OUTPUT"
sbatch -p cpu --cpus-per-task=4 --time=01:00:00 \
  --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,REFERENCE_DIR="$HF_EVAL_DIR",CANDIDATE_DIR="$CACHE_EVAL_DIR",COMPARE_OUTPUT="$COMPARE_OUTPUT" \
  scripts/compare_blt_cache_eval.sh
```

916171 전수 비교는 2,634건 중 32건의 token ID·문자열 불일치로 의도된
exit 1을 반환했다. 입력/체크포인트/설정/완료 파일은 일치했고, 생성 시간
합계는 기준 9,074.297초→global 캐시 6,620.644초(1.371배)였다. 이
보고서는 덮어쓰지 않는다. 아래 진단은 **32개 불일치 행의 첫 분기까지**
기준 접두부를 재생하고 그 지점의 캐시 추론, global 재계산, HF식 마스크
전달 결과를 기록한다. `global_skip_changes_greedy_choice`는 동일 접두부의
재생과 기준 ID 재현이 모두 확인된 경우에만 붙인다. 진단 결과의 exit 0은
분석 완료를 뜻하며 캐시 출력 동일성을 뜻하지 않는다. 사용자가 Neuron의
공유 checkout에서 실행 중인 작업을 마친 후 새 코드를 받는다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export COMPARE_REPORT=blt_hf_checks/results/p3a_native_full_global_06.json
export DIAG_OUTPUT=blt_hf_checks/results/p3a_native_divergence_diag_07.json
test -f "$CKPT_PATH/model.safetensors"
test -f "$COMPARE_REPORT"
test ! -e "$DIAG_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",COMPARE_REPORT="$COMPARE_REPORT",DIAG_OUTPUT="$DIAG_OUTPUT" \
  scripts/diagnose_blt_cache_divergence.sh
```

916179 진단은 exit 0, 32/32 첫 분기 재생 성공이었다. 17건은 global skip 후
기준과 다른 토큰이 선택됐지만 같은 입력을 full 재계산하면 기준 ID가 돌아와
재사용이 직접 원인임을 확인했다. 나머지 15건은 full 재계산도 저장된 HF
기준과 달랐고, 그중 11건은 cached/full logit 차이가 정확히 0이었다.
Transformers 5.16.1 `generate()`는 `logits_to_keep=1`을 명시하지만 v1
수동 경로는 기본값 0이다. 이 15건을 분리하기 위해 진단 코드에 HF식
`logits_to_keep=1`을 추가했다. 07은 보존하고 **새 08 결과**를 만든다.
아래 명령은 공유 checkout의 다른 작업이 종료된 뒤 사용자만 Neuron에서 실행한다.

```bash
cd /scratch/r984a02/phdq3
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export COMPARE_REPORT=blt_hf_checks/results/p3a_native_full_global_06.json
export DIAG_OUTPUT=blt_hf_checks/results/p3a_native_divergence_logit_shape_08.json
test ! -e "$DIAG_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",COMPARE_REPORT="$COMPARE_REPORT",DIAG_OUTPUT="$DIAG_OUTPUT" \
  scripts/diagnose_blt_cache_divergence.sh
```

916207의 08 진단은 32/32 첫 분기를 재현했고, HF와 같은
`logits_to_keep=1` full forward가 **32/32 기준 ID를 두 번 연속 복구**했다.
따라서 15건의 full0↔HF 차이는 logit projection 모양으로 설명된다.
17건은 이미 global skip이 greedy ID를 바꾼 것으로 확인됐으며, 두 요인이
독립적으로 존재한다. 기존 v1 출력과 평가 코드는 보존한다.

선택형 `global-prefix-greedy-v2`는 **logit projection만** HF와 같이 마지막
위치 하나로 바꿨다. global skip 조건과 decoder KV off는 v1과 같다.
새 backend의 효과를 먼저 v1 불일치 32건과 05의 길이별 대조군 12건을
끝까지 생성해 검사한다. 09는 표본 진단이며 전체 validation parity나
속도 판정이 아니다. 사용자가 Neuron의 공유 checkout에서 실행 중인
작업이 끝난 뒤 새 코드를 받아 아래 명령을 제출한다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export COMPARE_REPORT=blt_hf_checks/results/p3a_native_full_global_06.json
export CONTROL_REPORT=blt_hf_checks/results/p3a_native_complete_global_05.json
export BENCH_OUTPUT=blt_hf_checks/results/p3a_native_cache_v2_mismatch_09.json
test -f "$CKPT_PATH/model.safetensors"
test -f "$COMPARE_REPORT"
test -f "$CONTROL_REPORT"
test ! -e "$BENCH_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",COMPARE_REPORT="$COMPARE_REPORT",CONTROL_REPORT="$CONTROL_REPORT",BENCH_OUTPUT="$BENCH_OUTPUT" \
  scripts/bench_blt_cache_v2_mismatches.sh
```

916208의 09 시험은 exit 0, 기존 v1 불일치 32건 중 12건이 v2에서 HF와
일치했고 **20건은 남았다**. 대조군 12건은 모두 일치했다. 남은 20건 모두
07 진단에서 첫 분기에 global skip이 일어난 행이다. 선택형
`global-prefix-guarded-greedy-v1`은 v2의 `logits_to_keep=1`을 유지하고,
global skip으로 얻은 상위 두 합법 바이트의 logit 차이가 **1.0 이하**일 때만
같은 입력을 full forward로 재계산한다. 07에서 관찰된 첫 분기 최대 logit
차이 0.421875의 두 배보다 큰 보수적 *실험용* 문턱이며, 미관측 입력에
대한 동일성 보증은 아니다. 보고서의 `cache_preliminary_skips`와
`cache_guard_refreshes`로 재계산 빈도를 확인한다. 10은 같은 44문장
표본에서 출력·시간을 비교한다. 사용자만 Neuron에서 작업 중인 공유
checkout을 비운 후 실행한다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export COMPARE_REPORT=blt_hf_checks/results/p3a_native_full_global_06.json
export CONTROL_REPORT=blt_hf_checks/results/p3a_native_complete_global_05.json
export BENCH_OUTPUT=blt_hf_checks/results/p3a_native_cache_guarded_10.json
test -f "$CKPT_PATH/model.safetensors"
test -f "$COMPARE_REPORT"
test -f "$CONTROL_REPORT"
test ! -e "$BENCH_OUTPUT"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",COMPARE_REPORT="$COMPARE_REPORT",CONTROL_REPORT="$CONTROL_REPORT",BENCH_OUTPUT="$BENCH_OUTPUT" \
  scripts/bench_blt_cache_guarded_mismatches.sh
```

916210의 10 시험은 exit 0, 기존 불일치 32건과 대조군 12건의 완성 출력이
모두 HF 기준과 일치했다. 2,505회 global 재사용 후보 중 59회(2.4%)를
full forward로 갱신했다. 44문장 생성 시간 합계는 09의 117.76초에서
124.36초로 6.60초 증가했다. job 전체 시간 159→322초의 차이 중 큰 부분은
생성 외 구간이다. 환경 보고서 생성 시각과 전체 데이터 보고서 생성 시각
사이는 약 10→119초였지만, 이 간격은 CUDA 환경 검사·데이터 스캔·파일
기록을 함께 포함하므로 개별 원인은 미확인이다. 첫 문장 55의 생성은
3.90→11.29초로 늘었고 재계산 5회가 있었으나, 첫 추론 초기화와 노드
상태의 영향을 구분할 계시가 없다. 이 지연은 별도 관측 사항으로 보존하고,
캐시 채택 판단에는 아래 **전체 split 순수 생성 시간**을 사용한다.

### guarded 전체 native validation 생성·비교 (A100 1GPU)

동일한 native-main-01 불변 checkpoint와 2,634건 validation을 사용한다.
기준 HF 결과는 이미 `native-main-01-beam1-hf-p3a`에 완료되어 있다.
guarded 결과는 새 디렉터리에 저장한다. 실행 중인 공유 checkout 작업을
모두 마친 뒤 pull한다. 이 실험은 생성·출력·시간 비교이며 M2 채점은 하지
않는다. 1시간 55분 제한으로 exit 75가 나오면 **코드·설정을 바꾸지 않고**
같은 `sbatch` 명령을 다시 제출한다. 저장된 문장은 재생성하지 않는다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-main-01/step-00001155-24b756e2
export HF_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-hf-p3a
export CACHE_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-guarded-p3a
test -f "$CKPT_PATH/model.safetensors"
test -f "$HF_EVAL_DIR/shards/0000/complete.json"
test ! -e "$CACHE_EVAL_DIR/run.json"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:55:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",EVAL_DIR="$CACHE_EVAL_DIR",DATASET_TYPE=native,SPLIT=val,BLT_NUM_BEAMS=1,BATCH_SIZE=1,SHARD_COUNT=1,SHARD_ID=0,GENERATION_BACKEND=global-prefix-guarded-greedy-v1 \
  scripts/eval_blt_hf.sh
```

재개할 때는 위 `test ! -e "$CACHE_EVAL_DIR/run.json"`만 생략하고 같은
`sbatch` 명령을 다시 제출한다. `complete.json`이 생긴 뒤에만 비교한다.
기존 HF 실행과 새 캐시 실행의 공통 모델·데이터·scorer 코드는 같아야 한다.
비교기는 세 생성 경로의 코드 버전 차이만 허용하고 각 실행의 지문과 모든
batch 해시를 다시 확인한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
export HF_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-hf-p3a
export CACHE_EVAL_DIR=outputs/blt_hf_eval/native/val/native-main-01-beam1-guarded-p3a
export COMPARE_OUTPUT=blt_hf_checks/results/p3a_native_full_guarded_11.json
test -f "$HF_EVAL_DIR/shards/0000/complete.json"
test -f "$CACHE_EVAL_DIR/shards/0000/complete.json"
test ! -e "$COMPARE_OUTPUT"
sbatch -p cpu --cpus-per-task=4 --time=01:00:00 \
  --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,REFERENCE_DIR="$HF_EVAL_DIR",CANDIDATE_DIR="$CACHE_EVAL_DIR",COMPARE_OUTPUT="$COMPARE_OUTPUT" \
  scripts/compare_blt_cache_eval.sh
```

`p3a_native_full_guarded_11.json`의 2,634건 출력 일치·생성 시간·p50/p95·
갱신 횟수를 확인한다. 속도 이득이 실사용에 유의미하면 별도 새 `RUN_ID`의
학습→guarded 생성→전체 validation GLEU→best 선택에 backend를 연결한다.
기존 학습 run이나 HF 기본 경로는 바꾸지 않는다.

### 11 전수 비교 결과와 guarded 통합 GLEU 학습

사용자가 업로드한 916222·916259의 guarded 생성은 같은 코드·출력
디렉터리에서 2,634건 완료됐다. 로컬 CPU 비교 보고서
`blt_hf_checks/results/p3a_native_full_guarded_11.json`은 batch 해시와
실행 지문을 확인한 뒤 **2,634/2,634 token ID·EOS·문자열 일치**를 기록했다.
동일 native checkpoint의 순수 생성 합계는 HF 9,074.30초, guarded
6,906.76초(1.314배, 약 36.1분 절약)다. p50은 2.77→1.98초,
p95는 6.47→5.48초다. guarded의 global 재사용 후보 150,352회 중
1,792회 full refresh였다. 5090/다른 데이터셋/beam 4의 결과로 일반화하지
않는다. 두 Neuron job의 준비·데이터 검사 시간은 순수 생성 합계에 없다.

다음은 **새** native 10-epoch 본 학습이다. 학습 역전파는 기존 byte
target loss 그대로이며, 매 epoch 전체 validation의 beam-1 생성만
`global-prefix-guarded-greedy-v1`으로 수행하고 corpus GLEU 최고 checkpoint를
선택한다. 최대 10 epoch으로 schedule을 처음부터 고정하고, 전체 validation GLEU가
엄격히 증가하지 않는 epoch가 연속 3회 나오면 그 epoch 뒤에 중단한다. 동점도
미개선으로 센다. 최소 개선 폭은 두지 않으며 test 점수는 중단·선택에 쓰지 않는다.
2-epoch 시험을 거쳐 연장하지 않는다. 모델·scorer·기존
HF 기본값은 바꾸지 않았다. 캐시 개발 창이
별도 worktree에서 작업 중이어도, 이 run이 시작되면 Neuron의 공유 checkout
코드와 이 run의 GPU 수·설정을 완료 때까지 고정한다.

```bash
cd /scratch/r984a02/phdq3
git status --short --branch
git pull --ff-only origin main
conda activate phdq_blt_hf
test ! -e outputs/blt_hf/native/native-gleu10-guarded-b1-s0/run.json
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --time=06:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=native-gleu10-guarded-b1-s0,DATASET_TYPE=native,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,EARLY_STOPPING_PATIENCE=3,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1,VALIDATION_BATCH_SIZE=1,VALIDATION_GENERATION_BACKEND=global-prefix-guarded-greedy-v1 \
  scripts/train_blt_hf.sh
```

6시간 제한으로 exit 75가 나면 같은 코드·설정·4 GPU로 아래 명령을 반복
제출한다. 최대 10 epoch이 한 job에서 끝날 필요는 없다. 최종 확인 대상은
`completed.json`의 `status=complete`, `planned_epochs=10`, 실제 `epoch`,
`stop_reason=max_epochs|gleu_patience`, 완료한 각 epoch의
`validation/epoch-XXXX/metrics.json`에 기록된 `generation_backend`, `gleu`,
`cache_guard_refreshes`, 그리고 최고 GLEU checkpoint를 가리키는
`best_gleu.json`이다. 기존 2-epoch checkpoint는 이 run에 재개하지 않는다.
후속 learner·union·lang8 비교에도 최대 10 epoch, 같은 GLEU 개선·patience
규칙과 seed를 적용하고 실제 완료 epoch를 함께 보고한다. 2026-09-28 결정에 따라
데이터셋별 guarded/HF 출력 일치 검사는 후순위로 미룬다. 후속 학습은 guarded
validation으로 먼저 진행하되, 다른 데이터셋에서의 출력 동등성은 아직 주장하지
않는다. 최종 test 점수는 HF 생성과 같은 scorer로 따로 산출한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
test -f outputs/blt_hf/native/native-gleu10-guarded-b1-s0/latest.json
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --time=06:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=native-gleu10-guarded-b1-s0,DATASET_TYPE=native,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,EARLY_STOPPING_PATIENCE=3,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1,VALIDATION_BATCH_SIZE=1,VALIDATION_GENERATION_BACKEND=global-prefix-guarded-greedy-v1,RESUME=outputs/blt_hf/native/native-gleu10-guarded-b1-s0/latest.json \
  scripts/train_blt_hf.sh
```

### 후속 learner: native test 채점과 독립적으로 시작

native의 개선된 학습·선택 절차를 `korean_learner`에 새 RUN_ID로 적용한다.
기존 3-epoch `learner-main-01` checkpoint에서 이어가지 않는다. 데이터셋별
guarded/HF 출력 일치 검사는 뒤로 미룬 상태이므로 이 validation GLEU는
guarded backend의 결과로 기록한다. native test 채점 중에도 다른 Neuron job으로
제출할 수 있다. 같은 checkout의 실행 코드는 학습 완료까지 고정한다.
2026-09-28 결정으로 A100 4GPU 작업시간을 12시간으로 늘린다.
`--time`과 내부 종료 한도 `MAX_SECONDS`를 함께 지정한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
test ! -e outputs/blt_hf/korean_learner/learner-gleu10-guarded-b1-s0/run.json && \
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --time=12:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=learner-gleu10-guarded-b1-s0,DATASET_TYPE=korean_learner,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,EARLY_STOPPING_PATIENCE=3,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1,VALIDATION_BATCH_SIZE=1,VALIDATION_GENERATION_BACKEND=global-prefix-guarded-greedy-v1,MAX_SECONDS=42000 \
  scripts/train_blt_hf.sh
```

시간 제한에 따라 `exit 75`로 중단되면 아래 명령으로 재개한다. `latest.json`이
있고 `completed.json`이 없는 상태인지 먼저 확인한다. 전체 validation은 epoch
안에서 재개되지 않으므로, 반복해서 같은 validation에서 시간 제한에 걸리면
작업시간 또는 validation 저장 절차를 개선하기 전에는 재제출하지 않는다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
test -f outputs/blt_hf/korean_learner/learner-gleu10-guarded-b1-s0/latest.json && \
test ! -e outputs/blt_hf/korean_learner/learner-gleu10-guarded-b1-s0/completed.json && \
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --time=12:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=learner-gleu10-guarded-b1-s0,DATASET_TYPE=korean_learner,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,EARLY_STOPPING_PATIENCE=3,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1,VALIDATION_BATCH_SIZE=1,VALIDATION_GENERATION_BACKEND=global-prefix-guarded-greedy-v1,MAX_SECONDS=42000,RESUME=outputs/blt_hf/korean_learner/learner-gleu10-guarded-b1-s0/latest.json \
  scripts/train_blt_hf.sh
```

### 후속 lang8: H200 1GPU 장시간 작업

learner와 별도 GPU partition에서 동시에 제출한다. `lang8`은 학습 76,692행,
validation 16,434행이며, 검증 2,634행의 native가 A100 4GPU에서 약 29분
걸린 기록만으로는 H200 1GPU 속도를 정확히 예측할 수 없다. GPU 수와
표본 수를 단순 환산하면 validation만 약 12시간이고 H200 성능·문장 길이에
따라 크게 달라진다. 학습까지 합쳐 한 epoch는 대략 10–20시간으로 보고
첫 job에 24시간을 배정한다. 이는 실측값이 아니며 queue 대기는 포함하지
않는다. 10 epoch 모두 필요하면 대략 100–200 GPU시간이고 조기 종료는
이를 줄일 수 있다. 첫 epoch가 완료되면 그 로그로 다음 job 길이를 조정한다.
기본 스크립트의 6시간과 `MAX_SECONDS=21000`을 각각 24시간과 85,200초로
덮어쓴다. 한 epoch의 validation이 23시간 40분 안에도 끝나지 않으면
현재 코드는 그 validation을 재개하지 못하므로 같은 설정으로 반복 제출하지 않는다.
partition이 24시간 요청을 거절한다면 허용 `MaxTime`을 확인하고 다시 계획한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
test ! -e outputs/blt_hf/lang8/lang8-gleu10-guarded-h200-1gpu-b1-s0/run.json && \
sbatch -p amd_h200nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=24:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=lang8-gleu10-guarded-h200-1gpu-b1-s0,DATASET_TYPE=lang8,NUM_GPUS=1,TRAIN_MODE=train,EPOCHS=10,EARLY_STOPPING_PATIENCE=3,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1,VALIDATION_BATCH_SIZE=1,VALIDATION_GENERATION_BACKEND=global-prefix-guarded-greedy-v1,MAX_SECONDS=85200 \
  scripts/train_blt_hf.sh
```

`exit 75`이고 `completed.json`이 없다면 같은 GPU 수와 설정으로 재개한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
test -f outputs/blt_hf/lang8/lang8-gleu10-guarded-h200-1gpu-b1-s0/latest.json && \
test ! -e outputs/blt_hf/lang8/lang8-gleu10-guarded-h200-1gpu-b1-s0/completed.json && \
sbatch -p amd_h200nv_8 --gres=gpu:1 --cpus-per-task=8 \
  --time=24:00:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=lang8-gleu10-guarded-h200-1gpu-b1-s0,DATASET_TYPE=lang8,NUM_GPUS=1,TRAIN_MODE=train,EPOCHS=10,EARLY_STOPPING_PATIENCE=3,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,SELECTION_METRIC=val_gleu,VALIDATION_BEAMS=1,VALIDATION_BATCH_SIZE=1,VALIDATION_GENERATION_BACKEND=global-prefix-guarded-greedy-v1,MAX_SECONDS=85200,RESUME=outputs/blt_hf/lang8/lang8-gleu10-guarded-h200-1gpu-b1-s0/latest.json \
  scripts/train_blt_hf.sh
```

union은 두 작업의 결과와 사용 가능한 GPU를 보고 별도 RUN_ID로 제출한다.

### 기존 분리형 10-epoch 절차

새 사이클은 2026-09-18의 3-epoch run과 다른 RUN_ID를 쓴다. 기존
`union-h200-4gpu-main-01`은 재개하지 않는다. 재개 identity는 Git commit 전체가 아니라
학습 단계에서 실제 실행하는 파일의 `code_hash`와 데이터·모델·schedule·world size로
결정한다. 로그만 추가한 commit은 재개를 깨뜨리지 않는다. commit·job·GPU 전환 기록은
`provenance.jsonl`에 별도로 누적한다. 실행 파일 hash가 달라지면 재개를 거부한다.

```bash
python3 - <<'PY'
import json
from pathlib import Path
from blt_hf.runtime import code_identity, TRAIN_RUNTIME_FILES
run = Path('outputs/blt_hf/union/union-v2-s0/run.json')
saved = json.loads(run.read_text())
current = code_identity(TRAIN_RUNTIME_FILES)
print('saved code hash:', saved['code_hash'])
print('current code hash:', current['code_hash'])
print('different code files:', sorted(k for k in set(saved['code_files']) | set(current['code_files'])
                                     if saved['code_files'].get(k) != current['code_files'].get(k)))
print('saved world/effective batch:', saved['world_size'], saved['effective_batch'])
PY
```

```bash
# seed 0의 새 10-epoch native run. learner/union/lang8도 서로 다른 RUN_ID를 쓴다.
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=native-v2-s0,DATASET_TYPE=native,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0 \
  scripts/train_blt_hf.sh

# 시간 제한/중단 후, 동일 코드·설정·GPU 수·RUN_ID로 이어서 수행한다.
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=native-v2-s0,DATASET_TYPE=native,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,RESUME=outputs/blt_hf/native/native-v2-s0/latest.json \
  scripts/train_blt_hf.sh

# 별도 실험 ID, 각 원본 split 유지. native 결과 확인 후 순차 제출한다.
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=learner-v2-s0,DATASET_TYPE=korean_learner,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0 \
  scripts/train_blt_hf.sh
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 --time=00:30:00 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=union-v2-s0,DATASET_TYPE=union,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,MAX_STEPS=1,MAX_SECONDS=1200 \
  scripts/train_blt_hf.sh
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=lang8-v2-s0,DATASET_TYPE=lang8,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0 \
  scripts/train_blt_hf.sh

# 재개 경로를 먼저 1 update만 검사할 때. 성공하면 latest.json이 1 step 전진한다.
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 --time=00:30:00 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=union-v2-s0,DATASET_TYPE=union,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,RESUME=outputs/blt_hf/union/union-v2-s0/latest.json,MAX_STEPS=1,MAX_SECONDS=1200 \
  scripts/train_blt_hf.sh

# 위의 최초 step과 재개 step이 모두 성공한 뒤 같은 checkpoint에서 제한 없이 계속한다.
sbatch -p amd_a100nv_8 --gres=gpu:4 --cpus-per-task=32 \
  --export=ALL,CONDA_ENV=phdq_blt_hf,RUN_ID=union-v2-s0,DATASET_TYPE=union,NUM_GPUS=4,TRAIN_MODE=train,EPOCHS=10,WARMUP_RATIO=0.05,EFFECTIVE_BATCH=32,SEED=0,RESUME=outputs/blt_hf/union/union-v2-s0/latest.json \
  scripts/train_blt_hf.sh
```

학습 기본값은 10 epoch, LR 1e-5, 전체 update의 5% warmup, cosine decay,
AdamW `(0.9,0.95)`, eps 1e-8, weight decay 0.1, grad clip 1.0이다.
먼저 seed 0으로 native→learner→union→lang8의 학습·validation 선택·test 채점을 끝내
전체 절차를 확인한다. 그다음 `SEED=1`, `SEED=2`를 각각 `-s1`, `-s2` RUN_ID로
반복하며 같은 RUN_ID에 seed를 섞지 않는다.
main 모델 전체(해시 임베딩 포함)를 학습하며 entropy patcher는 고정한다.
원본·변환 artifact와 동일하게 파라미터·gradient·Adam 상태·연산은 BF16을 유지하고
gradient checkpointing을 쓴다. 코드가 각 dtype을 검사하며 불일치는 즉시 실패한다.
2026-09-17 이전 job 909747·910019는 잘못된 FP32 학습 정책으로 실행됐으므로 BF16
학습 검증으로 인정하지 않으며 새로운 RUN_ID로 smoke를 다시 실행한다.

microbatch는 rank당 **무패딩 1개**이며 `EFFECTIVE_BATCH`를 GPU 수에 맞춰 나누고
`no_sync`로 누적한다. 마지막 step도 원본 샘플을 버리거나 중복 가중하지 않는다.
남는 rank는 zero-loss 동기화 계산만 하며, loss는 실제 supervised token의 전역 평균이다.
DDP는 optimizer 상태 할당 전에 gradient bucket view를 준비하는 2회의 계산을 수행한다.
이때 파라미터는 업데이트하지 않고 RNG를 복구한다.

**메모리·저장공간**: 파라미터 수에는 큰 hash embedding이 포함되므로 “1B이니 작다”는
가정을 하지 않는다. BF16 Adam/DDP도 GPU당 전체 상태를 복제한다. 코드는 최소 메모리
추정치를 검사하지만 activation·통신·allocator까지 보장하지는 않으므로 긴 입력 smoke가
필요하다. OOM이면 full training을 강행하지 않고 sharding 계획을 추가한다. 현재 FSDP/ZeRO
자동 전환은 없다. checkpoint 하나는 BF16 모델과 Adam 상태를 함께 저장해 수십 GB다.
저장 전에 scratch 여유 공간/쿼터를 확인한다. 체크포인트는 자동 삭제하지 않는다.

- 경로: `outputs/blt_hf/<dataset>/<RUN_ID>/step-<update>-<id>/`
- `model.safetensors`, `training.pt`, `checkpoint.json`을 staging directory에 완성한 뒤 게시한다.
- `latest.json`/`best.json`은 작은 포인터다. 기본 분리형 run의 `best.json`은 전체
  validation target-token loss 기준이다. `SELECTION_METRIC=val_gleu` 통합 run에서는
  GLEU 기준이며 `best_gleu.json`도 같은 checkpoint를 가리킨다.
  `epoch_checkpoints.json`은 모든 epoch 끝 checkpoint를 기록한다.
- 기본 `SAVE_EVERY=500` update, epoch 끝 및 중단 때 저장한다. `MAX_STEPS`는 이번 job의
  실행량만 제한하며 학습 schedule을 다시 만들지 않는다.
- optimizer·epoch 내 다음 배치·global step·rank별 RNG·실행 manifest를 복원한다.
  코드/데이터/모델/환경 설정/world size가 달라지면 같은 run의 resume를 거부한다.
- 진짜 epoch가 끝난 뒤 전체 validation을 평가하며, 중단된 validation 부분 점수로
  best를 갱신하지 않는다. `completed.json`이 있어야 계획한 최대 epoch 또는
  사전 지정된 early stopping 규칙에 따라 run이 완료된 것이다.

아래 기존 분리형 run의 최종 비교 checkpoint는 각 epoch checkpoint에 대해 전체
validation 생성·채점을 끝낸 뒤 GLEU가 가장 높은 것을 선택한다. 모든 validation
조건은 같아야 하며 누락된 epoch가 있으면 선택을 거부한다. 동률은 M2 F0.5,
그다음 이른 epoch 순서로 결정한다. 위 통합 run은 학습 중 GLEU로 best를 선택하며
동률일 때 이른 epoch를 유지한다.

```bash
python -m blt_hf.select_best \
  --run-dir outputs/blt_hf/native/native-v2-s0 \
  --evaluation-dir outputs/blt_hf_eval/native/val/native-v2-s0-epoch01-beam1 \
  --evaluation-dir outputs/blt_hf_eval/native/val/native-v2-s0-epoch02-beam1
# 실제 실행에서는 epoch_checkpoints.json의 10개 epoch evaluation directory를 모두 지정한다.
```

Lang-8은 기존 제공 데이터 파일을 수정하지 않는다. `union = lang8 + korean_learner + native`의
정확한 순서와 train/val/test의 76,692/16,434/16,434행을 검사한 뒤
`data/Preprocessed/lang8/`에 다른 데이터셋처럼 split TSV/M2와 원문·교정문 파일을
만든다. train/eval/score shell은 DATASET_TYPE=lang8일 때 이를 자동으로 검사·생성하며
suffix나 M2 annotation이 다르면 실패한다.

## 3. 생성

### native guarded 통합 학습의 선택 checkpoint test 평가 (2026-09-28)

`native-gleu10-guarded-b1-s0`은 9 epoch 뒤 GLEU patience 3으로 정상 완료됐다.
최고 전체 validation GLEU 66.0768은 epoch 6의 불변 checkpoint
`step-00002310-db6b4886`이다. 이번 **test** 생성은 기존 native beam-1 결과와
같은 HF no-cache backend를 사용한다. guarded backend의 전체 출력 일치 검사는
native validation에서만 수행됐으므로 test의 동등성을 가정하지 않는다.
다음 `EVAL_DIR`은 이전 평가와 겹치지 않는 새 경로다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
export CKPT_PATH=outputs/blt_hf/native/native-gleu10-guarded-b1-s0/step-00002310-db6b4886
export EVAL_DIR=outputs/blt_hf_eval/native/test/native-gleu10-guarded-b1-s0-epoch06-beam1-hf
test -f "$CKPT_PATH/model.safetensors"
test ! -e "$EVAL_DIR/run.json"
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --time=01:55:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,CKPT_PATH="$CKPT_PATH",EVAL_DIR="$EVAL_DIR",DATASET_TYPE=native,SPLIT=test,BLT_NUM_BEAMS=1,BATCH_SIZE=1,SHARD_COUNT=1,SHARD_ID=0,GENERATION_BACKEND=hf-generate-exact-length-unpadded-v1 \
  scripts/eval_blt_hf.sh
```

생성이 시간 제한으로 exit 75이면 `test ! -e "$EVAL_DIR/run.json"` 확인은
반복하지 않고 **같은** `CKPT_PATH`, `EVAL_DIR`, beam, batch, backend로 위
`sbatch` 명령만 다시 제출한다. `shards/0000/complete.json`이 생긴 뒤에만
아래 CPU 채점을 제출한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
export EVAL_DIR=outputs/blt_hf_eval/native/test/native-gleu10-guarded-b1-s0-epoch06-beam1-hf
test -f "$EVAL_DIR/shards/0000/complete.json"
sbatch -p cpu --cpus-per-task=8 \
  --time=01:55:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,EVAL_DIR="$EVAL_DIR",DATASET_TYPE=native,SPLIT=test,M2_WORKERS=8 \
  scripts/score_blt_hf.sh
```

채점이 부분 완료·exit 75라면 같은 `EVAL_DIR`에서 아래 로컬 저널형 명령으로
재개한다. 최종 결과는 `$EVAL_DIR/scored/metrics.json`의 `status=complete`,
GLEU, M2 F0.5, 전체 행 수를 확인한다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
export EVAL_DIR=outputs/blt_hf_eval/native/test/native-gleu10-guarded-b1-s0-epoch06-beam1-hf
test -f "$EVAL_DIR/scored/gleu.json"
test -f "$EVAL_DIR/scored/m2/run_config.json"
sbatch -p cpu --cpus-per-task=8 \
  --time=01:55:00 --comment="field=nlp;appl=pytorch" \
  --export=ALL,CONDA_ENV=phdq_blt_hf,EVAL_DIR="$EVAL_DIR",DATASET_TYPE=native,SPLIT=test,M2_WORKERS=8 \
  scripts/score_blt_hf_local.sh
```

### 범용 생성 명령

학습 완료 후 validation GLEU 선택을 수행했다면 `best_gleu.json`, 그렇지 않으면
`best.json`이 가리키는 **불변 step 디렉터리**를 확인해 `CKPT_PATH`로
지정한다. 포인터가 학습 중 움직이면 다른 checkpoint의 shard를 합칠 수 없도록 실패한다.
beam 1과 4, batch 설정별로 **서로 다른 EVAL_DIR**를 사용한다.

```bash
# 아래 STEP_DIRECTORY를 best.json의 실제 checkpoint 이름으로 바꾼다.
export CKPT_PATH=outputs/blt_hf/native/native-main-01/STEP_DIRECTORY
export EVAL_DIR=outputs/blt_hf_eval/native/test/native-main-01-beam1
sbatch -p amd_a100nv_8 --gres=gpu:1 --cpus-per-task=4 \
  --export=ALL,CKPT_PATH="$CKPT_PATH",EVAL_DIR="$EVAL_DIR",DATASET_TYPE=native,BLT_NUM_BEAMS=1,BATCH_SIZE=1,SHARD_COUNT=1,SHARD_ID=0 \
  scripts/eval_blt_hf.sh

# interrupted generation: 같은 인자와 EVAL_DIR로 다시 제출하면 완료 batch부터 재개한다.
# 여러 shard가 필요할 때만 SHARD_COUNT를 먼저 고정하고 SHARD_ID=0..N-1 각각 제출.
# active/running job 제한을 지키며 앞선 job 완료 후 다음 shard를 제출한다.
```

기본은 eager/bf16, beam1, no-cache, batch1, max_new_bytes768, context2048이다.
`BLT_NUM_BEAMS=4`도 지원한다. batch를 늘리면 prompt 길이가 정확히 같은 행만 묶고
원래 문장 순서로 복원한다. padding/문서 packing/cache는 사용하지 않는다.
실측 없이 batch8의 처리량을 가정하지 않으며 A100 batch 변경은 생성 fixture로 확인한다.
source/target 절단은 없다. BOS/SEP 보존·prompt+출력 budget 한도를 검사한다.
EOS 미도달·budget 소진·잘못된 UTF-8·잘못된 token ID를 각 예측에 기록한다.
UTF-8 오류는 replacement 문자로 드러내며 해당 문장을 평가에서 제외하지 않는다.
점수용 출력은 공백을 정규화하지만 raw text와 token IDs도 보존한다.

`shards/<id>/batches/*.json`은 완성 batch 단위 불변 기록이며 출력 SHA256을 완료 보고서에
기록한다. 여러 job이 같은 shard를 쓰면 잠금으로 실패한다. shard topology·모델·data·
code·config·scorer·생성 설정 fingerprint가 다르면 재개·합산을 거부한다.

## 4. 전체 성능 평가

```bash
# 모든 shard 완료 후 CPU job에서 GLEU + M2 P/R/F0.5를 계산한다.
sbatch --export=ALL,EVAL_DIR="$EVAL_DIR",DATASET_TYPE=native,M2_WORKERS=8 \
  scripts/score_blt_hf.sh

# gleu.json과 m2/run_config.json이 생긴 부분 채점을 이어서 할 때는
# 노드 로컬 M2 저널을 사용한다. 공유 저장소에는 주기적으로 snapshot을 남긴다.
sbatch --export=ALL,EVAL_DIR="$EVAL_DIR",DATASET_TYPE=native,M2_WORKERS=8 \
  scripts/score_blt_hf_local.sh
```

scorer는 `/Users/esoterikos/Nextcloud/QLab/phdq`에서 확인한 **기존 실험 구현**을 독립
이식했다. 원본 파일·commit·SHA256·namespace 변경 diff·라이선스는
`blt_hf/vendor/provenance.json` 및 vendor 디렉터리에 있다. `blt_gec`/이전 프로젝트를
runtime으로 import하지 않으며, 새 평가에는 NumPy/SciPy도 필요 없다.
기존 GLEU wrapper의 단일 reference·6자리 반올림·100배 척도를 유지한다.
M2는 기존 NUS scorer Python3 사본, beta=.5, max unchanged words=2, case/공백 무시 옵션 false다.
빈 줄이 빠진 합본에서도 새 `S ` 행을 문장 경계로 처리하므로 union의 데이터셋 접합부를
앞 문장 annotation에 합치지 않는다.

합산 시 모든 canonical 행이 정확히 1회 있어야 한다. GLEU는 corpus 충분통계를 합산하고
M2 annotator 선택도 원래 문장 순서로 수행한다. shard별 점수를 평균하지 않는다.
GLEU는 먼저 `scored/gleu.json`에 저장한다. M2는 문장별 통계를 저널에 저장하고
30초→120초→480초→1920초 timeout pass로 재시도한다. wall-time이 오면 중단 후 재개한다.
미완료 문장이 있으면 partial/75를 반환하고 전체 F0.5를 발표하지 않는다.
완료 시 `scored/metrics.json`에 GLEU(0–100), M2 P/R/F0.5(0–1), EOS/copy율,
UTF-8 오류·budget 소진 건수 및 전체 행 수를 기록하고, 상위
`scored/progress.json`도 `complete`로 갱신한다. 이전 코드의 learner 결과는
`metrics.json`은 완전하지만 상위 progress만 `partial`로 남았으며 GSM 사본을
복구했다. 당시 실제 M2 채점 시간은 job 914870 로그의 77.65초이고
`metrics.json`의 0.055초는 완료 후 재집계 시간이다.
그 결과는 당시 scorer hash와 함께 보존한다. 현재 `main`은 union 접합부의 M2
문장 경계를 수정했으므로 이전 native/learner 평가 디렉터리에 재채점을 시도하지
않는다.

## 검증 범위

최종 단위 검사: Mac 76개 중 56 통과/torch 관련 20 skip, itcerdo 76개 전부 통과.
증거: `blt_hf_checks/results/neuron_final_checks_20260916.json`.
Mac: 분할·누락·중복·hash·재개·지표·shell/CLI 테스트.
itcerdo: 준비된 HF 환경의 단위 테스트, 실제 B의 beam1/4 생성 및 혼합 정밀도 **forward만**.
Neuron: 실제 backward/optimizer/DDP/메모리/중단 재개는 위 사용자 smoke로 확인해야 한다.
원본 BLT 수치 동질성은 학습·성능 평가의 게이트가 아니다.

구현 참고: [PyTorch 2.11 DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html),
[AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).
노드 정책의 정본은 프로젝트 `ssh.md`와 제출 시 `showque`/`showappl`이다.
