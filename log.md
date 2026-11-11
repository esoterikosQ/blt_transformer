# 일자별 작업 내역

## 2026-09-27 — guarded 전체 native 출력 일치·통합 학습 준비

- 916222의 예정된 exit 75 이후 916259가 같은 평가 identity에서
  2,634문장 생성을 완료했다. 로컬 CPU 전수 비교 11은 HF 기준과 guarded
  출력의 token ID/EOS/문자열 불일치 0건, 생성 9,074.30→6,906.76초
  (1.314배)를 확인했다.
- 새 `--validation-generation-backend`를 통합 학습에 추가했다. 기본 HF는
  유지하고 guarded는 beam 1·batch 1·val GLEU 선택에만 허용한다. run
  manifest, epoch별 GLEU 및 best 포인터에 backend·갱신 횟수를 기록한다.
- 새 native 최대 10-epoch A100 4GPU 본 학습과 같은 설정의 재개 명령을
  `NEURON.md`에 기록했다. 매 epoch 전체 validation GLEU로 best를 선택하고
  연속 3 epoch 미개선이면 중단한다. 실제 완료 epoch와 중단 사유를 기록한다.
  Neuron 제출은 사용자가 맡는다.

## 2026-09-27 — guarded 10 결과와 전체 native 비교 11 준비

- 916210 A100 결과는 44/44 HF 출력 일치, 2,505회 global skip 후보 중
  59회 full refresh. 순수 생성 합계 117.76초(v2)→124.36초(guarded),
  job 전체 159→322초. 생성 외 지연과 첫 문장 55 이상치는 원인 미확인으로
  별도 기록하고 캐시 속도 판단에서는 제외한다.
- 구버전 HF 기준 run과 새 guarded run을 비교하도록 생성 경로 세 파일의
  코드 해시 차이만 허용한다. 모델·데이터·체크포인트·나머지 코드 일치와
  batch 해시 검사는 유지한다.
- 전체 native validation 2,634건을 새 디렉터리에서 생성하고 기존 HF
  기준 출력과 순수 생성 시간·p50/p95·출력을 비교하는 11 명령을 준비했다.
  Neuron 제출은 사용자가 수행한다. 속도 이득 확인 후 통합 GLEU 학습에 연결한다.

## 2026-09-27 — v2 09 결과와 guarded 10 시험 준비

- 사용자 커밋 `99692b2`의 916208 보고서는 exit 0. v1 불일치 32건 중
  v2에서 12건 일치, 20건 불일치. 대조군 12/12 일치. 남은 20건 모두
  이전 첫 분기에서 global skip이 발생했다.
- 상위 두 byte logit 간격 1.0 이하에서만 global을 full 재계산하는
  선택형 guarded backend와 44문장 10 시험을 추가했다. 문턱은 관찰된
  logit 차이를 바탕으로 한 실험값이며 일반 동등성 보증은 아니다.

## 2026-09-27 — 08 logit projection 결과와 v2 표본 실험 준비

- 사용자 커밋 `16a890a`의 916207 보고서는 exit 0. HF식
  `logits_to_keep=1` full forward가 32/32 첫 분기 기준 ID를 반복 복구했다.
  15건은 projection 차이, 17건은 global skip 영향으로 분리했다.
- 기존 v1을 변경하지 않고 projection만 맞춘 선택형 v2를 추가했다.
  v1 불일치 32건+대조군 12건의 완성 생성 09 명령을 준비했다.
  Neuron 작업 제출은 사용자에게 남겨 둔다.

## 2026-09-27 — 첫 분기 07 결과·logit projection 08 진단

- 사용자 커밋 `4e96a6c`의 916179 보고서: exit 0, 32/32 재생. 17건은
  global skip이 직접 greedy ID를 바꿨고 15건은 full 재계산에서도 저장된
  HF ID가 나오지 않았다. 15건 중 11건은 cache/full logit이 같았다.
- HF `generate()`는 `logits_to_keep=1`, v1 수동 경로는 기본값 0인 것을
  확인했다. 08 진단에 HF식 projection 결과를 추가하고 새 보고서 명령을
  준비했다. 기존 backend와 07 결과는 변경하지 않았다.

## 2026-09-27 — native cache 전수 비교와 첫 분기 진단 준비

- 사용자 동기화 커밋 `9a483d2`의 916171 보고서를 검증했다. 2,634건 모두
  완료, 32건 token ID·문자열 불일치, EOS/UTF-8 차이 없음. exit 1은 비교
  도구의 의도된 반환. 생성 시간 합계 9,074.297초→6,620.644초(1.371배).
- 동일 접두부의 첫 분기를 32건 재생해 global skip과 full forward를 비교하는
  A100 1GPU 진단 코드를 추가했다. Neuron 접속·제출은 하지 않는다.

## 2026-09-26 — native 완성 생성 A100 915874 결과

- 사용자 동기화 커밋 `b542bd8`의 05 보고서·SLURM 로그 수신. exit 0,
  native validation 길이별 12문장의 전체 token ID·EOS·UTF-8·문자열
  12/12 일치, 모두 EOS 종료, 예산 소진 0.
- 기준 생성 43.594초→global 캐시 29.711초(1.467배). 첫 기준 문장의
  초기 비용을 빼면 약 1.432배다. 전체 split 동등성·속도는 미측정.
- 전체 2,634건을 두 backend로 별도 생성 후 지문/완료 파일·출력·생성
  시간을 전수 비교하는 CPU 도구와 Neuron 절차를 준비했다. decoder KV off.

## 2026-09-26 — 선택형 global 캐시 생성 연결

- 실용 생성 속도 개선을 위해 `global-prefix-greedy-v1`을 beam1/batch1에
  연결했다. decoder KV는 지속적인 추가 이득이 관측되지 않아 사용하지 않는다.
- 기존 HF no-cache가 기본이며 새 backend는 eval manifest와 실행 파일
  hash로 격리된다. 학습·4-beam은 변경하지 않았다.
- native validation 길이별 12문장 완성 생성의 token/EOS/문자열·시간을
  양쪽에서 측정하는 Neuron 스크립트를 준비했다. 사용자 실행 결과를 받은
  뒤 전체 validation 비교를 진행한다.
- itcerdo의 분리 checkout·기존 `phdq_blt_hf` 환경에서 작은 OSC BLT의
  HF↔global 캐시 완성 생성 및 cache reset 포함 6개 테스트 통과. 임시
  checkout 정리; Neuron 미접속·미제출.

## 2026-09-26 — Neuron A100 캐시 지연 진단 915864 결과

- 사용자 동기화 커밋 `0b91fdf`의 04 보고서와 SLURM 로그 수신. exit 0,
  24/24 다음 ID 일치, 217초. native checkpoint·validation TSV identity 동일.
- 2290 step 3의 첫 decoder 재사용만 569.42ms로 지연 재현. 후속 on/off는
  모두 약 26ms. GC 없음, 큰 메모리 증가 없음. 첫 사용 내부 원인은 미확정.
- 297 step 27은 patch 시작점 76이 index 43에 삽입돼 정상적으로 캐시를
  무효화했다. 반복 on/off는 약 43ms로 263ms 지연은 미재현.
- 캐시 분기 결함은 발견되지 않아 규칙은 변경하지 않는다. 03의 두 지연을
  정상 시간으로 대체한 약 1.47배는 제한적 추정. 선택형 beam-1 완성 생성과
  native 전체 validation parity·속도를 다음에 확인한다.

## 2026-09-26 — 3차 캐시 판정 정정 및 지연 진단 준비

- 사용자는 속도 향상이 확인된 시제품을 2배 목표 미달만으로 배제한
  에이전트의 판단을 철회하도록 지시했다. 적용 여부는 지연 원인과 전체
  생성 결과를 확인한 뒤 결정한다.
- 2290 step 3은 첫 decoder 재사용, 297 step 27은 patch 경계 변경으로
  global/decoder 재사용을 둘 다 중단한 fallback이다. 두 지연의 계산·호스트
  원인은 기존 로그만으로 확정할 수 없어 동일 접두부 교차 반복 진단을 준비했다.
- 현재 no-cache는 기본 경로에 캐시가 통합되지 않았기 때문이며 시제품
  폐기 결정이 아니다. Neuron 실행은 사용자 담당이다.

## 2026-09-26 — Neuron A100 decoder 재사용 probe 915710 결과

- 사용자 동기화 커밋 `e20b43a`의 결과·SLURM 로그를 확인했다. 동일 native
  checkpoint/data/sample/code에서 decoder KV만 켜 다음 ID 384/384 일치,
  exit 0, GPU peak 9.303GB, EOS 사례 0개였다.
- 첫 step을 제외한 forward 합계는 16.405→11.917초(1.377배)로, decoder를
  끈 02의 1.463배보다 낮았다. 578.85ms·263.19ms의 지연 두 건은 원인
  미확정이며 하나는 decoder 비재사용 step이었다.
- 전체 validation·EOS·beam4 미검증, 2배 목표 미달. 당시의 적용 배제 판단은
  위 정정으로 철회했다.

## 2026-09-26 — Neuron A100 수정 캐시 probe 915655 결과

- 사용자 동기화 커밋 `e0ed93b` 수신. 동일 checkpoint/data/sample/env의
  수정 시제품에서 다음 byte 384/384 일치, exit 0, peak 9.289GB.
- forward 합계 기준 15.659초→시제품 10.701초(1.463배). 재계산 73 step의
  역전은 평균 42.17→42.44ms로 거의 제거. 2배 채택 기준은 미달.
- 기존 no-cache 평가 유지. 다음 진단은 선택형 decoder KV를 같은 12문장·
  32 step에 적용해 추가 이득과 token parity를 본다. Neuron 작업은 사용자 실행.

## 2026-09-26 — Neuron A100 캐시 probe 915640 결과 확인

- 사용자 동기화 커밋 `ae8f45c`의 A100 native probe와 SLURM 로그를 수신.
  12문장 × 32 step에서 다음 byte ID 불일치 0, job exit 0, GPU peak 9.33GB.
  forward 합계는 기준 16.346초, 시제품 11.917초로 1.372배 개선.
- global 생략 299 step은 1.65배 빠르지만 재계산 73 step은 0.81배로 느림.
  전체 validation·EOS·beam4와 2배 채택 기준은 미충족/미검증.
- 경계 변경 step의 KV tail 경로를 제거하고 global full forward로 되돌려
  작은 OSC 및 itcerdo 합성 1B 검사를 통과. A100 재측정은 사용자 실행 대기.

## 2026-09-26 — P3a 단계 0 실측 및 global/decoder 재사용 시제품

- itcerdo의 실제 BF16 1B 추론 검사에서 접두부 확장 시 한영 혼합 입력의
  patch 경계가 5/36개 지점에서 변경됐다. 닫힌 global/decoder activation도
  길이에 따라 달라졌으며 동일 길이 반복 실행은 비트 단위 일치했다.
- 매 step 전체 entropy patcher를 유지하고 경계 일치 구간만 구조적으로
  재사용하는 실험용 global/decoder 시제품을 작성했다. 합성 greedy 96 step의
  다음 바이트 ID는 기준과 같았고 평균 시간 개선은 1.35~1.50배 범위였다.
  2배 채택 기준과 fine-tuned A100/full validation parity는 아직 미달·미검증.
- 현재 학습/평가 기본 경로는 바꾸지 않는다. 사용자가 Neuron에서 실행할
  fine-tuned A100 probe 스크립트와 `blt_hf/NEURON.md` 명령만 준비했다.
  최종 EOS 보호 반영 1B 보고서도 96/96 ID 일치, 평균 1.49~1.56배 개선.
  Neuron 미접속·미제출.

## 2026-09-26 — P3a 증분 생성 작업 착수 승인

- 사용자가 `plan/P3a_generation_cache_20260925.md`의 단계별 실행안과
  native validation 생성 시간 2배 개선 채택 기준을 승인했다.
- 기존 P3a의 P1 종료·cache probe·batch 처리량 트리거를 이 착수에서는
  사용자 결정으로 대체한다. 단계 0에서 실제 OSC 패치 경계의 접두부 안정성과
  encoder/global/decoder 상태의 재사용 범위를 확인한다.
- 구현은 기존 BF16/eager/no-cache 학습·평가를 보존하는 선택형 backend로
  진행한다. Neuron 실행은 사용자가 담당한다.

## 2026-09-16 — Neuron 제출 구조를 직접 sbatch 방식으로 교정

- 실제 사용 이력이 있는 BART job script를 기준으로 train/eval/score 맨 앞에
  `#!/bin/bash`, field/appl comment, stdout/stderr, partition, node/task, 자원, wall time,
  `B:TERM@300` SBATCH header를 완성했다.
- 로그인 shell에서 Bash wrapper를 실행하는 `submit_blt_hf.sh`를 삭제했다. 사용자는
  프로젝트 root에서 각 batch script를 `sbatch --export=...`로 직접 제출한다.
- 할당된 job 안에서는 Python 및 torchrun을 `srun --ntasks=1`로 실행한다.
  `SLURM_SUBMIT_DIR`, comment, partition, CPU/GPU 조건을 compute job에서 재검사하고
  job/node/CUDA/자원 정보를 로그에 출력한다.
- shell syntax 및 SBATCH/srun 구조 회귀 테스트 5개 통과. 실제 Neuron 제출은 사용자 실행.

## 2026-09-16 — Neuron 제출 helper 수정

- 프로젝트 field를 `nlp` 기본값으로 고정하고 실제 comment 인자를
  `field=nlp;appl=pytorch` 한 개의 shell argument로 전달하도록 수정.
- `showque`/`showappl`이 정보를 출력한 뒤 nonzero를 반환해도 `set -e`로 제출이
  조기 종료되지 않도록 경고 처리. 제출 직전 mode/partition/CPU/GPU를 출력한다.
- 두 정보 명령이 각각 nonzero인 모의 Neuron 환경에서도 `sbatch` 도달 및 전체 인자를
  검사하는 회귀 테스트 추가. 로컬 전체 77개 중 57 통과/torch 관련 20 skip.

## 2026-09-16 — GitHub 공개 범위 정리

- `data/` 전체, artifacts/outputs/환경 캐시·가중치·인증정보·ssh.md를 Git에서 제외.
- 원문이 포함된 GLEU fixture 및 patch 보고서, Meta 검토용 원본 사본을 제외.
- GPL 표기가 상충하는 M2 소스·예제·원문 diff는 로컬 보존하고 공개본에서 제외.
  공개 코드에서 실행할 때 필요한 정확한 사본·hash·준비 절차는 THIRD_PARTY_NOTICES.md에 기록.
- 배포 허용 고지가 있는 HF/GLEU 파일의 저작권·라이선스와 변경 기록 유지.
- 공개 파일만 추출한 디렉터리에서 76개 검사 중 52 통과/의존성·비공개 입력 관련 24 skip.
  원본 모델 runtime 및 기존 검증 JSON은 변경하지 않았다.


## 2026-09-16 — Neuron 사용자 실행용 학습·성능 평가 구현

- ssh.md의 scratch root·A100 GPU/CPU 비율·showque/showappl·field/appl comment를 반영한
  학습/생성/CPU 채점/제출 스크립트 구현. neuron 미접속·미제출.
- DDP 누적·무패딩 학습·FP32 optimizer 상태·entropy 고정·전체 validation·checkpoint/RNG 재개.
- beam1/4 생성, 불변 batch 기록·shard 합산·전체 corpus GLEU/M2·timeout/중단 재개 구현.
- 사용자 지정 이전 프로젝트에서 GLEU/M2 scorer 확보 및 독립 이식. 이전 GLEU 4개,
  M2 2개 fixture 정확히 일치. 원본 파일/데이터/동결 패키지는 수정하지 않음.
- 로컬 76개 중 56 통과/20 skip, itcerdo HF 환경 최종 코드 76개 통과. 실제 B beam1/4 및
  FP32 parameter/bf16 autocast forward 통과. backward/optimizer는 실행하지 않음.
- Neuron 실제 학습 검사는 사용자 smoke에서 수행. 실행 문서: blt_hf/NEURON.md.

## 2026-09-16 — 검증 범위 변경 및 실제 B 전 계층 마스크 검사 완료

- 사용자 결정에 따라 P1/CLAUDE에서 전체 동질성 게이트·미확인 시 결과 폐기를 제거.
  변환 관련 검사는 가중치 보존·attention mask로 한정, 학습/평가 유효성 테스트 유지.
- 기존 515→510 텐서 전수 보존 증거 재사용, B 출력 파일 hash 및 strict load 재확인.
- itcerdo tmux에서 실제 B의 자동 패칭 6개 fixture × 59개 attention 모듈,
  강제 가상/0길이 patch fixture × 45개 모듈 검사 통과. window/EOS/causal/cross
  전체 마스크 및 미래 byte 연결 검사. 전체 단위 테스트 54개 통과.
- 보고서 `blt_hf_checks/results/p1_masks_20260916.json`, 실행 exit_code=0.
  가중치·mask passed, 학습·평가 not_run. eager/bf16/no-cache/무패딩 범위.
- neuron 미접속, backward/optimizer 미실행, 데이터·동결 패키지 미수정.
  아래 과거 기록의 동질성 pending은 당시 상태이며 현재 진행 조건이 아니다.

## 2026-09-15 — B 전수 검증 및 OSC 후보 GPU 구동

- 원본 515개 → B 510개 tensor 전수 값/dtype·coverage 검증 통과, 실제 strict load 통과.
- eager/SDPA 둘 다 forward/generate 가능하지만 영어 greedy IDs 불일치. 기준 eager 고정.
- 프로젝트 사본에 entropy/local window+EOS 및 global EOS 마스크, next-token 패치 슬롯,
  원본 entropy 산술을 적용. site-packages·동결 데이터·blt_gec는 변경하지 않음.
- bf16 threshold 반올림의 검증 규칙을 명시하고 실패/실수 기준 차이를 보존.
  독립 scalar 대비 20/20 패치 경계 일치. 실제 1,817-token forward와 생성 정상.
- 전체 동질성은 pending. neuron 접속·backward·학습은 미실행. 자세한 범위는 NOTES.md.

## 2026-09-15 — itcerdo 원본 확보·CPU 변환 완료, tmux 인계

- 사용자 제공 프로젝트 HF 토큰으로 main/entropy 고정 revision 다운로드 완료.
- 실제 HF 환경에서 단위 테스트 35개 통과. 별도/내장 entropy 129개 텐서 전수 동일.
- 직접 변환본 B 생성 완료: 510개 bf16 텐서. 상태 `converted_unvalidated`, 동질성 pending.
- 사용자 이동 요청에 따라 `phdq3-p1` tmux로 전환. CPU 비교·변환 exit_code=0 확인.
  세션은 shell 대기 중이며 추가 실행 중인 job은 없다. 명령/로그 위치는 blt_hf/COMMANDS.md.
- neuron 접속·학습은 미수행. 다음은 B 정적 검증·strict load·OSC 구동 구현.

## 2026-09-15 — P1 로컬 구현 착수

- 독립 blt_hf 데이터 어댑터·실행 manifest, 환경/데이터 점검 CLI 구현.
- Python 표준 라이브러리 unittest 25개 통과. 실제 HF/GPU 실행 검증은 미수행.
- 9개 split 재검사: 201,534행, 최대 입력 1,380, 2,048 초과 0건; TSV–M2 정합성 확인.
- 사용자 지시: neuron 접속·원격 작업 금지. itcerdo 정보 파일 수신 후 원격 단계 진행.
- 현 상태/명령/다음 단계는 blt_hf/NOTES.md, blt_hf/COMMANDS.md 참조.
## 2026-09-24 — native 2-epoch 통합 GLEU 시험 준비

- 사용자 요청: A100 4GPU 학습 job에서 매 epoch 전체 validation 생성·GLEU를 함께
  수행하고 같은 GLEU로 checkpoint를 선택하는 native 2-epoch 시험.
- 학습 역전파는 token loss, checkpoint 선택은 validation corpus GLEU로 분리해
  기록한다. 이전 run과 충돌하지 않는 새 RUN_ID를 사용한다.
- 첫 시험은 beam1, 2 epoch로 통합 소요시간·GPU 메모리를 측정한다.
  beam4는 실측 후 별도 run으로 진행한다. Neuron 접속·제출은 사용자 담당.
