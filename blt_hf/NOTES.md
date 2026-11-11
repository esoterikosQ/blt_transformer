# P1 구현·판정 기록

## 2026-09-26 — P3a 캐시 단계 0·실험용 단계 1

실행 정본은 `plan/P3a_generation_cache_20260925.md`, 세부 측정과 보고서 경로는
`cache/DESIGN.md`다. itcerdo의 실제 변환 1B, BF16/eager/OSC에서
학습·가중치 변경 없이 검사했다. `patched/modeling_blt.py`와 site-packages는
편집하지 않았고, 원본 데이터·Neuron 작업도 건드리지 않았다.

- 512바이트 window/EOS를 포함한 다섯 fixture에서 한영 혼합 입력만
  36개 중 5개 접두부의 patch 경계가 바뀌었다. 전체 patcher를 매 step
  돌리는 것이 현재 no-cache 경계와 맞는다.
- 닫힌 patch의 global과 decoder 과거 activation은 입력 길이를 늘리면
  15/15 비교에서 달라졌다. 같은 길이로 다시 실행하면 entropy·encoder·
  global·decoder·logits 모두 15/15 비트 단위 일치했다.
- global 출력 교체 민감도 96 step, global 계산 생략 시제품 96 step,
  global+decoder 재사용 시제품 96 step의 다음 바이트 ID 불일치는 각각
  0개였다. logits 수치는 달라지므로 모델 상태의 정확한 동질성을 주장하지
  않는다. 사전학습 5090 합성 입력의 평균 속도 개선은 global 생략
  1.35~1.49배, decoder 병행 1.43~1.50배였다. 5090 공유 GPU의 절대 시간은
  실행 사이 크게 변했다.
- 최종 EOS 보호를 반영한 `p3a_global_final_20260926.json`도 96/96 다음
  byte ID 일치, 평균 개선 1.49~1.56배였다. 작은 OSC 모델의 연속 접두부·
  EOS·reset 테스트 5개를 itcerdo에서 통과했다.
- `cache/global_reuse.py`는 기존 HF generate/eval에 연결하지 않은 진단
  시제품이다. 무패딩 batch1/beam1의 연속 접두부만 지원한다. 매 step 전체
  patcher·local encoder를 재계산하고 경계 변동 시 global/decoder 꼬리를
  무효화한다. 실제 fine-tuned native 체크포인트, A100, validation 문장,
  beam4 및 전체 split의 토큰 동일성과 2배 속도 목표는 아직 미검증이다.

현재 no-cache 학습·평가는 그대로 유지한다. 사용자가 실행할 A100 1GPU
probe 스크립트는 `scripts/bench_blt_cache.sh`이며 명령은 COMMANDS.md에 있다.

## 2026-09-26 — Neuron A100 캐시 probe 915640 분석

사용자가 GitHub에 동기화한 `p3a_native_cache_probe_01.json`과 SLURM 로그를
확인했다. native fine-tuned step-00001155, A100-SXM4-80GB, BF16,
환경 검사 통과, exit 0, 208초. validation 12문장 × 32 step의 다음 byte ID는
384/384 같았다. 초기 step을 제외한 forward 측정 합계는 기준 16.346초,
시제품 11.917초로 1.372배다. GPU peak allocated 9.33GB.

global skip 299개에서는 43.94→26.59ms지만, 경계 변경 등 재계산 73개에서는
43.92→54.35ms로 역전됐다. 한 문장은 skip 10/32개라 전체 0.86배였다.
전체 생성·EOS·beam4 검사는 아직 아니다. 2배 채택 기준을 통과하지 않았고
현재 평가 backend를 변경하지 않는다. 재계산 구간의 KV tail 경로를 제거하고
기준 global full forward로 되돌려 작은 OSC와 itcerdo 1B 테스트를 통과했다.
당시 이 수정 후 Neuron A100 재측정이 필요했으며, 결과는 다음 절에 기록했다.

## 2026-09-26 — Neuron A100 수정 probe 915655 분석

동일한 native checkpoint·validation TSV·12개 sample·환경에서 코드 hash만
경계 변경 fallback 수정본으로 바뀌었다. job exit 0, 148초, GPU peak
9.289GB. 다음 byte ID 384/384 일치, EOS 사례 0. 초기 step을 제외한
forward 합계는 기준 15.659초, 수정본 10.701초로 1.463배 개선이다.
global 생략 299 step은 42.08→25.43ms, 재계산 73 step은
42.17→42.44ms로 이전의 큰 역전이 사라졌다. 한 문장도 기준보다 느리지
않았지만 2배 채택 기준은 여전히 미달. 전체 문장 완성·beam4·GLEU
평가에 시제품을 투입하지 않는다. 다음 제한적 probe는 같은 조건에서
`REUSE_DECODER=1`을 비교하는 것이었으며, 결과는 다음 절에 기록했다.

## 2026-09-26 — Neuron A100 decoder 재사용 probe 915710 분석

사용자가 동기화한 `p3a_native_cache_probe_03_decoder.json`을 확인했다.
02와 checkpoint·validation TSV·12개 sample·model/reuse/bench 코드 hash와
환경이 같고, decoder 재사용 옵션만 켰다. job exit 0, GPU peak 9.303GB,
다음 byte ID 384/384 일치, global skip/decoder reuse 각 299 step, EOS 0개다.
forward 합계는 기준 16.405초, 시제품 11.917초(1.377배)로 02의 1.463배보다
낮다. 재사용 구간 299 step은 평균 44.07→28.17ms, 비재사용 73 step은
44.21→47.85ms였다. 두 step에 578.85ms와 263.19ms의 지연이 기록됐는데
후자는 decoder를 재사용하지 않은 step이므로 decoder 결함으로 단정하지
않는다. 이 지연을 제외하면 약 1.47배다. 전체 생성·EOS·beam4·GLEU는
미검증이다. 이전의 2배 목표만으로 캐시를 배제한 결론은 철회한다.
297 step 27은 patch 시작점 변경 때문에 global/decoder 상태가 무효화된
것으로 분기 코드를 통해 확인했다. 다만 두 지연의 원인은 기존 벤치의
CUDA 이벤트만으로 특정할 수 없다. 같은 접두부를 decoder on/off로 반복하고
모듈/호스트/GC 시간을 기록하는 진단을 추가했다. 결과를 받아 원인을
확인하고 필요하면 수정한다. 기본 no-cache는 통합 backend가 아직 없어서
그대로이며, 캐시 적용 여부는 이번 두 지연과 전체 생성 검증 후 결정한다.

## 2026-09-26 — Neuron A100 지연 반복 진단 915864

사용자 동기화 보고서의 24/24 다음 ID 일치, exit 0을 확인했다. 2290 step 3은
첫 decoder 재사용 때 569.42ms로 지연이 재현됐지만 이후 같은 접두부는
decoder on/off 모두 약 26ms였다. GC는 없고 대규모 GPU 메모리 증가도 없어
첫 사용 비용으로 추정한다. 내부 kernel/라이브러리 단위 원인은 아직 미확인이다.
297 step 27은 patch index 43에 시작점 76이 생겨 캐시를 무효화한 것이며,
반복 시 decoder on 약 43.6ms/off 약 43.2ms로 이전 263ms가 재현되지
않았다. 경계 분기 오류는 발견되지 않았다. 03의 두 지연을 정상 반복 시간으로
대체한 약 1.47배는 추정일 뿐 전체 생성 측정값이 아니다. 재사용 규칙은
변경하지 않고, 이후 선택형 완성 생성·전체 native validation 출력/속도를
비교한다. 2배 목표 미달만으로 적용을 배제하지 않는다.

## 2026-09-26 — global 전용 캐시를 선택형 beam-1 생성에 연결

실제 생성 속도 개선을 위해 `GlobalPrefixReuse(reuse_decoder=False)`를
`generation.py`의 별도 backend `global-prefix-greedy-v1`에 연결했다.
decoder KV는 첫 사용 비용 이후 지속 이득이 관측되지 않아 이 경로에서
사용하지 않는다. batch 1·beam 1로 제한하고 byte 억제 및 decode 정책을
기준 생성과 공유한다. `eval.py`의 CLI·manifest와 평가 실행 파일 hash에
cache 코드를 포함시켰다. 기존 no-cache backend가 기본이며 4-beam 및
학습 중 validation은 바꾸지 않았다. Neuron에서 완성 생성 parity와 속도가
확인되기 전까지 새 backend의 동등성을 주장하지 않는다.
itcerdo의 분리 checkout에서 기존 `phdq_blt_hf` 환경으로
`PYTHONPATH=.:tests python -m unittest tests.test_hf_generation_runtime
tests.test_hf_cache_reuse`를 실행해 작은 실제 OSC BLT 모델의 완성 생성
HF↔global 캐시 결과와 cache reset을 포함한 6개 테스트가 통과했다. 분리
checkout은 정리했고 기존 itcerdo 작업 파일은 변경하지 않았다. 이 테스트는
fine-tuned A100 전체 생성 검증을 대신하지 않는다.

## 2026-09-26 — native 완성 생성 A100 915874

선택형 global 캐시(행당 beam1/batch1, decoder KV off)의 길이별 native
validation 12문장 완성 생성에서 HF 기준과 token ID·EOS·UTF-8·최종 문자열이
12/12 같았다. 모두 EOS 종료, 한도 소진 0. 기준 43.594초→캐시 29.711초
(1.467배), 첫 기준 문장의 초기 비용을 빼면 약 1.432배다. job exit 0.
전체 split 속도나 동등성은 아직 미확인이다. 두 평가 backend의 결과를
별도 EVAL_DIR에 만들고 완료 파일/지문을 검증해 전체 2,634건의 출력과
생성 시간을 전수 비교하는 CPU 도구를 추가했다. 채점은 이 비교에 필요 없다.

## 2026-09-27 — native 전체 생성 비교 916171

같은 checkpoint·split·beam1·batch1·BF16·no-cache 조건의 2,634건을
전수 대조했다. 불변 batch 지문과 전체 완료 파일은 정상. token ID·문자열
불일치 32건(1.21%), EOS·UTF-8 상태 차이 0건. 생성 시간 합계는
9,074.297초→6,620.644초(1.371배)였다. 비교 job exit 1은 도구가
불일치를 감지한 의도된 반환값이다. 캐시가 patch 시작점 동일 조건에서
global 결과를 건너뛴 것이 유력한 원인이나 아직 증명되지 않았다. 첫 분기
step의 재생·global 재계산 결과를 확인하는 Neuron 진단 도구와 명령을 추가했다.

## 2026-09-27 — 첫 분기 재생 916179

32/32 첫 분기가 재현됐다. 17건은 cache skip 뒤의 ID와 full 재계산 ID가
달랐고 full 쪽이 저장된 HF 기준과 일치했다. 15건은 full 재계산도 HF
기준과 달랐다. 그중 11건은 cache/full의 모든 유한 logit 차이가 0이다.
HF `generate`는 이 버전에서 `logits_to_keep=1`을 전달하고 v1 수동 경로는
기본값 0을 사용한다. 해당 차이를 분리하는 08 진단을 추가했다. 07 보고서는
변경하지 않는다.

## 2026-09-27 — 08 logit 모양 확인·선택형 v2

916207 A100 exit 0, 32/32의 저장된 HF 기준 ID를 `logits_to_keep=1`
full forward가 반복해서 복구했다. v1의 15건은 수동 forward가 전체 길이
logit을 계산한 차이, 17건은 global skip 효과다. 두 원인이 구분됐다.
기존 v1을 보존하고 `global-prefix-greedy-v2`는 **마지막 위치 logit만**
계산한다. patch/global/decoder 재사용 규칙과 checkpoint는 그대로다.
09는 v1 불일치 32건+기존 대조군 12건을 재생한다. 이를 전체 split의
동등성 또는 속도 확인으로 간주하지 않는다.

## 2026-09-27 — 09 v2 결과·선택형 guarded 경로

916208 A100 exit 0, v1 불일치 32건 중 v2에서 12건 일치, 20건 불일치.
대조군 12/12 일치. 남은 20건은 모두 07에서 global skip이 일어난 행이다.
별도 `global-prefix-guarded-greedy-v1`은 skip 직후 상위 두 합법 byte
logit 간격이 1.0 이하일 때 full reference-style forward로 재계산한다.
이는 07의 최대 관찰 logit drift 0.421875를 바탕으로 한 실험 문턱이며
완전 동등성 보증이 아니다. v1/v2 결과를 덮어쓰지 않는다. 10에서
기존 44문장 출력을 비교하고 fallback 횟수·생성 시간을 측정한다.

916210의 10 보고서는 44/44 token ID·EOS·문자열 일치, global skip 후보
2,505회 중 59회 refresh였다. 생성 합계는 09의 117.76초 대비 124.36초,
job 전체 시간은 159초 대비 322초였다. 환경 보고서 생성 시각부터 데이터
보고서 생성 시각까지 약 10→119초로 늘었지만, 환경 검사·전체 데이터
스캔·보고서 기록이 섞인 구간이라 병목은 특정되지 않았다. 첫 문장 55는
3.90→11.29초로 늘었고 refresh 5회가 있었으나 첫 추론 초기화/노드 상태와
분리되지 않았다. 이 지연은 후순위로 기록하고 전체 validation 생성
비교 11에서 순수 생성 속도와 전수 출력 일치를 확인한다.

기존 HF 기준 run은 구버전 생성 코드의 해시를 갖는다. 11의 비교기는
`generation.py`, `eval.py`, `cache/global_reuse.py`의 버전 차이만 허용하고,
모델·데이터·체크포인트·나머지 코드 해시 및 batch 무결성을 검사한다.
속도 이득이 확인되면 현재 통합 학습의 epoch별 GLEU 생성 경로에 선택형
backend를 연결하고 새 run에서만 학습한다.

## 2026-09-27 — 11 전수 guarded 결과와 통합 학습 연결

916222는 2,318/2,634에서 예정된 시간 제한으로 exit 75, 916259는 같은
identity에서 재개해 2,634/2,634 완료·exit 0이었다. 로컬
`compare_cache_eval.py`의 `p3a_native_full_guarded_11.json`은 양쪽
불변 batch 해시·지문을 확인한 뒤 HF 대비 token ID/EOS/문자열 불일치
0건을 기록했다. 순수 생성 시간은 HF 9,074.30초→guarded 6,906.76초,
1.314배(23.9% 시간 절감)이다. global skip 후보 150,352회 중
1,792회 full refresh였다. 10의 준비 시간 이상치는 별도 관측 사항으로 둔다.

새 통합 학습 옵션 `--validation-generation-backend`는 HF를 기본으로,
guarded를 beam 1·batch 1·GLEU 선택에만 허용한다. run manifest/code hash,
epoch별 GLEU·cache refresh 보고서와 best 포인터에 backend를 기록한다.
캐시 구현·가중치·학습 loss·GLEU scorer는 변경하지 않았다. 새 RUN_ID의
최대 10-epoch A100 4GPU 본 학습·재개 명령은 `NEURON.md`에 있다. 모든 완료
epoch의 전체 validation GLEU로 best를 고르며, 연속 3 epoch 미개선 시 중단한다.
실제 완료 epoch와 중단 사유를 남긴다.

## 2026-09-17 — FP32 학습 정책 폐기, BF16 레거시 조건 복구

변환 artifact B는 원본과 동일한 BF16이지만 초기 학습 코드가 근거 없이 main을 FP32로
승격했다. 이는 레거시 BF16 학습 조건과 다르고 hash embedding을 포함한 학습 상태를 거의
두 배로 늘렸다. `model.float()`를 제거하고 parameter·gradient·Adam `exp_avg`/
`exp_avg_sq`·연산을 BF16으로 고정했으며 실행 중 dtype을 검사한다.

- job 909747과 910019의 OOM은 폐기된 FP32 정책에서 발생했다. 로그는 실패 원인 기록으로만
  보존하며 BF16 backward/optimizer/DDP 검증으로 인정하지 않는다.
- 변환 가중치·마스크 증거는 BF16 artifact를 대상으로 했으므로 그대로 유효하다.
- 환경 검사에 H200/H100의 compute capability sm_90을 추가했다. A100 sm_80과 itcerdo
  검증용 RTX 5090 sm_120도 유지하며 capability뿐 아니라 native BF16 지원과 실제 BF16
  matmul을 검사한다. BF16 미지원 V100 sm_70은 계속 거부한다.
- Neuron의 실제 제출은 `ssh.md`에 확인된 A100/H200 partition만 사용한다. 새 RUN_ID로
  1 GPU와 2 GPU smoke를 다시 실행하기 전까지 BF16 학습 검사는 `not_run`이다.

## 2026-09-16 — Neuron 학습·생성·평가 코드 구현

실행 정본은 `NEURON.md`다. 에이전트는 neuron에 접속·전송·제출하지 않았다.

- 당시 `train.py`: FP32 main/Adam + bf16 autocast를 사용했으나 위 2026-09-17 결정으로 폐기.
  현재는 BF16 main/gradient/Adam/연산, 고정 entropy, gradient checkpointing,
  무패딩 microbatch1, 전역 token 평균 loss, DDP/no_sync, 마지막 배치 누락·중복 방지.
  rank별 RNG/optimizer/epoch 배치 위치를 불변 checkpoint directory에 저장한다.
  전체 validation loss로 best를 선택하며 중단된 validation의 부분 점수는 채택하지 않는다.
- `generation.py`, `eval.py`: beam1/4, 동일길이 무패딩 묶음, 원래 순서 복원,
  batch 단위 원자적 기록/재개, 전체 split의 정확한 coverage·fingerprint·출력 hash 검사.
  source/target 절단이나 평가 행 제외 없이 UTF-8 오류·EOS/budget 상태도 기록한다.
- 기존 프로젝트 `/Users/esoterikos/Nextcloud/QLab/phdq`에서 scorer를 읽어 독립 이식.
  GLEU wrapper 4개 fixture, M2 CLI 2개 fixture가 이전 구현과 정확히 일치했다.
  `results/metrics_legacy_20260916.json` 및 `vendor/provenance.json`에 출처/증거를 보존.
  기존 GLEU의 소수점 6자리 반올림 후 100배 척도도 유지한다.
- M2는 CPU job에서 충분통계 저널·timeout retry·중단 재개를 사용한다.
  미완료 문장이 있으면 최종 corpus F0.5를 출력하지 않는다. `gleu.json`은 먼저 보존한다.
- 제출 helper는 ssh.md 기준 A100 partition, GPU/CPU 비율, active job 제한, showque/showappl,
  새 field/appl comment를 적용한다. array/자동 재제출/자동 환경 변경은 하지 않는다.
- **실행 증거**: Mac 76개 중 56 통과/torch 필요 20 skip. itcerdo HF 환경에서는
  최종 코드 76개 전부 통과 (`neuron_final_checks_20260916.json`). 실제 B의 8-token beam1/4 생성과 mixed-length 입력의 동일길이 묶음
  결과가 단건 실행과 같았다. FP32 parameter + bf16 autocast forward도 유한 loss.
  `p1_neuron_code_v3_20260916.json`, peak allocated 24,505,433,600 bytes.
- 첫 config hash 검사는 loader의 `_name_or_path`와 명시적 dtype 적용으로 실패했다.
  경로 메타데이터만 제외하고 factory에 loader dtype을 명시해 해결했다.
  dtype/window/patcher 등 실행 설정은 fingerprint에 남긴다. v1/v2 실패 보고서도 보존했다.
- **미실행**: 실제 backward/optimizer/DDP 및 A100 메모리·시간·전체 평가.
  `submit_blt_hf.sh smoke`는 작은 BLT backward/재개 검사 후 실제 B 긴 train 입력을
  2 update 처리한다. overfit도 별도 진단이며 본 평가 checkpoint로 사용하지 못한다.
  코드 준비 완료를 A100 학습 검증이나 GEC 성능 확인으로 해석하지 않는다.

## 2026-09-16 — 가중치·attention mask 검사 완료 (현재 기준)

사용자 지시에 따라 `plan/P1.md`와 `CLAUDE.md`에서 전체 수치 동질성 게이트를
제거했다. 원본 runtime 재구축·독립 entropy forward·계층 logits parity·C 비교는
필수 작업이 아니다. 연구 주장은 고정된 Meta 가중치의 HF 변환 및 공개한 구현
조건에서의 GEC 성능이다. 과거 `pending/confirmed/not_confirmed`는 당시 정책의
이력이며 아래 과거 절의 동질성 대기·결과 폐기 규칙은 현재 적용하지 않는다.

- 새 보고서: `blt_hf_checks/results/p1_masks_20260916.json`.
  `weight_checks=passed`, `attention_mask_checks=passed`,
  `training_checks=not_run`, `evaluation_checks=not_run`.
- 가중치: 원본 515개→B 510개 전수 exact 값·dtype·shape·coverage 검사 통과
  보고서를 재사용했다. 이번 실행에서 B의 모든 출력 파일 SHA256을 재대조하고
  실제 patched 모델 strict load도 확인했다. 원본 텐서 비교를 다시 수행한 것은 아니다.
- 실제 B의 자동 패칭 6개 fixture에서 각각 **59개 attention 모듈** 검사:
  entropy self 14, encoder self 1/cross 1, global self 25, decoder self 9/cross 9.
  EOS/연속 EOS/행별 경계가 다른 2행 배치, 길이 511/512/513/1380/2048 포함.
- 강제 가상·0길이 patch fixture는 entropy를 우회하므로 **45개 모듈**을 검사했다.
  모든 계층의 전체 additive mask가 독립 규칙과 일치했다. 빈 encoder patch가
  모든 byte를 읽는다고 보수적으로 가정해도 활성 decoder 예측으로 미래 byte가
  전달되지 않음을 patch 연결 검사로 확인했다. 원본 forward 수치 비교는 아니다.
- 허용 범위: `attention_mode=osc`, eager, bf16, no-cache, 무패딩.
  `osc|lre`는 기존 코드 호환용 실행 식별자이며 원본/legacy 동등성 인증이 아니다.
  EOS self-attention 검사만으로 hash/pooling을 포함한 전체 문서 격리를 주장하지
  않는다. GEC 샘플 packing은 하지 않으며 padding/cache 지원은 별도 검사 후 추가한다.
- itcerdo 전체 단위 테스트 **54개 통과**, backward/optimizer 실행 없음.
  GPU peak allocated **11,037,369,344 bytes** (약 11.04GB; 검사 tensor 포함).
  tmux `phdq3-p1:masks-0916` 작업은 exit_code=0으로 완료했고 shell 대기 상태다.
- neuron 접속·전송·제출은 하지 않았다. 다음 단계는 neuron 사용자 실행용
  학습 루프·메모리/backward smoke·checkpoint 및 생성/평가 경로 구현이다.
  학습·평가 성능 검증이나 P1 전체 완료로 확대 해석하지 않는다.

## 2026-09-15 — 정적 검증·GPU 구동·OSC 후보 구현 (당시 기록)

현재 단계: B 가중치 전수 검증 및 실제 GPU 구동 완료. **전체 동질성 pending**,
neuron 학습 미실행. 이전 절의 미완료 목록 중 이번에 완료한 범위는 다음과 같다.

- `weights_p1_validation_20260915.json`: 원본 main+선택 entropy 515개 텐서 →
  B 510개. hash embedding 6개를 한 텐서로 합친 차이이며, 전수 값/dtype 동일,
  누락·추가·충돌 0건. HF meta 모델 510개 키·shape도 일치. 검사는 converter를
  import하지 않고 독립 경로 변환 및 fused row slice 비교를 수행했다.
  대응표는 `manifests/mapping_p1_validation_20260915.json`에 보존했다.
- 실제 eager/SDPA `from_pretrained`에서 missing/unexpected/mismatched/error 모두 0.
  파라미터 수는 **4,633,391,872**이며 hash embedding 포함 수치다. bf16 가중치 약
  9.27GB. GPU peak allocated 약 9.28GB. 한 GEC 예제의 HF loss와 수동 shift 1회
  cross entropy가 정확히 일치했다. backward/optimizer는 실행하지 않았다.
- eager/SDPA의 영어 greedy IDs는 불일치, 한국어 사례는 일치. **기준 backend는
  eager로 결정**. SDPA 동질성은 통과 처리하지 않는다. 64-byte 생성 한도에서
  한국어 마지막 글자의 UTF-8 바이트가 잘린 사례가 있어 `valid_utf8=false`로 남겼다.
  모델 구동 성공과 출력 UTF-8/교정 품질의 성공을 구분한다.

### OSC 후보 변경과 검증 규칙

설치된 transformers 5.16.1의 사본을 `blt_hf/patched/modeling_blt.py`로 두었다.
site-packages는 변경하지 않았다. 원본 대비 전체 diff는
`blt_hf_checks/manifests/modeling_blt_osc.diff`이며 재현 명령은 COMMANDS.md에 있다.
`blt_hf/model.py`가 attention_mode와 backend를 명시해 로드하고 원본 B hash를 확인한다.

1. EOS는 자신이 닫는 구간에 속한다. `segment(i)=Σ[j<i](token[j]==EOS)`.
   local/entropy 허용 조건은 `k≤q`, `q-k<512`, `segment(q)==segment(k)`의 교집합이다.
   global은 EOS를 포함한 patch에서 구간을 닫고 같은 구간의 과거 patch만 허용한다.
   이 정의는 pinned Meta `model/utils.py`, `model/blt.py`에 근거한다.
2. 로컬 encoder/decoder 및 entropy 모든 layer의 실제 전달 마스크를 hook으로 검사했다.
   작은 인공 입력에서 미래/window/EOS/batch 분리의 독립 scalar 정의와 일치한다.
   global EOS patch 분리도 검사했다. 무패딩 동일길이 batch는 축소 모델에서 단건과
   logits 허용오차 내 일치했다. **padding·cache 입력은 OSC 경로에서 명시적으로 거부**한다.
3. 고정 패치 경계·512 미만·중간 EOS 없음 조건에서 OSC와 full-causal logits가
   정확히 일치한다(축소 fp32 모델). 수정 사본의 lre 경로와 무수정 HF도 정확히 일치한다.
   이는 통제된 mask/사본 검사이며, 원본 스택 전체 forward parity의 증거는 아니다.
4. 원본 main은 `include_next_token=True`다. 패치 시작점은 `{0,1}` 및
   `i>0`에서 `H_i>threshold`인 `i+1`; 마지막 끝점은 `S+1`이다.
   HF 기본 경로의 끝점 `S`를 OSC 후보에서 `S+1`로 보완했다.
   `blt_hf/patching.py` tensor 구현을 독립 `plain_patcher.py` scalar 구현과 비교한다.
5. 원본 entropy 산술은 같은 dtype에서 `logp=log_softmax(logits)`,
   `H=-sum(exp(logp)*logp)`다. HF Categorical 경로와 bf16 연산 결과가 달라
   OSC에 원본 순서를 적용했다. `bytelatent/transformer.py`는 logits를 반환할 때
   fp32로 올리지 않는다. 원본 소스와 보충 hash는 `entropy_source_review.json`에 있다.
6. **검증 규칙 재검토**: 원본 tensor–Python scalar 비교는 threshold도 tensor dtype으로
   변환한다. 설정값 1.335442066192627의 bf16 유효값은 1.3359375다.
   최초 이상적인 실수 비교에서는 20개 중 4개 경계가 불일치했다(v4 실패 보고서 보존).
   이를 tolerance로 통과시키지 않고, scalar 기준에 유효 threshold를 명시했다.
   v5는 실제 threshold, 유효 threshold, 실수 기준 경계와의 차이를 모두 보존하며
   원본 dtype 비교 기준에서 **20/20 일치**한다. 이 변경은 threshold 근처의 동작을
   바꾸는 runtime 수정이 아니라 source 비교 의미를 reference에 반영한 것이다.

`p1_osc_v5_20260915.json`: 실제 B에서 71·1,817 token forward/finite loss 및 64-token
생성 정상. peak allocated 9,848,814,592 bytes. 패치 보고서의 scope는 **동일 logits의
entropy 산술+경계**이며 원본 entropy 신경망 forward 동질성을 포함하지 않는다.
원격 전체 49개 테스트 통과 후 추가 동일길이 batch 테스트를 포함한 모델 4개 테스트도
통과했다(현재 테스트 항목 합계 50). 로컬은 50개 중 34개 통과, HF/torch 필요 16개 skip.

남은 실행 준비: 실제 B의 패딩/다양한 길이 생성 정책, neuron backward·학습 루프 및
메모리 smoke. 동질성 검증은 entropy 독립 forward, component parity, legacy provenance,
C 비교, A100 결과를 계속 검토한다. **전체 동질성 검증 완료를 학습 시작 조건으로
추가하지 않는다.** 현 구동 보고서만으로 학습 시작 가능 또는 전체 P1 완료를 선언하지 않는다.

## 2026-09-15 — itcerdo 실행 및 이동 전 인계 (당시 기록)

아래 로컬 착수 기록 이후 `ssh.md`와 프로젝트 HF 토큰을 받아 원격 작업을 진행했다.
작업 범위는 `/home/itcmaster/projects/phdq3`이며 neuron에는 접속하지 않았다.

- itcerdo: Python 3.11.16, torch 2.11.0+cu128, transformers 5.16.1,
  RTX 5090 sm_120, CUDA matmul 정상, xFormers 없음. 실제 pip freeze를
  `requirements.lock.txt`에 보존했다. 환경·데이터 보고서는
  `blt_hf_checks/results/*itcerdo_20260915_bootstrap.json`에 있다.
- 원본 main revision `8134b32f0b1d25d1248c30e8c7bdfd442d3bb380`,
  entropy revision `f2aae511e44e2086b1204bc4ddec6ac6c9651332` 다운로드 완료.
  토큰은 원격 프로젝트의 `artifacts/hf_home/.hf_access`에서만 읽었으며
  로컬로 전송하거나 로그에 출력하지 않았다.
- 원본 entropy 두 파일은 직렬화 hash가 다르지만 **129개 텐서의 key/shape/dtype/값이
  전부 일치**했다. B에는 별도 `facebook/blt-entropy`의 고정 snapshot을 사용했다.
  과거 legacy 실행이 실제 내려받은 revision은 아직 회수되지 않아 같다고 단정하지 않는다.
- vendored converter upstream `2cba19507be799b7bef247ca6c1c4708bf881b5b`를
  수정해 revision·dtype·tokenizer·RoPE·정규화 설정·원본 attention 설정을 보존했다.
  `vendor.diff`가 수정 기록이다. 실제 HF 환경에서 **35개 단위 테스트 통과**.
- CPU 변환 완료: `artifacts/converted/blt-1b-hf-own` (itcerdo),
  510개 텐서, 전부 bf16. 원본 LICENSE 포함. 변환 보고서는
  `blt_hf_checks/manifests/conversion_B_20260915.json`.
- **상태는 converted_unvalidated / pending**. 모델 전 텐서 A↔B 검증,
  strict model load, 실제 GPU forward/generation, OSC 구현, 전체 동질성 판정은 남아 있다.
  원본 `original_spec` 설정을 저장한 것만으로 HF가 해당 마스크를 실행하는 것은 아니다.
- 원본 entropy는 local_block_causal, window=512, EOS=2다. local encoder/decoder도
  window=512이며 global은 block_causal이다. HF 설치 코드에는 단순 causal 경로가
  있으므로 OSC 구현 시 global EOS 격리도 포함해 검토한다. legacy entropy dump를
  정답으로 취급하지 않는다. 코드 사본과 출처는 `vendor/model_sources/` 및
  `manifests/model_source_review.json`에 보존했다.
- 사용자 이동 요청으로 **tmux 세션 `phdq3-p1`**을 만들었다. 첫 실행은 기존 SSH
  다운로드와 완료 시점이 겹쳐 동일 manifest 저장을 거부하고 종료했다. 원본 보고서를
  보존하고 재시작하여 비교·변환이 **2026-09-15 03:41:58 UTC에 exit_code=0**으로 끝났다.
  현재 세션은 shell 대기 상태이며 추가 학습/검증 job이 실행 중인 상태가 아니다.
  로그·status는 원격 `artifacts/logs/p1_artifacts_20260915.{log,status}`에 있다.

재개 순서: B 전 텐서 정적 검증과 strict load → OSC 후보 구동 준비 → 사용자 실행용
neuron 학습 코드. 학습은 동질성 검증 완료와 무관하게 진행하며, 검증 트랙은 별도로
기준·fixture·허용오차를 재검토한다. 동질성 미확인 시 후속 실험을 새로 계획한다.

## 2026-09-15 — 로컬 기반 구현

이 절은 최초 착수 시점의 기록이며 현재 상태는 위 최신 기록을 따른다.

현재 단계: 실행 준비 G 진행 중. 본 학습·동질성 검증 완료 상태가 아니다.

2026-09-24 통합 validation 시험 설계: 기존 별도 1GPU 생성 대신 A100 4GPU
학습 job 내부에서 rank별로 전체 validation 행을 분담한다. 이전 코드의 역전파
목표는 target-token loss로 유지하되, `SELECTION_METRIC=val_gleu`에서는 epoch별
전체 corpus GLEU가 높은 checkpoint를 `best.json`/`best_gleu.json`으로 선택한다.
generation.py와 이전 실험에서 이식한 GLEU scorer를 재사용하며 M2는 선택에서 제외한다.
첫 2-epoch native 시험은 beam1로 기존 생성 조건과 처리량·메모리를 비교한다.
beam4는 별도 RUN_ID로 검증한다. 전체 생성 rank의 index를 정확히 1회 수집하고,
score 입력·출력 hash를 불변 epoch 기록에 남긴다. Mac CPU 계약 테스트만 수행하며
Neuron GPU 실행은 사용자가 맡는다.

- 사용자: itcerdo/neuron의 환경 설치 완료. 원격 버전·GPU 구동은 아직 확인하지 않음.
- 사용자 지시로 에이전트는 neuron에 접속·파일 전송·작업 제출하지 않음.
  neuron용 코드·명령은 로컬에서 준비하고 사용자가 실행한다.
- itcerdo는 사용자가 제공할 접속정보·디렉터리 구조 파일을 기다린다.
- 초기 SSH 접근은 로컬 sandbox에서 실패했고, itcerdo 재시도 승인도 거절됨.
  그 이후 원격 접속하지 않았으며 원격 파일·환경·job을 변경하지 않았다.

구현한 항목:

1. `data_adapter.py`: strict TSV reader, canonical 9개 경로, 본문 자동 특수토큰
   비활성화, BOS/EOS 명시 삽입, EOS 포함 unshifted labels, 무절단·생성 예산 검사.
   현재 vocabulary 계약은 BOS=1, EOS=2, byte+4이며 실제 tokenizer 반환 IDs가
   다르면 실패한다. HF tokenizer의 정규화나 특수문자 해석을 조용히 허용하지 않는다.
2. 검증 전 collator는 동일길이 무패딩 batch만 허용한다. 길이가 섞인 batch는
   오류를 내며, batch_size=1 경로가 가능하다. pad-aware 모델 경로는 아직 미구현.
3. `manifest.py`: P1 D-1 identity 필수 필드, OSC/LRE·config·tokenizer·dataset
   hash 비교, 전체 split 논리 순서 보존, gold M2 source 정합성 검사,
   모델 artifact SHA256 검사, 기존 결과를 덮어쓰지 않는 원자적 JSON 저장.
4. `check_env.py`: 기존 환경을 읽고 버전·xFormers 부재·GPU capability·작은 CUDA
   matmul을 점검한다. 설치/업데이트/optimizer/backward는 수행하지 않는다.
   `--cpu-only`는 정보성 결과이며 GPU 환경 통과로 해석하지 않는다.
5. `analyze_data_lengths.py`: torch 없이 제공 TSV 9개·val/test M2 6개를 읽어
   길이·정합성·파일 hash를 기록한다. 모델/토크나이저 실행 결과와 구분한다.

로컬 확인:

- Python 3.11.2 표준 라이브러리로 unittest 25개 통과.
- tokenizer 테스트는 byte+4 테스트 대역을 사용한다. 실제 HF tokenizer parity,
  torch collate 실행, CUDA 모델 forward/backward 통과를 의미하지 않는다.
- `results/data_lengths_2026-09-15.json`: 201,534행, 최대 입력 1,380,
  최대 prompt 694, 최대 target 685, 1,024 초과 9건, 2,048 초과 0건.
  val/test의 TSV–M2 source가 공백 토큰화 기준 전부 일치.
- `results/env_mac_2026-09-15.json`: 로컬에는 torch/transformers 미설치.
  기존 원격 환경을 재설치하거나 로컬에 CUDA wheel을 설치하지 않았다.

다음 구현·실행:

1. itcerdo 정보 파일에서 프로젝트 경로, 환경 Python, 저장공간, 모델/cache 경로 확인.
2. 준비된 환경에서 check_env 실행, 실제 설치 버전 기반 lock 생성.
3. 원본 main/entropy snapshot 및 legacy provenance 확인, converter upstream SHA
   고정·vendoring·revision/dtype/tokenizer 수정, B 변환. 아직 converter/B 없음.
4. B의 실제 tokenizer와 모델을 로드해 adapter 및 구동 확인. 고정한 OSC 모델 구현과
   학습 루프를 준비해 사용자 실행용 neuron 명령으로 연결.
5. 파인튜닝은 구동 준비 이후 검증 완료와 무관하게 진행한다. 별도 트랙에서
   검증 규칙을 재검토·실행하고, 동질성 미확인 시 후속 실험을 새로 계획한다.

attn backend, mask 구현, entropy 기준, cache 동작은 아직 결정·검증되지 않았다.
출처를 확인하지 않은 converter commit이나 원격 환경 lock을 임의 생성하지 않는다.
## 2026-09-23 — learner 생성 완료와 다음 학습 사이클

- learner test beam1 생성은 A100 1GPU에서 4,265/4,265건 완료했다. job 912817과
  912933은 각각 exit 75로 checkpoint 후 멈췄고 job 913148이 같은 EVAL_DIR에서
  이어서 exit 0으로 끝났다. 세 job의 실행 시간 합계는 18,478초다. GLEU/M2 CPU
  채점은 별도 job으로 남아 있다.
- 기존 union `union-h200-4gpu-main-01`의 step 4479 checkpoint는 보존하되 재개하지
  않는다. 새 union은 아래 공통 10-epoch 정책으로 처음부터 실행한다.
- 재개 identity에서 Git commit 문자열을 제거했다. train/generate 단계가 실제 사용하는
  파일만 `code_hash`로 묶으며 commit·SLURM job·partition·node는 `provenance.jsonl`에
  기록한다. rank 0만 checkpoint hash를 읽고 결과/오류를 모든 rank에 broadcast한다.
- 다음 사이클은 BF16, A100 4GPU, effective batch 32, LR 1e-5, 10 epoch,
  5% warmup이다. epoch별 checkpoint를 보존하고 동일 조건의 전체 validation GLEU로
  `best_gleu.json`을 선택한 뒤 test에 사용한다.
- Lang-8은 제공 data를 수정하지 않고 union prefix로 파생한다. union suffix가
  korean_learner+native와 TSV/M2 모두 정확히 일치하는지 확인했다. 행 수는
  train 76,692 / val 16,434 / test 16,434다.
- union 합본 M2의 빈 separator 문제를 막기 위해 모든 새 `S ` 행을 논리 record
  경계로 처리한다. H200은 script에서도 한 job 최대 2GPU로 제한한다.

## 2026-09-24 — Lang-8 데이터 디렉터리 정리

- 요청에 따라 파생 Lang-8을 `data/Preprocessed/lang8`에 배치한다. 기존
  union/korean_learner/native 파일은 수정하지 않는다. split TSV/M2와 원문·교정문,
  val/test hanspell, train→test→val 순서의 합본 파일을 만든다.
- 학습·생성·채점의 `dataset_split_path`는 새 디렉터리를 읽는다. 세 job shell과
  로컬 M2 채점 shell은 실행 전에 동일한 파생 검사를 수행한다.
- 제공된 union val의 별도 corrected sidecar에는 TSV 정답과 끝 공백 한 곳의 차이가
  있다. Lang-8 원문·교정문 sidecar는 학습·평가의 정본인 TSV 열에서 생성한다.

## 2026-09-25 — 통합 validation 생성 병목

- native 2-epoch A100 4GPU 시험 job 914931은 exit 0, epoch 1/2 GLEU
  53.7270/55.1762, 총 7,585초였다. 각 epoch의 학습은 969/1,088초,
  validation 생성·GLEU는 2,477/2,458초로 생성이 우세하다.
- 현재 OSC는 cache/padding을 거부하고 통합 validation에서 문장별 batch 1로
  `generate()`를 호출한다. 우선 정확히 같은 길이의 prompt를 묶는 선택적
  `VALIDATION_BATCH_SIZE`를 추가했다. 기본값 1은 유지한다.
- 실제 checkpoint에서 beam 1/4 각각 단건과 batch 4의 token ID 일치, 시간,
  GPU 메모리를 재는 `bench_generation.py`를 추가했다. 성능 향상은 측정 전까지
  주장하지 않는다. native 2-epoch run의 재개 identity는 변경된다.
