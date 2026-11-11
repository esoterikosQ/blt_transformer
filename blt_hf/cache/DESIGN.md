# BLT 증분 생성: 단계 0 상태 수명 조사

상태: 2026-09-26 단계 0 실측 완료, 단계 1 민감도·시제품 조사 중. 실행 정본은
`plan/P3a_generation_cache_20260925.md`다. 이 문서는 캐시가 안전한 위치를
**증명하기 전의 가설**과 검사 결과를 구분한다.

## 실제 1B BF16 단계 0 결과

itcerdo의 converted BLT-1B, eager/OSC/BF16, RTX 5090에서 가중치 변경·학습
없이 조사했다. 원본 보고서:

- `blt_hf_checks/results/p3a_patch_causality_20260926_v2.json`:
  한국어 25, ASCII 16, 혼합 36, 512-window 56, EOS 26개 접두부를 검사했다.
  혼합 입력 5개 접두부에서 이전 patch 시작점이 바뀌었고 다른 fixture에서는
  경계 불일치가 없었다. 예를 들어 시작점 19를 결정하는 entropy가 전체
  입력에서는 1.3359375, 길이 22 접두부에서는 1.3515625였다. 유효 BF16
  threshold 1.3359375를 사이에 두므로 경계가 뒤집혔다.
- `blt_hf_checks/results/p3a_state_stability_repeat_20260926.json`:
  혼합·긴 문장 총 15개 접두부에서 닫힌 global patch와 decoder의 과거
  출력이 모두 바뀌었다. 이 비교의 최종 위치 argmax는 15/15 일치했지만
  전체 생성 출력이 일치한다는 뜻은 아니다. 같은 입력 길이를 다시 실행한
  entropy, encoder, global, decoder, logits는 15/15에서 비트 단위로
  동일했다. 입력 길이 변화가 수치 차이의 원인이다.
- `blt_hf_checks/results/p3a_forward_profile_20260926.json`:
  사전학습 가중치·반복 한글 합성 입력 64/128/256/512바이트의 full forward
  중앙값은 16.50/16.66/17.28/19.61ms, patcher는
  3.73/3.81/4.04/4.90ms, global은 7.25/7.27/7.29/7.29ms다.
  5090 한 대의 합성 입력 시간이며 A100의 실제 파인튜닝 모델 생성 속도나
  캐시 이득을 보장하지 않는다.

수학적으로 threshold patching이 인과적이어도, 현재 BF16 kernel의 텐서
형상 의존 연산 때문에 실제 entropy가 조금씩 변한다. 그러므로 최초 시제품은
매 step **전체 patcher를 다시 실행**해 현재 no-cache와 같은 경계를 사용한다.
`frontier.shared_closed_patch_count`는 경계가 같은 *구조적* 닫힌 구간만 찾으며
activation이 동일함을 보증하지 않는다. 닫힌 상태 재사용은 출력 ID 전수 비교
전에는 실험용으로만 취급한다.

## 단계 1 시제품의 현재 한계

`check_reuse_sensitivity.py`는 전체 global 계산 뒤 닫힌 patch 출력만 이전
step의 값으로 바꿨다. 사전학습 1B의 한국어·혼합·긴 입력에서 32 step씩 총
96개 다음 바이트 ID는 같았으나 logits는 최대 0.125 차이가 났다. 이 검사는
계산을 건너뛰지 않으므로 속도 결과가 아니다.

`cache/global_reuse.py`는 매 step **전체 entropy patcher와 local encoder**를
유지한다. 시작점이 그대로이고 patch가 둘 이상이면 decoder가 현재 열린
patch의 global 출력을 마지막 byte에서 참조하지 않는 특성을 이용해 global
forward를 건너뛴다. 첫 A100 실측 당시에는 새 시작점이 생기면 공통으로 닫힌
patch의 KV만 남기고 global tail을 재계산했다. 이후 재계산 구간의 비용을
확인해 **새 경계에서는 global 전체를 기준 경로처럼 재계산**하도록 고쳤다.
선택형 decoder KV 재사용은 경계가 그대로인 step에만
적용하고, 경계가 바뀌면 decoder 전체를 다시 계산한다. 이 구현은 batch 1,
빔 1, 연속 접두부, `eval()`과 `inference_mode()`로 제한된 **진단 시제품**이다.
module.forward를 한 실행 동안만 교체하므로 병렬 호출·학습·HF `generate`
기본 경로에 사용하지 않는다.

`p3a_global_skip_bench_20260926.json`에서 세 입력 각 32 step의 다음 ID
불일치는 0개였고 global pass를 건너뛴 step은 각각 27/27/21개였다.
같은 실행의 기준 대비 평균 forward 시간 개선은 1.49/1.45/1.35배였다.
`p3a_global_decoder_nocopy_20260926.json`의 global+decoder 시제품도
96/96 다음 ID 일치, decoder 재사용 27/27/21 step, 평균 개선
1.49/1.50/1.43배였다. 모두 **5090의 사전학습 가중치·합성 greedy
입력·짧은 forward** 검사다. 다른 시점의 5090 실행에서는 기준 시간이
약 16ms와 60ms 사이로 크게 달랐다. 따라서 2배 채택 목표 달성이나
A100 본 평가의 출력을 주장할 수 없다.

최종 EOS 재사용 차단·KV 복사 제거·반복 경계 검사 반영 뒤
`p3a_global_final_20260926.json`에서도 96/96 다음 ID 일치,
global/decoder 재사용 27/27/21 step, 평균 시간 개선 1.50/1.56/1.49배였다.
이 보고서의 `reuse_code_sha256`이 현재 시제품 소스를 식별한다.
`p3a_global_report_schema_20260926.json`은 최종 bench 코드의 8-step
보고서 필드(`probe_token_parity`, `peak_allocated_bytes`, 합산 속도)를
확인한 짧은 실행이며 32-step 속도 비교를 대신하지 않는다.

## A100 native 체크포인트 검사와 다음 판정

사용자가 Neuron A100 1GPU job `915640`을 실행했고, 보고서를 GitHub에서
받았다. `p3a_native_cache_probe_01.json`은 native `native-main-01`의
`step-00001155-24b756e2`(model SHA256 `cc8466be...7454ab28`)와 제공된
native validation TSV를 사용했다. GPU 환경 검사 통과, job 종료 코드 0,
소요 208초, peak allocated 9.33GB였다. 12개 길이별 문장에서 각 32 step,
총 384개 다음 byte ID가 모두 일치했지만 EOS가 나온 사례는 없었다.
보고서의 코드 SHA256은 이 수정 전 시제품 `2d81faa0...4fc1411014`와
일치하므로 당시 실행 코드를 식별할 수 있다.

초기 step을 제외한 전체 forward 측정은 기준 16.346초, 시제품 11.917초로
**1.372배**였다. 372개 측정 step 중 299개에서 global을 건너뛰었고,
이때 평균 43.94→26.59ms(1.65배)였다. 재계산한 73개에서는
43.92→54.35ms(0.81배)로 느렸다. 특히 한 사례는 32개 중 10개만
건너뛰어 전체 시제품 시간이 기준보다 길었다(0.86배). 현재 채택 기준인
native validation 전체 생성 2배 단축과 beam 4 parity는 입증되지 않았다.

비싼 KV 꼬리 재계산을 제거하고 경계 변경 step을 기준 global full forward로
되돌렸다. 작은 OSC 테스트와 itcerdo 1B의 합성 greedy 96 step은 다음 ID
불일치 0개였다(`p3a_boundary_fallback_20260926.json`). 이것은 **수정 후
A100 속도·fine-tuned parity 증거가 아니다**.

사용자의 수정 후 A100 job `915655` 결과는
`p3a_native_cache_probe_02.json`이다. 01과 checkpoint model SHA256,
native validation TSV SHA256, 12개 sample ID, model/bench 코드 hash,
A100·torch·transformers 버전이 같다. `reuse_code_sha256`만 수정본
`dd54916c...9019813`으로 바뀌었다. job 종료 코드 0, 소요 148초,
peak allocated 9.289GB. 12×32=384개 다음 byte ID 불일치 0개,
EOS 생성 0개, global skip 299개로 01과 같다.

첫 step을 제외한 전체 forward는 기준 15.659초, 수정본 10.701초로
**1.463배**다. global skip 299개에서는 평균 42.08→25.43ms(1.65배),
나머지 73개에서는 42.17→42.44ms(0.994배)였다. 01에서 가장 느렸던
문장도 0.86배에서 1.14배로 개선됐다. 경계 변경의 추가 비용은 제거됐지만
**전체 native validation 생성 2배**의 채택 기준은 아직 통과하지 않았다.
이 job 시간 148초에는 환경·데이터 검사와 체크포인트 로딩이 포함되므로
forward 합계와 혼동하지 않는다. 전체 생성, EOS, beam 4, GLEU 결과도
이 32-step probe로는 확인되지 않는다. 본 평가 backend는 no-cache로 유지한다.

사용자의 세 번째 A100 job `915710`은 같은 checkpoint·TSV·12개 sample,
같은 model/reuse/bench 코드 hash에서 `REUSE_DECODER=1`만 켠
`p3a_native_cache_probe_03_decoder.json`이다. 환경 검사와 job이 통과했고
종료 코드 0, 소요 189초, peak allocated 9.303GB였다. 다음 byte ID는
384/384 일치했으나 EOS 생성은 0개다. 측정 372 step 중 global skip 299개와
decoder 재사용 299개가 대응한다.

첫 step을 제외한 forward 합계는 기준 16.405초, 시제품 11.917초로
**1.377배**였다. decoder를 끈 02의 1.463배보다 낮다. global skip step은
평균 44.07→28.17ms, 재계산 73 step은 44.21→47.85ms였다. 03에는
sample 2290의 step 3(재사용 중 578.85ms)과 sample 297의 step 27
(재사용하지 않은 step에서 263.19ms)이라는 큰 지연이 있다. 지연 원인은
로그만으로 확정할 수 없고, 두 실행의 기준 시간도 다르므로 decoder KV가
그 자체로 항상 느리다고 결론 내리지 않는다. 이 실행에서 측정된 속도
향상은 1.377배이고, 지연 두 건을 제외하면 약 1.47배다. 02/03의 job 전체
시간은 로딩·검사를 포함하므로 생성 속도로 해석하지 않는다.

**2026-09-26 정정:** 03의 속도 저하 원인을 규명하기 전에 2배 목표만으로
시제품 적용을 배제한 판단은 철회한다. sample 2290 step 3은 첫 decoder
재사용 step이고, sample 297 step 27에서는 이전과 patch 경계가 달라져
global과 decoder를 모두 다시 계산하도록 설계된 분기로 들어갔다. 이 분기는
logits 차이 0으로 기준과 같았다. CUDA 이벤트가 `reuse.run()` 전체를
감싸므로 두 지연에 GPU 계산, Python/GC 대기, 첫 kernel 준비 중 무엇이
포함됐는지는 기존 로그만으로 구별할 수 없다.

`diagnose_cache_spikes.py`로 같은 두 접두부의 decoder on/off를 한 A100
job에서 교차 반복하고 patch 시작점, 모듈별 시간, GC 시간을 기록한다.
재현되는 계산 병목이나 잘못된 무효화가 있으면 수정·재검사한다. 전체 native
validation 생성, EOS, beam 4, GLEU 및 최종 문자열 동일성은 아직 미검증이다.
현재 no-cache가 기본인 것은 **통합 backend가 아직 없기 때문**이며 캐시
적용을 포기한 판정이 아니다. 2배는 속도 목표로 남기되, 실제 개선과 출력
검증 결과를 보고 선택형 적용을 결정한다.

사용자가 실행한 A100 job `915864`의
`p3a_native_cache_spike_diagnostic_04.json`은 같은 checkpoint·TSV·03 보고서
접두부를 복원해 decoder on/off, 계측 on/off를 각 3회 교차 실행했다. 환경 검사와
24/24 다음 ID 확인을 통과했고 exit 0, 전체 job 217초였다.

- sample 2290 step 3의 **첫** decoder 재사용 실행은 569.42ms로 03의
  578.85ms 지연을 재현했다. 그 다음 계측 실행은 25.57ms, 후속 반복은
  25.79~25.94ms다. 같은 step의 decoder off도 25.48~25.96ms였다.
  첫 실행에서 GC는 없었고 추가 GPU 할당은 약 2.8MB, 예약 메모리 증가는
  없었다. 한 번만 발생한 첫 사용 비용으로 보이지만 어느 kernel/라이브러리
  초기화인지 이번 계측만으로 특정할 수 없다. 첫 지연이 지난 후 모듈 시간은
  patcher 약 10.8~11.0ms, local encoder 약 1.5ms, global skip 약 0.03ms,
  local decoder 약 10.3~10.8ms로 decoder on/off 차이가 작았다.
- sample 297 step 27은 이전 patch 시작점의 index 43 값 79 앞에 새 시작점
  **76**이 삽입돼 경계가 변했다. 따라서 이전 global/decoder 상태를
  무효화한 것은 설계된 분기다. 이 step의 decoder on은 43.59~43.86ms,
  off는 42.74~43.41ms로, 03의 263.19ms가 반복되지 않았다. 계측 시
  patcher 약 11ms, global 약 17.3~17.6ms, decoder 약 10.6~10.8ms였고
  GC 기록은 없었다. 원래의 일회성 263ms 원인은 이 자료만으로 확정할 수
  없지만, 경계 변경마다 발생하는 비용이나 decoder 재사용 분기 비용이라는
  증거는 없다.

따라서 03의 1.377배 합산치는 이 두 지연의 영향을 받았다. 그 두 step만
정상 반복 시간으로 대체하면 같은 03 기준 약 **1.47배**지만, 이는 전체
생성 속도 측정값이 아닌 추정이다. 확인된 지속 이득은 주로 global skip에서
나온다. 이 진단에서 캐시 분기 오류는 발견되지 않았으므로 임의로 재사용
규칙을 바꾸지 않는다. 다음 검사는 첫 사용 비용을 로딩/warmup과 분리하고
선택형 beam-1 생성에서 완성 문장·EOS·전체 native validation 출력과
실제 처리 시간을 비교하는 것이다. beam 4는 그 다음 별도 검증 대상이다.

## 선택형 완성 생성 경로

`generation.py`에 `global-prefix-greedy-v1`을 beam 1·batch 1 전용으로
연결했다. 각 문장의 첫 입력부터 완성/EOS/바이트 예산까지
`GlobalPrefixReuse(reuse_decoder=False)`를 쓰며, 기존 byte ID 억제
(`0,1,3`)와 `decode_generated` 정책을 공유한다. **decoder KV는 사용하지
않는다.** 기존 `hf-generate-exact-length-unpadded-v1`이 기본이다.
`eval.py`는 backend를 실행 identity에 넣고 `EVAL_DIR`을 분리한다. 캐시
코드 hash도 평가 identity에 포함한다. 현재 통합 경로가 실제 full 생성에서
기준과 같은 ID·EOS·UTF-8을 내는지는 Neuron 실측 전이므로, 먼저
`bench_complete_generation.py`로 native validation 길이별 12문장을 끝까지
양쪽에서 생성한다. 이 검사가 통과한 후 전체 native validation을 비교한다.

사용자 A100 job `915874`의 `p3a_native_complete_global_05.json`은 같은
fine-tuned native checkpoint와 validation TSV에서 12개 길이별 문장을
양쪽 backend로 끝까지 생성했다. 모든 문장이 EOS로 종료했고 12/12에서
전체 token ID·EOS·UTF-8·최종 문자열이 같았다. 한도 소진은 0건, job exit 0.
문장별 생성 시간 합계는 기준 43.594초, global 캐시 29.711초로
**1.467배**였다. 첫 기준 문장의 2.420초에는 초기 실행 비용이 섞여 있어
그 문장을 제외하면 약 **1.432배**다. 모든 문장에서 캐시가 더 빨랐지만
표본 결과를 전체 native validation으로 일반화하지 않는다.

다음은 동일 checkpoint·설정으로 전체 native validation 2,634문장을
HF/global backend 각각 별도 `EVAL_DIR`에 생성한 뒤, 불변 batch/완료 파일을
검증하면서 각 문장의 token ID·EOS·UTF-8·문자열을 전수 대조하고 batch
생성 시간 합계·p50/p95를 비교하는 것이다. 이 단계는 M2/GLEU 채점과
분리한다. 현재 decoder KV는 계속 꺼져 있다.

사용자 916171 비교에서는 두 평가의 동일 checkpoint·split·코드와 2,634건
완료 파일이 검증됐다. 2,602건은 완성 출력이 같고 **32건(1.21%)**은
token ID·문자열이 달랐다. EOS·UTF-8 상태는 동일했다. 생성 시간은 기준
9,074.297초→global 캐시 6,620.644초(**1.371배**). 12문장 예비 검사의
출력 일치는 전체 split에 일반화되지 않았다. `global_reuse.py`는 patch
시작점이 그대로일 때 이전 global hidden을 반환하므로 이 분기가 유력한
원인이지만, 저장된 최종 출력만으로 직접 원인을 확정할 수 없다.
`diagnose_cache_divergence.py`는 32개 행의 첫 토큰 분기까지 같은 접두부를
재생하고 그 순간 재사용 여부와 full forward의 기준 ID 회복 여부를 기록한다.
결과를 확인하기 전에는 캐시 적용 범위를 변경하거나 출력 동등성을 주장하지
않는다.

첫 분기 재생 916179는 32/32 재현, **17건은 global skip이 직접 토큰
선택을 변경**했다. 나머지 15건은 global을 재계산해도 기존 HF 기준 ID와
달랐고, 11건은 캐시/full logit 자체가 같았다. HF `generate()`의
`logits_to_keep=1`과 v1 수동 forward의 기본값 0이라는 별도 구현 차이를
확인했다. 08 진단은 같은 입력에서 두 logit projection 모양의 ID를
기록해 나머지 15건을 분리한다. 새 backend나 재사용 조건 수정은 이 결과
이후에 결정한다.

08 A100 진단(916207)에서는 `logits_to_keep=1` full forward가 32/32
저장된 HF 기준 ID를 두 번 연속 복구했다. 15건의 full0↔HF 차이는
projected logit matrix 모양에 따른 수치 차이로 설명된다. 나머지 17건은
global skip 자체가 ID를 바꾼다. 이에 v1을 보존하고 선택형
`global-prefix-greedy-v2`에서 **logit projection만 1로 고정**했다.
global skip은 그대로여서 v2의 전체 출력 동등성을 주장할 수 없다. 먼저
기존 32개 불일치·12개 대조군의 완성 생성 09를 측정하고, 그 결과에 따라
global 재사용 조건을 별도로 수정·검증한다.

09(916208)는 v2에서 기존 32건 중 12건을 해결했고 20건은 남았다.
대조군 12건은 모두 일치했다. 남은 20건은 모두 07에서 global skip이
일어난 행이다. 선택형 guarded backend는 `logits_to_keep=1`을 유지하고
skip 후보의 상위 두 합법 byte logit 간격이 1.0 이하일 때 full forward로
다시 계산한다. 07의 관찰된 최대 per-logit drift 0.421875에서 두 후보의
차이는 최대 0.84375만큼 움직일 수 있다는 **관찰 범위 내** 동기다.
이것은 수학적 보증이나 미관측 입력의 오차 상한이 아니다. fallback은
entropy/local encoder까지 한 번 더 실행하므로 너무 자주 일어나면
속도 이득이 줄어든다. 10 표본에서 결과·refresh 빈도·시간을 먼저 측정한다.

## 기준 생성 경로

`blt_hf/generation.py`는 생성할 때마다 `[BOS] source SEP generated_prefix`를
`BltForCausalLM.generate(use_cache=False)`에 준다. `BltModel.forward()`는 매번
전체 바이트 hash embedding → entropy patcher → local encoder → global
transformer → local decoder를 실행한다. `attention.py:require_unpadded()`는
OSC에서 cache 입력을 거부하며 `BltModel.forward()`의 global transformer
호출은 `past_key_values`를 전달하지 않는다.

## 후보 재사용 경계

| 상태 | 현재 근거 | 확정 전 검사 |
|---|---|---|
| 과거 바이트 entropy | 같은 길이 재실행은 동일하지만 길이 변경 시 기존 값도 바뀜 | 초기 시제품은 전체 patcher 재실행 |
| patch 시작점 | 혼합 fixture 5/36에서 과거 시작점 변경, `include_next_token=True` | 매 step 새 경계를 비교하고 처음 바뀐 patch 전부터 무효화 |
| 마지막 열린 patch 표현 | pooling/cross-attention은 현재 patch의 바이트를 사용하므로 바뀔 수 있음 | 새 바이트 전후 hidden과 downstream 의존성 추적 |
| 확정 patch의 local encoder 출력 | 일부 닫힌 patch에서도 BF16 값 차이 관측 | output parity 없이는 재사용 금지 |
| 확정 patch의 global 출력 | 15/15 비교에서 닫힌 patch 값도 달라짐 | 재사용 시 다음 바이트 ID 민감도 측정 필수 |
| local decoder 과거 상태 | 15/15 비교에서 과거 byte 값도 달라짐 | 의존 구간과 output parity 측정 필수 |

`decoder_patch_ids = _patch_ids_from_lengths(patch_lengths[:, 1:], sequence_length)`
는 patch 한 칸 이동에 해당한다. 이것이 열린 patch의 변화가 과거 decoder
상태에 미치는 범위를 어디까지 제한하는지 실제 출력으로 검증한다. 안정성이
확인되지 않은 상태를 무조건 캐시하지 않는다.

## 단계 0의 검사 순서

1. 로컬 `tests/test_hf_cache_causality.py`에서 next-token 가상 슬롯의
   patch 길이 의미와 경계 비교기의 실패 감지를 검사한다.
2. BF16 GPU에서 `python -m blt_hf_checks.check_patch_causality --output <새 JSON>`를
   실행한다. 실제 1B entropy patcher에 짧은 한국어·영어·혼합 입력,
   512-window 경계, EOS 구간을 넣어 접두부 경계 안정성을 기록한다.
3. 실제 1B에서 local/global/decoder의 중간 출력 안정성과 동일 길이 반복
   오차를 측정했다. 반복 오차는 0이었으며 길이 변화 차이는 위에 기록했다.
4. `blt_hf_checks/bench_generation.py`로 동일 checkpoint의 현재 no-cache
   baseline을 A100에서 측정한다. 결과는 추후 증분 backend 성능 비교에 사용한다.

이 단계의 출력은 `check_patch_causality` 보고서, 구성요소별 상태 수명 보고서,
벤치마크 JSON이다. 이 증거가 나오기 전에는 증분 상태 구현을 시작하지 않는다.

## 소스와 운영 제약

- 모델 가중치·tokenizer·학습 코드는 변경하지 않는다. `blt_gec/`, 원본
  데이터, site-packages도 수정하지 않는다.
- Neuron에 에이전트가 접속하지 않는다. itcerdo는 사용자 제공 환경에서
  forward 검사만 수행하며 optimizer·학습 작업은 하지 않는다.
- 관련 소스: `blt_hf/patched/modeling_blt.py`, `blt_hf/patching.py`,
  `blt_hf/attention.py`, Meta의
  [threshold 패처](https://github.com/facebookresearch/blt/blob/main/bytelatent/data/patcher.py),
  [no-cache 생성](https://github.com/facebookresearch/blt/blob/main/bytelatent/generate_blt.py).
