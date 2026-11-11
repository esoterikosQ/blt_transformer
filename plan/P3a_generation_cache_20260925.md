# P3a: BLT 증분 생성·접두부 재사용 실행안 (2026-09-26 확정)

작성일: 2026-09-25. 이 문서는 현재 BF16/eager/OSC 모델의 **생성 시간 단축**을
위한 실행안이다. 사용자의 이번 요청에 따라 기존
`P3_조건부_후속_프로젝트_계획.md`의 P1 완료·처리량 트리거를 기다리지 않고
설계 검토를 시작한다. 2026-09-26 사용자 승인으로 이 실행안을 확정하고
단계 0부터 착수한다. 기존 P3a의 P1 완료·처리량 트리거는 이 작업에 적용하지
않으며, 단계별 correctness·성능 판정은 유지한다.

## 목표와 변경 범위

- 동일한 파인튜닝 체크포인트와 데이터로 생성할 때, 매 바이트마다 전체 접두부를
  다시 계산하는 비용을 줄인다. 우선 native validation 빔 1, 이후 빔 4와
  learner/lang8/union으로 확장한다.
- 가중치, 학습 손실, 제공 데이터, GLEU/M2 채점 규칙을 바꾸지 않는다. 기존
  no-cache 생성은 기준 구현이자 즉시 사용 가능한 fallback으로 보존한다.
- 출력 동일성이 입증된 경우에만 새 backend를 본 평가나 epoch별 GLEU 선택에
  사용한다. 비교 중에는 기존 결과 디렉터리를 재사용하지 않는다.
- 에이전트는 Neuron에 접속·전송·제출하지 않는다. 로컬에서 코드와 명령을
  준비하고, 실제 A100 작업은 사용자가 실행한다.
- 2026-09-27 작업 분리: 캐시 구현·작은 실제 1B 추론·단계별 프로파일은
  itcerdo의 전용 worktree에서 반복한다. 전체 split·A100 속도·학습 적용은
  사용자 Neuron 작업과 BLT 작업창에서 확인한다. 상세 소유권과 인계는
  저장소 루트 `cache.md`를 따른다.

## 현재 확인된 사실과 미확인 전제

- `blt_hf/generation.py`는 `use_cache=False`로 전체 접두부를 전달한다.
  `blt_hf/attention.py`는 OSC 캐시 입력을 거부한다. `modeling_blt.py`는
  global transformer에 이전 patch 상태를 전달하지 않는다. 단순 플래그
  변경으로는 증분 생성이 되지 않는다.
- 실제 1B 설정은 엔트로피 임계값 1.335442066192627, `monotonicity=false`,
  `threshold_add=null`, `max_patch_length=null`이다. 공식 패처의 인과적
  경계 결정은 수학적 구조 설명이며, 현재 BF16 구현의 **수치적 경계 불변성은
  성립하지 않았다**. 아래 단계 0 실측을 따른다.
- 실제 1B에서 닫힌 patch의 global/decoder 상태도 입력 길이에 따라 달라졌다.
  같은 길이의 재실행은 비트 단위로 동일했다. 따라서 구조적으로 닫힌 patch와
  수치적으로 그대로 재사용 가능한 activation을 구분한다.
- 기존 native 통합 validation 빔 1은 epoch당 약 41분(4 GPU 분담)이었다.
  이는 속도 비교 기준의 *관측값*이지만, 새 backend 비교는 동일한 체크포인트,
  샘플, GPU 수, dtype, 빔, 길이 제한에서 별도로 측정한다.
- 근거 소스: [Meta의 threshold 기반 패처](https://github.com/facebookresearch/blt/blob/main/bytelatent/data/patcher.py),
  [Meta의 no-cache 생성](https://github.com/facebookresearch/blt/blob/main/bytelatent/generate_blt.py),
  저장소의 `blt_hf/patched/modeling_blt.py` 및
  `blt_hf_checks/results/p1_neuron_code_v2_20260916.json`.

## 단계 0: 기준선·재사용 경계 확인

1. 고정된 native 체크포인트와 문장 ID 목록을 사용해 A100에서 빔 1·4,
   batch 1·4의 생성 시간, 생성 바이트 수, peak memory를 기록한다. 준비된
   `blt_hf_checks/bench_generation.py`를 활용한다. 단계별 프로파일은
   entropy/patcher, local encoder, global transformer, local decoder로 나눈다.
2. `check_patch_causality.py`를 작성해 한 문장의 전체 입력과 모든 접두부를
   각각 실행한다. 각 단계에서 확정된 patch 시작점, entropy, patch ID,
   encoder/global/decoder가 참조하는 범위를 비교한다. 임계값 근처,
   patch가 열려 있는 경우와 막 닫히는 경우, 511/512/513바이트, EOS,
   긴 한글 문장을 포함한다. 작은 CPU 모델과 실제 1B 모델을 구분해 기록한다.
3. 결과를 `blt_hf/cache/DESIGN.md`에 상태 수명표로 정리한다. **확정된
   patch보다 앞쪽의 값이 바뀌면** 그 값은 무조건 재사용하지 않는다. 필요한
   재계산 시작점을 실제 출력 의존성으로 결정한다. `global`이 새 patch에서만
   갱신된다는 기존 P3a 가정은 실제 BF16 계산에서는 기각됐다.

2026-09-26 itcerdo 실측: 한영 혼합 입력의 36개 접두부 중 5개에서 기존
patch 시작점이 달라졌다. 15개 activation 비교에서 닫힌 global patch와
decoder 출력은 모두 바뀌었다. 같은 길이의 반복 forward에서는 비교한
entropy/encoder/global/decoder/logits가 모두 비트 단위로 동일했다.
반복 한글 입력의 64~512바이트 forward 프로파일에서 patcher 약 3.8~4.9ms,
global 약 7.3~7.8ms였지만 이는 5090·사전학습 모델·합성 입력이며 A100
파인튜닝 체크포인트의 생성 처리량을 예측하는 수치는 아니다. 보고서와
해석은 `blt_hf/cache/DESIGN.md`에 있다.

작은 모델·정적 테스트는 로컬에서, 실제 1B의 짧은 forward 확인은 itcerdo에서
수행할 수 있다. A100 성능 job은 사용자에게 실행 명령과 예상 산출물을 전달한다.

판정: 구조적으로 닫힌 접두부는 존재하지만 수치적 상태 불변성은 거짓이다.
따라서 단계 1은 **실험용 backend**로만 진행한다. 기존 생성의 출력 동일성과
실측 속도 모두 통과하기 전에는 평가·체크포인트 선택 경로에 연결하지 않는다.

## 단계 1: 단건·빔 1 증분 backend

1. 먼저 실제 생성 경로의 연속 step에서 이전 닫힌 global/decoder 상태를
   교체 주입하는 민감도 실험을 한다. 출력 ID가 달라지는 조건과 top-1
   마진을 기록한다. 이 검사는 속도 개선을 주장하지 않는다.
2. `blt_hf/cache/`에 명시적 `GenerationState`를 만든다: 입력 바이트와
   위치, 현재 전체 patch 경계, 구조적으로 닫힌 patch, 재사용 후보인
   encoder/global/decoder 상태, EOS/문서 구간을 담는다.
3. **첫 구현은 매 step 전체 entropy patcher를 실행한다.** 현재 BF16
   경계가 과거 위치에서도 바뀔 수 있으므로 창 제한이나 patcher KV 재사용은
   별도 동등성 확인 전에는 허용하지 않는다. 이전·현재 시작점을 비교해
   공통된 닫힌 patch 앞부분만 구조적 후보로 남긴다. 숫자가 달라지는
   activation은 후보라 해도 출력 동일성 검증 없이 사용하지 않는다.
4. `past_key_values`를 무조건 켜지 않는다. local/global/decoder의 position,
   mask, cache 길이, patch 변경 시 무효화 지점을 명시하고 검증한다.
5. 우선 `incremental_generate`를 기존 HF `generate`와 분리해 빔 1·batch 1
   로 구현한다. 기존 `decode_generated`와 EOS/길이 정책을 공유한다.

판정: 고정 fixture의 **매 바이트** entropy 경계, 다음 바이트 logits,
출력 token ID를 no-cache와 비교한다. BF16 logits 허용오차는 반복 no-cache
실측의 수치 변동을 보고 정하며, 첫 불일치 위치와 top-1 격차를 기록한다.
최종 token ID와 EOS 상태가 불일치하면 이 backend는 본 평가에 사용하지 않는다.

첫 A100 native 체크포인트 probe(915640)는 12문장 × 32 step의 다음 ID가
384/384 같았으나 forward 시간은 1.372배 개선에 그쳤다. EOS·beam4·전체
validation은 아직 검사하지 않았다. 특히 경계 변경 step이 기준보다 느려
KV tail 경로를 제거하고 전체 global 재계산으로 수정했다. 수정 후 A100
재검사(915655)는 동일 384/384 ID 일치, 1.463배 개선이었다. 경계 변경
역전은 거의 사라졌지만 2배 목표는 미달이다. 이 부분 검사를 단계 1 완료나
단계 3 채택으로 간주하지 않는다. 상세 측정은 `blt_hf/cache/DESIGN.md`에 있다.

선택형 decoder KV 재사용을 켠 세 번째 A100 검사(915710)는 같은 384/384
다음 ID가 일치했으나 forward 개선이 1.377배로 떨어졌고, 두 step에 큰
지연이 있었다. 원인은 미확정이며 decoder 자체의 결함으로 단정하지 않는다.
EOS·전체 생성·beam4는 계속 미검증이다. 사용자 지시에 따라 2배 목표만으로
시제품 적용을 배제한 결론을 철회한다. 두 지연의 분기와 단계별 비용을
반복 진단한 뒤 필요한 수정과 재실험을 진행한다.

네 번째 A100 진단(915864)에서 2290의 첫 decoder 재사용 지연은 569ms로
한 번 재현되고 같은 접두부의 후속 실행은 약 26ms였다. 297의 경계 변경은
patch 시작점 76 삽입으로 확인됐으며 263ms 지연은 반복되지 않았다.
GC·대규모 메모리 증가는 관측되지 않았고 정확한 첫 사용 내부 원인은
미확인이다. 캐시 분기 오류는 발견되지 않아 재사용 규칙 변경 근거는 없다.
다음은 첫 사용 비용을 따로 보고 선택형 beam-1 완성 생성과 전체 native
validation의 출력 일치·실제 시간을 측정하는 단계다.

선택형 `global-prefix-greedy-v1`은 decoder KV를 끄고 global 재사용만
연결했다. 먼저 길이별 12문장 완성 생성의 토큰·EOS·문자열과 시간부터
기준 backend와 대조하고, 통과하면 전체 native validation을 비교한다.
beam 4와 decoder KV는 이 경로의 검증 범위가 아니다.

A100 완성 생성 검사(915874)는 길이별 native 12문장 모두 token ID·EOS·
문자열 일치, EOS 종료, 기준 43.594초→global 캐시 29.711초(1.467배)였다.
첫 기준 문장의 초기 비용을 제외한 표본 속도비는 약 1.432배다. 다음에는
전체 native validation 2,634건을 동일 조건의 별도 평가 디렉터리에서
생성하고 전수 출력·실제 생성 시간을 대조한다.

전체 native validation 비교(916171)는 **32/2,634건(1.21%) 출력 불일치**,
기준 9,074.297초→global 캐시 6,620.644초(1.371배)였다. 동일 출력
조건은 아직 충족하지 않는다. 단계 1의 다음 진단은 32건의 첫 토큰 분기에서
같은 접두부의 캐시 출력을 재생하고, global을 재계산했을 때 기준 ID가
복구되는지 판별하는 것이다. 직접 원인이 확인되면 재사용 조건을 수정한 뒤
새 backend 버전·새 EVAL_DIR에서 전수 비교한다. 기존 생성 결과는 보존한다.

첫 분기 07 재생(916179)은 17건에서 global skip이 기준 ID 이탈을 직접
유발한 것을 확인했다. 나머지 15건은 full forward도 저장된 HF 기준과
달랐다. `generate()`의 `logits_to_keep=1`과 v1 직접 forward의 기본값 0
차이를 08 진단에서 확인한 뒤, global 재사용 조건과 함께 수정할 범위를
정한다.

08(916207)에서 `logits_to_keep=1`의 full forward는 32/32 첫 분기의
HF 기준 ID를 반복 재현했다. 기존 v1은 보존하고 projection만 수정한
v2를 별도 backend로 만들었다. 09에서는 기존 불일치 32건+대조군 12건의
완성 출력을 검증한다. global skip 때문에 남는 차이는 별도 정책·실험으로
다루며 새 출력 identity를 사용한다.

09(916208)는 기존 불일치 32건 중 v2가 12건을 해결했고 20건이 남았다.
선택형 guarded backend는 상위 두 byte logit의 간격이 관찰된 BF16
drift에 비해 작을 때만 full forward로 재계산한다. 10에서는 기존 32건과
대조군 12건의 완성 출력·fallback 횟수·시간을 확인하고, 이후 전체 split
출력·성능을 별도 identity에서 검증한다.

10(916210)은 44/44 출력 일치, 2,505회 skip 후보 중 59회 full refresh였다.
44문장 생성 시간은 v2의 117.76초에서 guarded 124.36초로 늘었다. job
전체 시간 159→322초에는 생성 외 구간 지연과 첫 문장 55의 이상치가 있어
캐시 자체의 전수 속도 결론으로 사용하지 않는다. 원인 미확정 지연은 별도
기록으로 보류하고 11에서 기존 HF 기준 전체 2,634건과 guarded 새 출력을
비교한다. 순수 생성 시간·p50/p95·fallback 수와 전수 출력 일치를 확인해
실사용 속도 이득이 있으면 통합 validation 생성 backend에 연결한다.

## 단계 2: 빔 4·배치·재개

1. 빔 4에서 부모 빔 선택 시 entropy, patch, encoder/global/decoder 상태를
   함께 재정렬한다. 먼저 단순 `index_select`를 사용한다.
2. 서로 다른 patch 확정 시점과 EOS를 갖는 배치 1·4를 지원한다. 동일 길이
   묶음은 패딩 회피를 위해 유지하되, 속도 이득을 가정하지 않는다.
3. 검증 문장은 짧음/김, 한글 3바이트 경계, 복사 출력, 조기 EOS,
   생성 예산 소진, patch 경계 직전·직후, EOS 구간을 포함한다.
4. 중단·재개 때 cache 자체를 저장할 필요는 없다. 기존 평가의 원자적
   문장/배치 기록을 사용하고, 미완료 단위는 새로 생성한다.

판정: 동일 모델·입력·생성 조건에서 빔 1·4와 batch 1·4의 token ID,
최종 UTF-8, EOS/budget 상태가 기준 경로와 일치해야 한다. native 전체
validation에서도 mismatch를 전수 보고한다.

## 단계 3: 성능 판정·운영 통합

1. 동일 A100 자원과 같은 문장 목록에서 no-cache 대비 문장/초,
   생성 바이트/초, p50/p95 문장 시간, peak GPU memory를 측정한다.
   초기 로딩·warmup과 순수 생성 시간을 분리한다. 41분짜리 기존 통합
   validation과 비교할 때는 동일한 4 GPU 분담 조건으로 재실행한다.
2. 전체 native validation 생성 시간 **2배 단축은 성능 목표**다. 이 숫자
   하나만으로 개선된 캐시의 적용을 배제하지 않는다. 출력 일치, 실제 생성
   속도, 메모리, 6시간 job 한도를 확인하고 선택형 적용 범위를 결정한다.
   원인 미확정 지연은 먼저 진단하고 수정·재실험한다.
3. `generation_backend`에 별도 버전을 부여하고 eval manifest/fingerprint,
   validation 보고서, checkpoint 선택 기록에 반영한다. 새 `EVAL_DIR`과
   새 학습 `RUN_ID`에만 적용한다. 이미 실행 중인 Neuron 작업이 있을 때
   공유 checkout의 코드를 갱신하지 않는다.
4. 검증 완료 전에는 현재 학습·생성·평가 기본값을 변경하지 않는다.
   full-split test GLEU/M2 채점 규칙은 그대로 둔다. 필요하면 기존
   체크포인트의 validation만 새 backend로 재생성해 비교한다.

## 구현·검증 산출물

- 코드: `blt_hf/cache/`, `blt_hf/generation.py`의 선택형 backend,
  평가·통합 validation의 backend 인자 및 manifest 반영.
- 테스트: prefix 경계 안정성, 구성요소별 step 비교, 빔 상태 재정렬,
  batch 불변성, 중단 후 재개, 전체 native validation token ID 비교.
- 증거: 기준선/캐시 benchmark JSON, parity JSON, 성능 판정 기록,
  `blt_hf/NOTES.md`의 수정 diff·재현 명령. HF 원본이나 site-packages는
  직접 편집하지 않는다.

이 실행안은 확정되었으며 단계 0 결과에 따라 재사용 가능한 상태 범위를
기술적으로 수정할 수 있다. 설계 변경과 그 근거는 `blt_hf/cache/DESIGN.md`에
기록한다.
