# blt_transformers

Meta의 BLT(Byte Latent Transformer) 1B 가중치를 `transformers>=5.0.0`의 `BltForCausalLM`으로
변환·로딩·실행하는 레퍼런스 코드입니다. 필요한 작업을 `blt_transformers` 패키지로 묶었습니다.

- 검증 환경: Python 3.11, `torch==2.11.0+cu128`, `transformers==5.16.1`, BF16 GPU
  (RTX 5090, A100, H200)
- 원본 BLT 구현과 hidden/logits 값이 같은지는 검증하지 않았습니다. 가중치 보존과 attention mask만 검사했습니다.

## 설치

```bash
pip install "git+https://github.com/esoterikosQ/blt_transformers.git"
```

## 1. 가중치 변환

`facebook/blt-1b`는 원본(Meta) 형식이라 Transformers에서 바로 읽을 수 없습니다. gated 저장소이므로
접근 승인을 받고 `hf auth login` 또는 `HF_TOKEN`으로 인증한 뒤 변환합니다. CPU에서 실행하며 여유 RAM이 약 28GiB 필요합니다.

```bash
python -m blt_transformers.convert --output-dir ./blt-1b-hf
```

변환 내용:

- `facebook/blt-1b`와 엔트로피 패처 `facebook/blt-entropy`를 고정 revision으로 내려받아 하나의 체크포인트로 합칩니다.
- 원본 파라미터 이름을 HF 이름으로 바꾸고, hash embedding 6개를 하나의 행렬로 합칩니다.
- `rms_norm_eps`, `rope_theta` 등 설정값을 HF 구성 키에 맞춰 보존합니다.
- byte tokenizer를 만들되 BOS/EOS 자동 삽입은 끕니다. upstream 변환 스크립트의 기본값은 자동 삽입입니다.
- 원본 attention 설정(`local_block_causal`, window 512, EOS id 등)을 `config.original_spec`에 기록합니다.

공개 변환본 `itazap/blt-1b-hf`도 경로 대신 넘겨 로딩할 수 있습니다. 다만 이 프로젝트에서는 검증하지 않았습니다.

## 2. 로딩

```python
from blt_transformers import load_model

model = load_model("./blt-1b-hf", device="cuda")
```

`load_model`은 다음 설정을 고정합니다.

| 설정 | 값 | 이유 |
|---|---|---|
| dtype | `torch.bfloat16` | 원본 가중치와 같은 정밀도 |
| `attn_implementation` | `"eager"` | eager와 sdpa의 greedy 출력이 달랐음 |
| `use_cache` | `False` | HF BLT 패처가 KV cache를 각 층에 전달하지 않아 cache가 동작하지 않음 |
| strict loading | 켬 | missing/unexpected/mismatched key가 있으면 예외 |

## 3. 실행

```python
import torch
from blt_transformers import encode, decode, generate

# 생성: 문자열을 넣으면 BOS를 붙여 인코딩하고, EOS 또는 길이 한도에서 멈춘다.
out = generate(model, "안녕하세요. 오늘 날씨가", max_new_bytes=64)
print(out.text, out.finished)

# 학습/손실: labels는 input_ids와 같은 위치로 넣는다(모델이 내부에서 한 칸 shift).
ids = torch.tensor([encode("오늘 날씨가 좋아요.", eos=True)], device="cuda")
loss = model(input_ids=ids, labels=ids, use_cache=False).loss
```

토큰 규칙은 다음과 같습니다.

| id | 의미 |
|---|---|
| 0 | `<boe>` |
| 1 | BOS |
| 2 | EOS |
| 3 | PAD |
| 4–259 | UTF-8 byte + 4 |

BOS와 EOS는 직접 붙입니다(`encode(text, bos=True, eos=False)`). `generate`는 0·1·3번 토큰의 생성을 막고,
`decode`는 EOS 이후를 버립니다.

## 주의사항

- **padding 미검증:** batch는 1개 또는 길이가 같은 입력만 사용합니다.
- **입력 길이:** 2048 byte 이내로 유지하고, 넘으면 자르지 말고 실패로 처리합니다.
- **원본 attention mask:** HF 기본 구현은 로컬·엔트로피 모델에 일반 full-causal mask를 씁니다.
  원본 설정(window 512, EOS 구간 분리)을 적용하려면 `load_model(path, attention_mode="osc")`를 사용합니다.
  - 이 경로는 [blt_hf/patched/modeling_blt.py](blt_hf/patched/modeling_blt.py)의 수정 모델을 씁니다.
  - `blt_transformers.convert`로 만든 체크포인트가 필요합니다.
  - `transformers==5.16.1`에서만 검증했습니다.
  - 원본 대비 변경은 [modeling_blt_osc.diff](blt_hf_checks/manifests/modeling_blt_osc.diff)에 있습니다.
- **생성 속도:** cache가 없어 바이트마다 전체 입력을 다시 계산합니다. A100에서 측정한 값은 바이트당 약 44ms입니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `blt_transformers/` | 위 1–3단계 라이브러리 |
| `blt_hf/`, `blt_hf_checks/`, `scripts/`, `tests/` | 이 코드로 수행한 한국어 GEC 파인튜닝 실험과 검증 코드 |

GEC 실험의 실행 안내는 [blt_hf/NEURON.md](blt_hf/NEURON.md)에 있습니다. 외부 코드의 출처와 라이선스는
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 따릅니다.
