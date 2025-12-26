# SetFit-Router: Hybrid Intent Classification System

반도체 패키징 비전 검사 시스템을 위한 **SetFit 기반 Intent Router**와 **SLLM(Small LLM)** 하이브리드 시스템

## Overview

이 시스템은 사용자 쿼리를 분류하고, **MC Dropout** 기반 불확실성 측정을 통해 적절한 처리 경로로 라우팅합니다.

```
[User Query]
     │
     ▼
┌─────────────────┐
│  SetFit Router  │  ← MC Dropout으로 불확실성 측정
│  (Intent 분류)   │
└────────┬────────┘
         │
    ┌────┴────┬────────────┬────────────┐
    ▼         ▼            ▼            ▼
[확실+도메인] [불확실]    [Vague]    [OOS]
    │         │            │            │
    ▼         ▼            ▼            ▼
  DB 검색   SLLM 처리   명확화 요청   거절
```

## Project Structure

```
ai_engine/
├── config/
│   └── settings.py          # 설정 파일 (모델 경로, 라벨 정의, 임계값)
├── src/
│   ├── router.py            # SetFit 기반 Intent Router (MC Dropout)
│   ├── hybrid_system.py     # 메인 하이브리드 시스템 (라우팅 로직)
│   ├── sllm_wrapper.py      # SLLM (Qwen/Mistral) 래퍼
│   └── csv_handler.py       # CSV 규칙 기반 핸들러
├── training/
│   └── train_router.py      # 라우터 학습 스크립트
├── models/
│   └── router_distilbert/   # 학습된 SetFit 모델
├── data/
│   └── raw/                 # 학습/테스트 데이터 (CSV)
└── main_test.py             # 테스트 스크립트
```

## Intent Labels (라벨 정의)

| ID | Label | Description |
|----|-------|-------------|
| 0 | BGA_QUESTION | BGA 관련 질문 |
| 1 | Calibration_QUESTION | Calibration 관련 질문 |
| 2 | common_prompt | 공통 프롬프트 (창이 안 열려있을 때) |
| 3 | ConfirmLog | Yes/No 확인 응답 |
| 4 | History_QUESTION | History 관련 질문 |
| 5 | LGA_QUESTION | LGA 관련 질문 |
| 6 | Light_QUESTION | Light 관련 질문 |
| 7 | Mapping_QUESTION | Mapping 관련 질문 |
| 8 | QFN_QUESTION | QFN 관련 질문 |
| 9 | Settings_QUESTION | Settings 관련 질문 |
| 10 | Strip_QUESTION | Strip 관련 질문 |
| 11 | vague | 애매모호한 질문 |
| 12 | OUT_OF_SCOPE | 도메인 밖 질문 |

## Routing Logic (라우팅 로직)

`hybrid_system.py`의 `process_query()` 메서드는 다음 순서로 처리합니다:

| Case | 조건 | 처리 | Intent |
|------|------|------|--------|
| 1 | `label_id == 12` | 도메인 외 질문 거절 | `OUT_OF_SCOPE` |
| 2 | `label_id == 11` | 명확화 요청 | `VAGUE` |
| 3 | `is_uncertain == True` | SLLM으로 전달 | 라우터 라벨 유지 |
| 4 | `label_id == 2` | 공통 프롬프트 응답 | `COMMON_PROMPT` |
| 5 | `label_id == 3` | 확인 응답 | `CONFIRM_LOG` |
| 6 | `label_id in [0,1,4~10]` | DB 검색 | 도메인별 Intent |

## Key Features

### 1. MC Dropout (Monte Carlo Dropout)
- 20회 샘플링으로 불확실성 측정
- `agreement_ratio`(일치율)가 `CONFIDENCE_THRESHOLD`(0.8) 미만이면 불확실로 판단
- 불확실한 경우 SLLM으로 라우팅

```python
# router.py - predict_mc_dropout()
for _ in range(settings.MC_SAMPLES):  # 20회
    # Dropout 활성화 상태로 추론
    predictions.append(pred_label)
```

### 2. Negative Data Augmentation
- KeyBERT로 핵심 키워드 추출
- 키워드 삭제/대체로 OOS 데이터 생성

```python
# train_router.py
if random.random() > 0.7:  # 30% 비율
    aug_text = generate_negative_augmentation_with_keybert(text)
    aug_data.append({"text": aug_text, "label": 12})  # OOS
```

### 3. Confidence & Uncertainty Score
- `confidence_score`: 일치율 (0.0 ~ 1.0)
- `uncertainty_score`: 불확실성 (1.0 - confidence_score)

```python
# 반환 예시
{
    "query": "BGA 티칭 창 열어",
    "detected_intent": "BGA_QUESTION",
    "confidence_pct": "95.0%",
    "uncertainty_pct": "5.0%",
    "is_uncertain": False,
    "routing_source": "DB (Domain: BGA)",
    "latency": "0.1234s"
}
```

## Installation

```bash
# 필수 패키지 설치
pip install torch transformers setfit datasets pandas keybert

# CUDA 지원 (RTX 3060 권장)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. 라우터 학습

```bash
cd ai_engine
python training/train_router.py
```

### 2. 테스트 실행

```bash
python main_test.py
```

### 3. 코드에서 사용

```python
from src.hybrid_system import HybridSystem

system = HybridSystem()
result = system.process_query("BGA 티칭 창 열어")

print(f"Intent: {result['detected_intent']}")
print(f"Confidence: {result['confidence_pct']}")
print(f"Response: {result['response']}")
```

## Configuration

`config/settings.py`에서 주요 설정 변경:

```python
# 모델 설정
ROUTER_BASE_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
SLLM_MODEL_ID = "Qwen/Qwen3-0.6B"

# MC Dropout 설정
MC_SAMPLES = 20                  # 샘플링 횟수
CONFIDENCE_THRESHOLD = 0.8       # 불확실성 임계값 (0.0 ~ 1.0)
```

## Output Example

```
--------------------------------------------------
Q: BGA 티칭 창 열어
Intent: BGA_QUESTION
Confidence: 100.0% | Uncertainty: 0.0% (Uncertain: False)
Source: DB (Domain: BGA)
Latency: 0.0892s
Response: [BGA DB 검색] 'BGA 티칭 창 열어'에 대한 정보를 조회합니다.
--------------------------------------------------
Q: 오늘 날씨 어때?
Intent: OUT_OF_SCOPE
Confidence: 85.0% | Uncertainty: 15.0% (Uncertain: False)
Source: Router (Blocked OOS)
Latency: 0.0756s
Response: 죄송합니다. 저는 반도체 패키징 전문가라 그 질문에는 답할 수 없습니다.
--------------------------------------------------
```

## Tech Stack

- **SetFit**: Few-shot text classification
- **Sentence Transformers**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- **SLLM**: `Qwen/Qwen3-0.6B` (또는 `Mistral-7B`)
- **KeyBERT**: 키워드 추출 (데이터 증강용)
- **PyTorch**: Deep Learning Framework
