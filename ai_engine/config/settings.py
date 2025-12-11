import os
import torch

# === 기본 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ROUTER_MODEL_PATH = os.path.join(MODEL_DIR, "router_distilbert")

# === [추가] CSV 기반 Rule 설정 ===
CSV_FILE_PATH = os.path.join(BASE_DIR, "data", "raw", "single.csv")

# 호출할 외부 API 서버
EXTERNAL_API_BASE_URL = "http://localhost:3000"

# === 하드웨어 설정 ===
# RTX 3060 활용
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 모델 설정 ===
# 1. Router: 논문에서 사용한 MPNet (SetFit 백본) 
ROUTER_BASE_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# SLLM 모델
# 실제 파인튜닝된 경로가 있다면 그 경로를 입력
# mistralai/Mistral-7B-Instruct-v0.3
# Qwen/Qwen3-0.6B
SLLM_MODEL_ID = "Qwen/Qwen3-0.6B"

# === 라우팅 전략 (논문 핵심) ===
# MC Dropout 샘플링 횟수 (논문에서는 최대 20 사용 [cite: 389, 393])
MC_SAMPLES = 20

# 불확실성 임계값 (0.0 ~ 1.0)
# 일치율이 이 값보다 낮으면 불확실하다고 판단
CONFIDENCE_THRESHOLD = 0.8

# === 의도(Intent) 정의 ===
# 0, 1: SetFit(Router) 처리, 2: SLLM 처리, 3: OOS
ID2LABEL = {
    0: "SINGLE_QUESTION",       # 단일 질문
    1: "COMPLEX_ANALYSIS",   # 복합 분석 (SLLM 필요)
    2: "OUT_OF_SCOPE"        # 도메인 밖
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}