import os
import torch

# === 기본 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ROUTER_MODEL_PATH = os.path.join(MODEL_DIR, "router_distilbert")

# === 하드웨어 설정 ===
# RTX 3060 활용
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 모델 설정 ===
# 1. Router: 논문에서 사용한 MPNet (SetFit 백본) 
ROUTER_BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

# SLLM 모델 (사용자님의 Qwen-0.6B 또는 유사 모델)
# 실제 파인튜닝된 경로가 있다면 그 경로를 입력하세요.
# mistralai/Mistral-7B-Instruct-v0.3
SLLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# === 라우팅 전략 (논문 핵심) ===
# MC Dropout 샘플링 횟수 (논문에서는 최대 20 사용 [cite: 389, 393])
MC_SAMPLES = 10

# 불확실성 임계값 (0.0 ~ 1.0)
# 일치율이 이 값보다 낮으면 불확실하다고 판단
CONFIDENCE_THRESHOLD = 0.8

# === 의도(Intent) 정의 ===
# 0, 1: SetFit(Router) 처리, 2: SLLM 처리, 3: OOS
ID2LABEL = {
    0: "FACTUAL_SPEC",       # 단순 스펙 질문
    1: "PROCEDURE_GUIDE",    # 절차 질문
    2: "COMPLEX_ANALYSIS",   # 복합 분석 (SLLM 필요)
    3: "OUT_OF_SCOPE"        # 도메인 밖
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}