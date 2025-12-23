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
# BAAI/bge-m3
# snunlp/KR-SBERT-V40K-klueNLI-augSTS
# nlpai-lab/KURE-v1
# BAAI/bge-m3
ROUTER_BASE_MODEL = "nlpai-lab/KURE-v1"

# SLLM 모델
# 실제 파인튜닝된 경로가 있다면 그 경로를 입력
# mistralai/Mistral-7B-Instruct-v0.3
# Qwen/Qwen3-0.6B
SLLM_MODEL_ID = "Qwen/Qwen3-0.6B"

# === 라우팅 전략 ===
# MC Dropout 샘플링 횟수 (논문에서는 최대 20 사용 [cite: 389, 393])
MC_SAMPLES = 20

# 불확실성 임계값 (0.0 ~ 1.0)
# 일치율이 이 값보다 낮으면 불확실하다고 판단
CONFIDENCE_THRESHOLD = 0.65

# === 의도(Intent) 정의 ===
# 0, 1: SetFit(Router) 처리, 2: SLLM 처리, 3: OOS
ID2LABEL = {
    0: "BGA_QUESTION",   # BGA 질문
    1: "Calibration_QUESTION",   # Calibration 질문
    2: "common_prompt",   # 공통 프롬프트
    3: "ConfirmLog",   # yes/no
    4: "History_QUESTION",   # History 질문
    5: "LGA_QUESTION",   # LGA 질문
    6: "Light_QUESTION",   # Light 질문
    7: "Mapping_QUESTION",   # Mapping 질문
    8: "QFN_QUESTION",   # QFN 질문
    9: "Settings_QUESTION",   # Settings 질문
    10: "Strip_QUESTION",   # Strip 질문
    11: "vagueWindow", # 창 없는 질문
    12: "vagueValue",# value없는 질문
    13: "OUT_OF_SCOPE" # 도메인 밖
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}