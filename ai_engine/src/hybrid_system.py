import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 필요한 모듈 임포트

from src.router import UncertaintyRouter
from src.sllm_wrapper import SLLMWrapper
from src.csv_handler import CSVHandler
from config import settings
class HybridSystem:
    def __init__(self):

        # 0. CSV 핸들러 (Rule-based)
        # self.csv_handler = CSVHandler()

        # 1. 라우터 로드 (학습된 모델이 있으면 그것을 로드)
        if os.path.exists(settings.ROUTER_MODEL_PATH):
            self.router = UncertaintyRouter(settings.ROUTER_MODEL_PATH)
        else:
            print("Warning: Trained router not found. Using base model.")
            self.router = UncertaintyRouter()

        # 2. SLLM 로드 (메모리에 상주)
        self.sllm = SLLMWrapper()

    def process_query(self, query):
        start_time = time.time()

        # 기본값 초기화 (에러 방지용)
        final_response = {
            "answer": "죄송합니다. 처리 중 오류가 발생했습니다.",
            "intent": "ERROR"
        }
        source = ""
        uncertainty_score = 0.0

        # # Step 1: 라우터로 불확실성 체크 (MC Dropout)
        # mc_preds = self.router.predict_mc_dropout(query)
        # routing_result = self.router.check_uncertainty(mc_preds)

        # final_response = {}
        

        

        # === [Step 0] CSV 규칙 매칭 (최우선 순위) ===
        # csv_result = self.csv_handler.check_and_execute(query)

        # if csv_result:
        #     # CSV 규칙에 걸리면 바로 리턴 (SLLM/Router 생략 -> Latency 대폭 감소)
        #     latency = time.time() - start_time
        #     return {
        #         "query": query,
        #         "response": csv_result,
        #         "detected_intent": "API_EXECUTION",
        #         "routing_source": "CSV Rule (External API)",
        #         "uncertainty_score": 0.0, # 100% 확실
        #         "latency": f"{latency:.4f}s"
        #     }
        # # CSV에 없을 때만 실행
        mc_preds = self.router.predict_mc_dropout(query)
        routing_result = self.router.check_uncertainty(mc_preds)


        label_id = routing_result["final_label_id"]
        label_name = routing_result["final_label"]
        is_uncertain = routing_result["is_uncertain"]

        # ============================================================
        # Case 1: OUT_OF_SCOPE (라벨 12) - 도메인 밖 질문 즉시 거절
        # ============================================================
        if label_id == 12:
            source = "Router (Blocked OOS)"
            print(f"🛑 Blocked OOS query... ({source})")
            final_response = {
                "answer": "죄송합니다. 저는 반도체 패키징 전문가라 그 질문에는 답할 수 없습니다.",
                "intent": "OUT_OF_SCOPE"
            }

        # ============================================================
        # Case 2: vague (라벨 11) - 애매모호한 질문 -> SLLM으로 명확화 요청
        # ============================================================
        elif label_id == 11:
            source = "SLLM (Reason: Vague Query)"
            print(f"🤔 Vague query detected... ({source})")
            final_response = {
                "answer": "질문이 명확하지 않습니다. 좀 더 구체적으로 질문해 주시겠어요?",
                "intent": "VAGUE"
            }

        # ============================================================
        # Case 3: Uncertain (불확실) - 라우터가 확신하지 못함 -> SLLM 처리
        # ============================================================
        elif is_uncertain:
            source = "SLLM (Reason: Uncertain)"
            print(f"🚀 Routing to SLLM (Uncertain)... ({source})")
            final_response = {
                "answer": "LLM으로 넘어가서 분석",
                "intent": label_name
            }

        # ============================================================
        # Case 4: common_prompt (라벨 2) - 창이 안 열려있을 때 프롬프트
        # ============================================================
        elif label_id == 2:
            source = "Router (Common Prompt)"
            print(f"💬 Common prompt detected... ({source})")
            final_response = {
                "answer": "창이 안 열려있을 때 프롬프트.",
                "intent": "COMMON_PROMPT"
            }

        # ============================================================
        # Case 5: ConfirmLog (라벨 3) - Yes/No 확인 응답
        # ============================================================
        elif label_id == 3:
            source = "Router (Confirm Log)"
            print(f"✅ Confirm log detected... ({source})")
            final_response = {
                "answer": "확인되었습니다.",
                "intent": "CONFIRM_LOG"
            }

        # ============================================================
        # Case 6: 도메인 질문 (라벨 0, 1, 4~10) - 확실한 도메인 내 질문 -> DB 처리
        # BGA(0), Calibration(1), History(4), LGA(5), Light(6),
        # Mapping(7), QFN(8), Settings(9), Strip(10)
        # ============================================================
        else:
            # 도메인별 처리 로직
            domain_labels = {
                0: ("BGA", "BGA_QUESTION"),
                1: ("Calibration", "CALIBRATION_QUESTION"),
                4: ("History", "HISTORY_QUESTION"),
                5: ("LGA", "LGA_QUESTION"),
                6: ("Light", "LIGHT_QUESTION"),
                7: ("Mapping", "MAPPING_QUESTION"),
                8: ("QFN", "QFN_QUESTION"),
                9: ("Settings", "SETTINGS_QUESTION"),
                10: ("Strip", "STRIP_QUESTION"),
            }

            if label_id in domain_labels:
                domain_name, intent_name = domain_labels[label_id]
                source = f"DB (Domain: {domain_name})"
                print(f"📂 Domain query [{domain_name}]... ({source})")
                final_response = {
                    "answer": f"[{domain_name} DB 검색] '{query}'에 대한 정보를 조회합니다.",
                    "intent": intent_name
                }
            else:
                # 예상치 못한 라벨 - Fallback to SLLM
                source = "SLLM (Fallback)"
                print(f"⚠️ Unknown label {label_id}, fallback to SLLM... ({source})")
                final_response = {
                    "answer": "LLM으로 넘어가서 분석",
                    "intent": label_name
                }

        latency = time.time() - start_time
        confidence_score = routing_result["agreement_ratio"]
        uncertainty_score = 1.0 - confidence_score

        return {
            "query": query,
            "response": final_response["answer"],
            "detected_intent": final_response["intent"],
            "routing_source": source,
            "confidence_score": confidence_score,  # 일치율 (0.0 ~ 1.0)
            "confidence_pct": f"{confidence_score * 100:.1f}%",  # 퍼센트 형식
            "uncertainty_score": uncertainty_score,  # 불확실성 점수 (0.0 ~ 1.0)
            "uncertainty_pct": f"{uncertainty_score * 100:.1f}%",  # 퍼센트 형식
            "is_uncertain": routing_result["is_uncertain"],
            "latency": f"{latency:.4f}s"
        }