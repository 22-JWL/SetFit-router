import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.router import UncertaintyRouter
from src.sllm_wrapper import SLLMWrapper
from src.csv_handler import CSVHandler
from config import settings

class HybridSystem:
    def __init__(self):

        # 0. CSV 핸들러 (Rule-based)
        self.csv_handler = CSVHandler()

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
        source = ""

<<<<<<< HEAD
        # 기본값 초기화 (에러 방지용)
        final_response = {
            "answer": "죄송합니다. 처리 중 오류가 발생했습니다.",
            "intent": "ERROR"
        }
        uncertainty_score = 0.0

        # === [Step 0] CSV 규칙 매칭 (최우선 순위) ===
        csv_result = self.csv_handler.check_and_execute(query)

        if csv_result:
            # CSV 규칙에 걸리면 바로 리턴 (SLLM/Router 생략)
            source = "CSV Rule (External API)"
            final_response = {
                "answer": csv_result,
                "intent": "API_EXECUTION"
            }
            uncertainty_score = 0.0 # 100% 확실
=======
        # Step 2: 라우팅 결정 [cite: 121]
        # Case A: 불확실하거나(Uncertain), 의도가 '복합 분석(Complex)'인 경우 -> SLLM
        if routing_result["is_uncertain"] or routing_result["final_label_id"] == 2:
            source = "SLLM (Reason: " + ("Uncertain" if routing_result["is_uncertain"] else "Complex Intent") + ")"
            print(f"🚀 Routing to SLLM... ({source})")
            answer = self.sllm.generate_response(query)
            final_response = {"answer": answer, "intent": routing_result["final_label"]}

        # Case B: 확실하고(Certain), 단순 질문인 경우 -> 라우터/DB 처리
>>>>>>> parent of b957b49 (feat : OOS (도메인 밖) -> 즉시 거절 로직 추가)
        else:
            # === [Step 1] 기존 라우터 & SLLM 로직 ===
            # (CSV에 없으면 AI 엔진 가동)
            try:
                # Step 1: 라우터로 불확실성 체크 (MC Dropout)
                mc_preds = self.router.predict_mc_dropout(query)
                routing_result = self.router.check_uncertainty(mc_preds)

                # 불확실성 점수 계산
                uncertainty_score = 1.0 - routing_result.get("agreement_ratio", 1.0)
            


                # Step 2: 라우팅 결정 [cite: 121]
                # Case 1: OOS (도메인 밖) -> 즉시 거절
                if routing_result["final_label_id"] == 3:
                    source = "Router (Blocked OOS)"
                    print(f"🛑 Blocked OOS query... ({source})")
                    final_response = {
                        "answer": "죄송합니다. 저는 반도체 패키징 전문가라 그 질문에는 답할 수 없습니다.",
                        "intent": "OUT_OF_SCOPE"
                    }

                # Case 2: 불확실하거나(Uncertain), 의도가 '복합 분석(Complex)'인 경우 -> SLLM
                elif routing_result["is_uncertain"] or routing_result["final_label_id"] == 2:
                    source = "SLLM (Reason: " + ("Uncertain" if routing_result["is_uncertain"] else "Complex Intent") + ")"
                    print(f"🚀 Routing to SLLM... ({source})")
                    answer = self.sllm.generate_response(query)
                    final_response = {"answer": answer, "intent": routing_result["final_label"]}

                # Case 3: 확실하고(Certain), 단순 질문인 경우 -> 라우터/DB 처리, 로컬 DB/규정집 검색
                # else:
                #     source = "Router/DB (Reason: Certain & Simple)"
                #     print(f"✅ Handling locally... ({source})")
                #     # 실제로는 여기서 SQL DB나 미리 정의된 매뉴얼을 조회합니다.
                #     dummy_db_answer = f"[DB 검색 결과] '{query}'에 대한 스펙/절차 정보를 표시합니다."
                #     final_response = {"answer": dummy_db_answer, "intent": routing_result["final_label"]}
                
                # uncertainty_score = 1.0 - routing_result["agreement_ratio"]
            except Exception as e:
                print(f"[Error] Processing query failed: {e}")
                final_response["answer"] = f"AI 처리 중 오류 발생: {str(e)}"
        
        latency = time.time() - start_time

        return {
            "query": query,
            "response": final_response["answer"],
            "detected_intent": final_response["intent"],
            "routing_source": source,
            "uncertainty_score": uncertainty_score,
            "latency": f"{latency:.4f}s"
        }