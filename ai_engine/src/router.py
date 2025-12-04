import torch
import torch.nn.functional as F
from collections import Counter
from setfit import SetFitModel # [중요] SetFitModel import 확인
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class UncertaintyRouter:
    def __init__(self, model_path=None):
        self.device = settings.DEVICE
        
        path = model_path if model_path else settings.ROUTER_BASE_MODEL
        print(f"Loading Router Model (SetFit) from: {path}")
        
        # SetFit 모델 로드
        # 저장된 모델을 로드할 때는 자동으로 config를 읽어오므로 
        # use_differentiable_head=True가 적용된 상태로 로드됩니다.
        self.model = SetFitModel.from_pretrained(path).to(self.device)

    def predict_mc_dropout(self, text):
        """
        [수정됨] 논문 방식의 MC Dropout 구현
        - 논문: "dropout across hidden and attention layers in the backbone"
        - 핵심: Backbone(Body)을 Train 모드로 두고 매번 임베딩을 다시 계산해야 함.
        """
        """
        SetFit 모델에 대한 MC Dropout 구현.
        SetFit은 (Embedding Body) + (Classification Head)로 구성됩니다.
        Head 부분의 Dropout을 활성화하여 추론합니다.
        """
        # 1. 텍스트 토큰화 (한 번만 수행)
        inputs = self.model.model_body.tokenizer(
            [text], 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        # [핵심 1] Body와 Head 모두 Train 모드로 전환 (Dropout 활성화)
        # SentenceTransformer의 encode() 메서드는 내부적으로 eval()을 강제할 수 있으므로
        # forward()를 직접 호출하거나 모드를 수동으로 제어해야 합니다.
        self.model.model_body.train() 
        self.model.model_head.train()

        predictions = []
        
        # [핵심 2] Gradients는 계산하지 않되(No update), Dropout은 켜둔 상태로 루프
        with torch.no_grad():
            for _ in range(settings.MC_SAMPLES):
                # A. Body 통과 (매번 다른 Dropout 마스크 적용됨 -> 다른 임베딩 생성)
                # SentenceTransformer 모델의 forward 호출
                features = self.model.model_body(inputs) 
                embeddings = features['sentence_embedding']

                # B. Head 통과
                outputs = self.model.model_head(embeddings)
                
                # 튜플 처리
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # C. 결과 저장
                probs = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                predictions.append(pred_label)

        # 3. 추론 후 안전하게 Eval 모드로 복귀
        self.model.model_body.eval()
        self.model.model_head.eval()

        return predictions

    def check_uncertainty(self, predictions):
        """
        불확실성 판단 (논문과 동일)
        """
        counts = Counter(predictions)
        # 가장 많이 예측된 라벨 찾기
        if not counts:
             return {"is_uncertain": True, "final_label": None}
             
        most_common_label, frequency = counts.most_common(1)[0]
        agreement_ratio = frequency / len(predictions)

        # 설정된 임계값보다 일치율이 낮으면 '불확실'로 판단 (LLM으로 라우팅)
        is_uncertain = agreement_ratio < settings.CONFIDENCE_THRESHOLD

        return {
            "is_uncertain": is_uncertain,
            "final_label_id": most_common_label,
            "final_label": settings.ID2LABEL[most_common_label],
            "agreement_ratio": agreement_ratio,
            "raw_predictions": predictions
        }