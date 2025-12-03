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
        논문의 MC Sampling 로직 [cite: 122, 388]
        모델을 train 모드로 두어 Dropout을 켠 상태로 여러 번 추론합니다.
        """
        """
        SetFit 모델에 대한 MC Dropout 구현.
        SetFit은 (Embedding Body) + (Classification Head)로 구성됩니다.
        Head 부분의 Dropout을 활성화하여 추론합니다.
        """
        # 1. 먼저 임베딩을 생성합니다 (Body는 고정)
        embeddings = self.model.model_body.encode([text], convert_to_tensor=True, device=self.device)

        # [핵심] Dropout 활성화를 위해 train 모드로 전환
        # 2. Classification Head를 Train 모드로 전환 (Dropout 활성화)
        self.model.model_head.train()

        predictions = []
        with torch.no_grad():
            for _ in range(settings.MC_SAMPLES):
                # Head 통과
                outputs = self.model.model_head(embeddings)
                
                # [수정된 부분] 튜플이면 첫 번째 요소(logits)만 추출
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Softmax 계산
                probs = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                predictions.append(pred_label)

        # 추론 후 다시 eval 모드로 복귀
        # Eval 모드 복귀
        self.model.model_head.eval()

        return predictions

    def check_uncertainty(self, predictions):
        """
        불확실성 판단 [cite: 129, 390]
        예측값들의 분산(여기서는 빈도)을 보고 결정합니다.
        """
        counts = Counter(predictions)
        most_common_label, frequency = counts.most_common(1)[0]

        agreement_ratio = frequency / len(predictions)

        # 논문: 예측이 갈리면(variance가 크면) 불확실함
        is_uncertain = agreement_ratio < settings.CONFIDENCE_THRESHOLD

        return {
            "is_uncertain": is_uncertain,
            "final_label_id": most_common_label,
            "final_label": settings.ID2LABEL[most_common_label],
            "agreement_ratio": agreement_ratio,
            "raw_predictions": predictions
        }