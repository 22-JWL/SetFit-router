import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

# 상위 폴더 경로 추가 (설정 파일 import용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class UncertaintyRouter:
    def __init__(self, model_path=None):
        self.device = settings.DEVICE

        # 학습된 모델이 없으면 베이스 모델 로드 (최초 실행 시)
        path = model_path if model_path else settings.ROUTER_BASE_MODEL
        print(f"Loading Router Model from: {path}")

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=len(settings.ID2LABEL)
        ).to(self.device)

    def predict_mc_dropout(self, text):
        """
        논문의 MC Sampling 로직 [cite: 122, 388]
        모델을 train 모드로 두어 Dropout을 켠 상태로 여러 번 추론합니다.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # [핵심] Dropout 활성화를 위해 train 모드로 전환
        self.model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(settings.MC_SAMPLES):
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                predictions.append(pred_label)

        # 추론 후 다시 eval 모드로 복귀
        self.model.eval()

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