import sys
import os
import torch
from transformers import Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


# === 데이터셋 정의 ===
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def generate_negative_augmentation(text):
    """
    논문의 Negative Data Augmentation
    키워드를 삭제하거나 랜덤 문자열로 대체하여 OOS 데이터 생성
    """
    words = text.split()
    if len(words) < 2: return text

    # 50% 확률로 단어 삭제, 50% 확률로 랜덤 마스킹
    idx = random.randint(0, len(words) - 1)
    if random.random() > 0.5:
        words.pop(idx)  # 삭제
    else:
        words[idx] = "XZY_UNK"  # 랜덤 문자열 대체

    return " ".join(words)


def main():
    # 1. 학습 데이터 (반도체 패키징 예시)
    # 실제로는 파일에서 로드하세요.
    raw_data = [
                   ("What is the melting point of SAC305?", 0),
                   ("Procedure for wafer dicing.", 1),
                   ("Why did the bump crack happen?", 2),
                   ("Who won the baseball game?", 3),  # OOS
               ] * 10  # 데이터 증폭 (데모용)

    # Negative Augmentation 적용 (데이터 증강)
    aug_data = []
    for text, label in raw_data:
        # 정상 데이터 추가
        aug_data.append((text, label))
        # OOS 데이터 증강 (기존 문장 망가뜨리기 -> 라벨 3(OOS) 부여)
        if random.random() > 0.7:  # 30% 비율로 생성
            aug_text = generate_negative_augmentation(text)
            aug_data.append((aug_text, 3))  # 강제로 OOS 라벨 부여

    texts, labels = zip(*aug_data)

    # 2. 토크나이징
    tokenizer = DistilBertTokenizer.from_pretrained(settings.ROUTER_BASE_MODEL)
    encodings = tokenizer(list(texts), truncation=True, padding=True)
    dataset = IntentDataset(encodings, labels)

    # 3. 모델 준비
    model = DistilBertForSequenceClassification.from_pretrained(
        settings.ROUTER_BASE_MODEL,
        num_labels=len(settings.ID2LABEL)
    ).to(settings.DEVICE)

    # 4. 학습 설정
    training_args = TrainingArguments(
        output_dir=settings.ROUTER_MODEL_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(">>> Training Router Model with Negative Augmentation...")
    trainer.train()

    # 모델 저장
    model.save_pretrained(settings.ROUTER_MODEL_PATH)
    tokenizer.save_pretrained(settings.ROUTER_MODEL_PATH)
    print(f">>> Model saved to {settings.ROUTER_MODEL_PATH}")


if __name__ == "__main__":
    main()