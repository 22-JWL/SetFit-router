from pydoc import text
import sys
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def generate_negative_augmentation(text):
    """
    논문의 Negative Data Augmentation
    키워드를 삭제하거나 랜덤 문자열로 대체하여 OOS 데이터 생성
    """
    """논문의 Negative Data Augmentation 구현 [cite: 112]"""
    words = text.split()
    if len(words) < 2: return text
    idx = random.randint(0, len(words)-1)
    # 50% 확률로 단어 삭제, 50% 확률로 랜덤 마스킹
    # 논문 방식: (a) 단어 제거 또는 (b) 랜덤 문자열로 대체 [cite: 112]
    if random.random() > 0.5:
        words.pop(idx)# 단어 삭제
    else:
        words[idx] = "XZY_UNK"# 랜덤 문자열로 대체
    return " ".join(words)



def main():
    print(f">>> Device: {settings.DEVICE}")
    # 1. 학습 데이터 (반도체 패키징 예시)
    # 실제로는 파일에서 로드하세요.
    # 튜플 형태: (Text, Label_ID)
    raw_data = [
        ("What is the melting point of SAC305?", 0),
        ("Process flow for wafer bumping.", 1),
        ("Analyze the root cause of bump crack.", 2),
        ("Show me the weather forecast.", 3), # OOS (Out-of-Scope)
        ("Define Underfill in packaging.", 0),
        ("Steps for die attach process.", 1),
        ("Relationship between CTE mismatch and warpage.", 2),
        ("Who is the president of USA?", 3),  # OOS
    ] * 10  # 데이터 증폭 (데모용)(논문에서는  * 20)

    # Negative Augmentation 적용 (데이터 증강)
    aug_data = []
    for text, label in raw_data:
        # 정상 데이터 추가
        aug_data.append({"text": text, "label": label})
        # OOS 데이터 증강 (기존 문장 망가뜨리기 -> 라벨 3(OOS) 부여)
        if random.random() > 0.7:  # 30% 비율로 생성
            aug_text = generate_negative_augmentation(text)
            aug_data.append({"text": aug_text, "label": 3})  # 강제로 OOS 라벨 부여

    df = pd.DataFrame(aug_data)
    train_dataset = Dataset.from_pandas(df)

# 3. Pandas DataFrame -> HuggingFace Dataset 변환
    # [수정 포인트] 컬럼명이 'text', 'label'인지 명확히 확인

    print(f">>> Dataset Features: {train_dataset.features}") # 디버깅용 출력

    # SetFit 모델 로드 (MPNet 백본) [cite: 108, 139]
    print(f">>> Loading SetFit Model: {settings.ROUTER_BASE_MODEL}")
    model = SetFitModel.from_pretrained(
        settings.ROUTER_BASE_MODEL,
        use_differentiable_head=True,  # <--- 이 옵션이 필수입니다!
        head_params={"out_features": len(settings.ID2LABEL)}
    )

    # 4. 모델을 GPU로 이동
    model.to(settings.DEVICE)

    # 5. 학습 설정 (TrainingArguments 사용 권장)
    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,# Head 학습 Epoch
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=10,
    )

    # 6. SetFit Trainer 초기화 (Deprecated된 SetFitTrainer 대신 Trainer 사용)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"} # [수정 포인트] 컬럼 매핑 명시
    )

    print(">>> Training Router with Differentiable Head...")
    
    # Differentiable head를 쓸 때는 model.freeze()로 바디를 고정하고 헤드만 학습하는 것이 일반적입니다.
    #model.freeze("body") 
    trainer.train()
    #model.unfreeze("body") # 필요시 전체 미세조정

# 7. 모델 저장
    # 폴더가 없으면 생성
    os.makedirs(settings.ROUTER_MODEL_PATH, exist_ok=True)
    model.save_pretrained(settings.ROUTER_MODEL_PATH)
    print(f">>> Model saved to {settings.ROUTER_MODEL_PATH}")


if __name__ == "__main__":
    main()