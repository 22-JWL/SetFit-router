from pydoc import text
import sys
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


# === 증폭 설정 (여기서 배수를 조절하세요) ===
AMPLIFICATION_FACTOR = 5  # 데이터를 5배로 뻥튀기
OOS_GENERATION_PROB = 0.3 # 30% 확률로 부정 데이터(OOS) 생성

def load_csv_data(file_path, label_id):
    """
    CSV 파일을 읽어 B열(2번째 컬럼)의 데이터를 지정된 라벨로 로드합니다.
    """
    data_list = []
    if not os.path.exists(file_path):
        print(f"[Warning] Training file not found: {file_path}")
        return data_list

    try:
        # 인코딩 자동 감지 (UTF-8 -> CP949)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')
        
        # B열 데이터 추출 (인덱스 1)
        count = 0
        for _, row in df.iterrows():
            if len(row) >= 2 and pd.notna(row.iloc[1]):
                text = str(row.iloc[1]).strip()
                if text:
                    data_list.append((text, label_id))
                    count += 1
        
        print(f">>> Loaded {count} examples for Label {label_id} from {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        
    return data_list

def apply_minor_noise(text):
    """
    [Positive Augmentation]
    데이터 증폭 시, 문장을 아주 약간 변형하여 모델의 일반화 성능을 높임
    """
    # 1. 랜덤하게 소문자로 변환
    if random.random() > 0.5:
        text = text.lower()
    
    # 2. 끝에 마침표나 물음표가 있으면 랜덤하게 제거
    if random.random() > 0.5 and text[-1] in ['.', '?', '!']:
        text = text[:-1]
        
    # 3. 양옆 공백 실수 시뮬레이션
    if random.random() > 0.8:
        text = " " + text
        
    return text

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
    
    # === 1. CSV 파일 경로 정의 ===
    # ai_engine/data/raw 폴더 안에 있다고 가정
    base_data_path = os.path.join(settings.BASE_DIR, "data", "raw")
    single_csv_path = os.path.join(base_data_path, "single.csv")
    composite_csv_path = os.path.join(base_data_path, "composite.csv")


    raw_data = []
#--------------------------
    # # 1. 학습 데이터 (반도체 패키징 예시)
    # # 실제로는 파일에서 로드하세요.
    # # 튜플 형태: (Text, Label_ID)
    # raw_data = [
    #     ("What is the melting point of SAC305?", 0),
    #     ("Process flow for wafer bumping.", 1),
    #     ("Analyze the root cause of bump crack.", 2),
    #     ("Show me the weather forecast.", 3), # OOS (Out-of-Scope)
    #     ("Define Underfill in packaging.", 0),
    #     ("Steps for die attach process.", 1),
    #     ("Relationship between CTE mismatch and warpage.", 2),
    #     ("Who is the president of USA?", 3),  # OOS
    # ] * 10  # 데이터 증폭 (데모용)(논문에서는  * 20)
#----------------------

# === 2. CSV 데이터 로드 ===
    # [요청사항] single.csv (B열) -> Label 0 (FACTUAL_SPEC)
    single_data = load_csv_data(single_csv_path, 0)     # Label 0

    # [요청사항] composite.csv (B열) -> Label 2 (COMPLEX_ANALYSIS)
    composite_data = load_csv_data(composite_csv_path, 2) # Label 2

    # === 3. 부족한 라벨 보강 (Label 1 & 기본 데이터) ===
    # CSV에 없는 Label 1(절차)이나 기본 패턴이 부족할 수 있으므로 최소한의 데이터를 하드코딩으로 추가
    basic_data = [
        # Label 1: 절차/가이드 (CSV가 없으므로 수동 추가 필요)
        ("Process flow for wafer bumping.", 1),
        ("Steps for die attach process.", 1),
        ("How to clean flux residue?", 1),
        ("Reflow profile setting guide.", 1),
        ("Wire bonding parameters setup.", 1),
        
        # Label 3: OOS (기본 시드 데이터)
        ("Show me the weather forecast.", 3),
        ("Who is the president of USA?", 3),
        ("Recommend a dinner menu.", 3),
    ]
    raw_data += basic_data

# === 3. [핵심] 데이터 증폭 (Amplification) ===
    print(f"\n>>> Amplifying Data by Factor: {AMPLIFICATION_FACTOR}x")
    
    # 단순히 리스트를 곱하는 게 아니라, 섞어서 증폭 리스트에 넣음
    raw_data += (single_data * AMPLIFICATION_FACTOR)
    raw_data += (composite_data * AMPLIFICATION_FACTOR)
    raw_data += (basic_data * AMPLIFICATION_FACTOR) # 기본 데이터도 증폭

    print(f">>> Total examples after amplification: {len(raw_data)}")


    
    
    # Negative Augmentation 적용 (데이터 증강)
# === 4. 증강 적용 (Augmentation) ===
    final_train_data = []
    
    for text, label in raw_data:
        # [Positive Augmentation]
        # 증폭된 데이터들이 서로 조금씩 다르도록 노이즈 추가
        noisy_text = apply_minor_noise(text)
        final_train_data.append({"text": noisy_text, "label": label})
        
        # [Negative Augmentation]
        # 일정 확률로 OOS 데이터 생성 (문장 파괴) -> Label 3
        if random.random() < OOS_GENERATION_PROB: 
            oos_text = generate_negative_augmentation(text)
            final_train_data.append({"text": oos_text, "label": 3})

    # 데이터셋 변환 및 셔플링
    df = pd.DataFrame(final_train_data)
    df = df.sample(frac=1).reset_index(drop=True) # 전체 셔플

    train_dataset = Dataset.from_pandas(df)
    print(f">>> Final Dataset Size for Training: {len(train_dataset)}")

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
        logging_steps=50,
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
    print(">>> Training Router with CSV Data...")
    
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