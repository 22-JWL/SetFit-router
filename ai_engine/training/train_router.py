from pydoc import text
import sys
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from keybert import KeyBERT
from datasets import Dataset
import pandas as pd
import random

# KeyBERT 모델 로드
# 가벼운 모델을 원하면 'paraphrase-multilingual-MiniLM-L12-v2' 사용
kw_model = KeyBERT(model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


AMPLIFICATION_FACTOR = 30  # 데이터를 30배로 뻥튀기


def load_csv_data(file_path, label_id):
    """
    CSV 파일을 읽어 B열(2번째 컬럼)의 데이터를 지정된 라벨로 로드
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


def generate_negative_augmentation_with_keybert(text):
    """
    논문의 Negative Data Augmentation
    키워드를 삭제하거나 랜덤 문자열로 대체하여 OOS 데이터 생성
    """
    """
    KeyBERT를 사용하여 핵심 키워드를 추출하고 변형하는 정석 구현
    """
    words = text.split()
    if len(words) < 2: return text
    
    # 1. 키워드 추출 (상위 1개)
    # 한국어의 경우 형태소 분석기를 쓰지 않으면 조사가 붙어 나올 수 있지만, 
    # OOS 생성용으로는 단순 split 매칭도 충분히 효과적.
    # keyphrase_ngram_range=(1, 1): 추출할 키워드의 길이를 지정합니다. (1, 1)은 한 단어짜리 키워드(unigram)만 추출하라
   # stop_words=None: 불용어(stop words, 예를 들어 "은", "는", "이", "가"와 같은 의미 없는 단어)를 필터링하지 않겠다.
   # top_n=1: 추출된 키워드 중에서 가장 점수가 높은(가장 관련성 높은) 키워드를 1개만 반환하라.
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=1)
    
    target_idx = -1


    # 키워드가 추출되었고, 그 키워드가 문장 내에 존재한다면 인덱스 찾기
    if keywords:
        keyword = keywords[0][0]  # ('word', score) 튜플에서 단어만 추출
        for i, word in enumerate(words):
            # 단순 포함 관계 확인 (조사 등 고려)
            if keyword in word:
                target_idx = i
                break

    # 키워드를 못 찾았으면 기존처럼 랜덤 선택 (Fallback)
    if target_idx == -1:
        target_idx = random.randint(0, len(words)-1)

    # 2. 변형 적용 (삭제 or 대체)
    if random.random() > 0.5:
        words.pop(target_idx) # 키워드 삭제
    else:
        words[target_idx] = "XZY_UNK" # 키워드를 랜덤 문자열로 대체

    return " ".join(words)



def main():
    print(f">>> Device: {settings.DEVICE}")
    # 1. 학습 데이터 (반도체 패키징 예시)
    
    # === 1. CSV 파일 경로 정의 ===
    # ai_engine/data/raw 폴더 내의 CSV 파일들
    base_data_path = os.path.join(settings.BASE_DIR, "data", "raw")
    bga_csv_path = os.path.join(base_data_path, "bga.csv")
    lga_csv_path = os.path.join(base_data_path, "lga.csv")
    light_csv_path = os.path.join(base_data_path, "light.csv")
    mapping_csv_path = os.path.join(base_data_path, "mapping.csv")
    qfn_csv_path = os.path.join(base_data_path, "qfn.csv")
    settings_csv_path = os.path.join(base_data_path, "settings.csv")
    strip_csv_path = os.path.join(base_data_path, "strip.csv")
    history_csv_path = os.path.join(base_data_path, "history.csv")
    common_csv_path = os.path.join(base_data_path, "common_prompt.csv")
    confirmlog_csv_path = os.path.join(base_data_path, "confirmlog.csv")
    calibration_csv_path = os.path.join(base_data_path, "calibration.csv")
    vague_csv_path = os.path.join(base_data_path, "vague.csv")
    oos_csv_path = os.path.join(base_data_path, "oos.csv")


    raw_data = []

# === 2. CSV 데이터 로드 ===
    # single.csv (B열) -> Label 0 (FACTUAL_SPEC)
    bga_data = load_csv_data(bga_csv_path, 0)     # Label 0

    # composite.csv (B열) -> Label 1 (COMPLEX_ANALYSIS)
    calibration_data = load_csv_data(calibration_csv_path, 1)     # Label 1
    common_data = load_csv_data(common_csv_path, 2)     # Label 2
    confirmlog_data = load_csv_data(confirmlog_csv_path, 3)     # Label 3
    history_data = load_csv_data(history_csv_path, 4)     # Label 4
    lga_data = load_csv_data(lga_csv_path, 5)     # Label 5
    light_data = load_csv_data(light_csv_path, 6)     # Label 6
    mapping_data = load_csv_data(mapping_csv_path, 7)     # Label 7
    qfn_data = load_csv_data(qfn_csv_path, 8)     # Label 8
    settings_data = load_csv_data(settings_csv_path, 9)     # Label 9
    strip_data = load_csv_data(strip_csv_path, 10)     # Label 10
    vague_data = load_csv_data(vague_csv_path, 11)     # Label 11
    oos_data = load_csv_data(oos_csv_path, 12)     # Label 12

# === 3. 부족한 라벨 보강 (Label 1 & 기본 데이터) ===
    # CSV에 없는 Label 1(절차)이나 기본 패턴이 부족할 수 있으므로 최소한의 데이터를 하드코딩으로 추가
    basic_data = [
        # Label : 절차/가이드 (CSV가 없으므로 수동 추가 필요)
        # Label 12: OOS (기본 시드 데이터)
    ]

    raw_data += (bga_data * AMPLIFICATION_FACTOR)
    raw_data += (calibration_data * AMPLIFICATION_FACTOR)
    raw_data += (confirmlog_data * AMPLIFICATION_FACTOR)
    raw_data += (history_data * AMPLIFICATION_FACTOR)
    raw_data += (lga_data * AMPLIFICATION_FACTOR)
    raw_data += (light_data * AMPLIFICATION_FACTOR)
    raw_data += (mapping_data * AMPLIFICATION_FACTOR)
    raw_data += (qfn_data * AMPLIFICATION_FACTOR)
    raw_data += (settings_data * AMPLIFICATION_FACTOR)
    raw_data += (strip_data * AMPLIFICATION_FACTOR)
    raw_data += (common_data * AMPLIFICATION_FACTOR)
    raw_data += (vague_data * AMPLIFICATION_FACTOR)
    raw_data += (oos_data * AMPLIFICATION_FACTOR)
    #raw_data += (basic_data * 10) # 기본 데이터도 증폭

    print(f">>> Total examples after amplification: {len(raw_data)}")

    # Negative Augmentation 적용 (데이터 증강)
    aug_data = []
    for text, label in raw_data:
        # 정상 데이터 추가
        aug_data.append({"text": text, "label": label})
        # OOS 데이터 증강 (기존 문장 망가뜨리기 -> 라벨 12(OOS) 부여)
        if random.random() > 0.7:  # 30% 비율로 생성
            aug_text = generate_negative_augmentation_with_keybert(text)
            aug_data.append({"text": aug_text, "label": 12})  # 강제로 OOS 라벨 부여

    df = pd.DataFrame(aug_data)
    train_dataset = Dataset.from_pandas(df)

# 3. Pandas DataFrame -> HuggingFace Dataset 변환
    # [수정 포인트] 컬럼명이 'text', 'label'인지 명확히 확인

    print(f">>> Dataset Features: {train_dataset.features}") # 디버깅용 출력

    # SetFit 모델 로드 (MPNet 백본)
    print(f">>> Loading SetFit Model: {settings.ROUTER_BASE_MODEL}")
    model = SetFitModel.from_pretrained(
        settings.ROUTER_BASE_MODEL,
        use_differentiable_head=True,  # <--- 이 옵션이 필수입니다!
        head_params={"out_features": len(settings.ID2LABEL)}
    )

    # 4. 모델을 GPU로 이동
    model.to(settings.DEVICE)

    # === [핵심 변경] Body(임베딩 모델) 얼리기 ===
    print(">>> Freezing Model Body (Training Head Only)...")
    
    # 1. Body의 모든 파라미터를 Freeze (학습 제외)
    for param in model.model_body.parameters():
        param.requires_grad = False
    model.model_body.eval()
        
    # 2. Head는 반드시 학습 가능해야 함 (Unfreeze)
    for param in model.model_head.parameters():
        param.requires_grad = True
    # Head는 학습 모드로 설정
    model.model_head.train()

    # 5. 학습 설정 (TrainingArguments 사용 권장)
    args = TrainingArguments(
        batch_size=32,            # 배치 크기
        num_epochs=3,             # 에폭 수
        head_learning_rate=1e-3,  # [수정] learning_rate -> head_learning_rate
        body_learning_rate=0.0,    # Body는 얼렸으므로 0
        num_iterations=0
        # logging_steps 등 미지원 인자 제거
    )

    # 6. SetFit Trainer 초기화 (Deprecated된 SetFitTrainer 대신 Trainer 사용)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"} # [수정 포인트] 컬럼 매핑 명시
    )

# === [핵심 2] 1단계(임베딩 학습) 강제 스킵 ===
    # train_embeddings 함수를 빈 함수(pass)로 바꿔치기합니다.
    # 이렇게 하면 1단계는 0초 만에 끝나고, 바로 2단계(Head 학습)로 넘어갑니다.
    def skip_embedding_training(*args, **kwargs):
        print(">>> [Info] Skipping Embedding Training (Stage 1)...")
        pass
    
    trainer.train_embeddings = skip_embedding_training

    print(">>> Training Router Head...")
    print("Unique labels in dataset:", set(train_dataset["label"]))
    trainer.train()
# 7. 모델 저장
    # 폴더가 없으면 생성
    os.makedirs(settings.ROUTER_MODEL_PATH, exist_ok=True)
    model.save_pretrained(settings.ROUTER_MODEL_PATH)
    print(f">>> Model saved to {settings.ROUTER_MODEL_PATH}")


if __name__ == "__main__":
    main()