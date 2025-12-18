import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd 
from config import settings
from src.hybrid_system import HybridSystem


def load_csv_data(file_path):
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
                    data_list.append((text))
                    count += 1
        
        print(f">>> Loaded {count} examples for from {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        
    return data_list

def main():
    print(">>> Initializing Hybrid SLLM System...")
    base_data_path = os.path.join(settings.BASE_DIR, "data", "raw")
    # single_csv_path = os.path.join(base_data_path, "single.csv")
    # composite_csv_path = os.path.join(base_data_path, "composite.csv")
    bga_data = load_csv_data(os.path.join(base_data_path, "bga.csv"))
    calibration_data = load_csv_data(os.path.join(base_data_path, "calibration.csv"))
    common_data = load_csv_data(os.path.join(base_data_path, "common_prompt.csv"))
    confirm_log_data = load_csv_data(os.path.join(base_data_path, "confirmlog.csv"))
    history_data = load_csv_data(os.path.join(base_data_path, "history.csv"))
    lga_data = load_csv_data(os.path.join(base_data_path, "lga.csv"))
    light_data = load_csv_data(os.path.join(base_data_path, "light.csv"))
    mapping_data = load_csv_data(os.path.join(base_data_path, "mapping.csv"))
    qfn_data = load_csv_data(os.path.join(base_data_path, "qfn.csv"))
    settings_data = load_csv_data(os.path.join(base_data_path, "settings.csv"))
    strip_data = load_csv_data(os.path.join(base_data_path, "strip.csv"))
    # vague_data = load_csv_data(os.path.join(base_data_path, "vague.csv"))
    test100_data = load_csv_data(os.path.join(base_data_path, "TestData100.csv"))



    system = HybridSystem()

    # single_data = load_csv_data(single_csv_path)
    # composite_data = load_csv_data(composite_csv_path)
    # synthesize_data = (
    #     bga_data + calibration_data + common_data + confirm_log_data +
    #     history_data + lga_data + light_data + mapping_data +
    #     qfn_data + settings_data + strip_data
    # )
    

    test_queries = [# 단순 (Router 예상) # OOS (Router가 OOS 혹은 불확실로 잡음)
    ]

    # test_queries += single_data
    # test_queries += composite_data
    test_queries += test100_data



    print(f"\n>>> 총 테스트할 문장 개수: {len(test_queries)}")
    #print(f">>> composite 데이터 샘플 확인 (마지막 3개): {test_queries[-3:]}")

    print("\n>>> Starting Test Loop\n")
    for q in test_queries:
        result = system.process_query(q)
        print("-" * 50)
        print(f"Q: {result['query']}")
        print(f"Intent: {result['detected_intent']}")
        print(f"Confidence: {result['confidence_pct']} | Uncertainty: {result['uncertainty_pct']} (Uncertain: {result['is_uncertain']})")
        print(f"Source: {result['routing_source']}")
        print(f"Latency: {result['latency']}")
        print(f"Response: {result['response']}")
        print("-" * 50)


if __name__ == "__main__":
    main()