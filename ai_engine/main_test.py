from src.hybrid_system import HybridSystem


def main():
    print(">>> Initializing Hybrid SLLM System...")
    system = HybridSystem()

    test_queries = [
        "BGA 티칭 창 열어줘",  # 단순 (Router 예상)
        "코너 각도 2.5로 만드러",  # 복합 (SLLM 예상)
        "오늘 날씨 어때?",  # OOS (Router가 OOS 혹은 불확실로 잡음)
    ]

    print("\n>>> Starting Test Loop\n")
    for q in test_queries:
        result = system.process_query(q)
        print("-" * 50)
        print(f"Q: {result['query']}")
        print(f"Intent: {result['detected_intent']}")
        print(f"Source: {result['routing_source']}")
        print(f"Latency: {result['latency']}")
        print(f"Response: {result['response']}")
        print("-" * 50)


if __name__ == "__main__":
    main()