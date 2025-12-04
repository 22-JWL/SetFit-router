import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class SLLMWrapper:
    def __init__(self):
        print(f"Loading SLLM ({settings.SLLM_MODEL_ID})... This may take a moment.")
        self.device = settings.DEVICE

        # RTX 3060 메모리 효율을 위해 fp16 사용
        self.tokenizer = AutoTokenizer.from_pretrained(settings.SLLM_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.SLLM_MODEL_ID,
            torch_dtype=torch.float16,
            device_map=self.device
        )
# ============================================
# class SLLMWrapper:
#     def __init__(self):
#         # 논문에서 사용한 Mistral-7B (메모리 문제로 4비트 로드 권장)
#         # 만약 기존 Qwen을 계속 쓰시려면 settings.py의 모델 ID만 유지하면 됩니다.
#         model_id = "mistralai/Mistral-7B-Instruct-v0.3" 
        
#         print(f"Loading LLM: {model_id}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
#         # 4비트 양자화 로드 (RTX 3060 최적화)
#         from transformers import BitsAndBytesConfig
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16
#         )
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             quantization_config=bnb_config,
#             device_map="auto"
#         )

# ============================================
    def generate_response(self, query):
        """
        반도체 패키징 전문가 페르소나를 주입하여 답변 생성
        """
        messages = [
            {"role": "system",
             "content": "너는 반도체 공정 비전 검사 시스템의 AI assistant야. 사용자가 말로 명령을 내리면 그에 맞게 url로만 답변해줘."},
            {"role": "user", "content": query}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response