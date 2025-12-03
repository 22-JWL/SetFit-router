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

    def generate_response(self, query):
        """
        반도체 패키징 전문가 페르소나를 주입하여 답변 생성
        """
        messages = [
            {"role": "system",
             "content": "You are an expert AI assistant specialized in Semiconductor Packaging processes."},
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