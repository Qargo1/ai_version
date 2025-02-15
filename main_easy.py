import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel


# Параметры модели
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"


class SimpleChatBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.initialize_tokenizer()
        self.initialize_model()
        
        self.model.eval()
        
    def initialize_model(self):
        self.model = GPTQModel.load(self.model_name)
        
    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_response(self, user_input):
        # Токенизация входного текста
        input_tensor = self.tokenizer.apply_chat_template(user_input, add_generation_prompt=True, return_tensors="pt")

        outputs = self.model.generate(input_ids=input_tensor.to(self.model.device), max_new_tokens=512)
        return self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)


chat_bot = SimpleChatBot()

# Пример использования
user_input = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
            {"role": "user", "content": "How can I design a data structure in C++ to store the top 5 largest integer numbers?"},
        ]
print(chat_bot.generate_response(user_input))
