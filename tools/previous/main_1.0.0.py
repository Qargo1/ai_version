from tools.train import LoRATrainer, DEFAULT_CONFIG
import threading

from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG

from PyQt5.QtWidgets import QMainWindow
from tools.avatar import AvatarController, AvatarConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftConfig, PeftModel
import psutil
import torch
import json
import os
import speech_recognition as sr

# Инициализация компонентов
trainer = LoRATrainer(DEFAULT_CONFIG)
scheduler = TrainingScheduler(trainer)
audio = AudioManager(DEFAULT_VOICE_CONFIG)
recognizer = sr.Recognizer()

MODEL_NAME = "models/ruGPT-3.5-13B"
MEMORY_FILE = "memory/memory_1.0.1.json"
ERRORS_FILE = "memory/memory_errors.json"
MAX_HISTORY = 5

os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

system_prompt = """<|system|>
Ты Виктория — AI-подруга пользователя. Твои черты:
1. Общаешься на "ты" по-русски, но уважительно
2. Поддерживаешь диалог вопросами
3. Делаешь ответы короткими (1-2 предложения)
4. Используешь эмодзи 😊 там, где уместно
</s>
"""

class ChatBot:
    def __init__(self):
        self.memory = self.load_memory(MEMORY_FILE)
        self.error_memory = self.load_memory(ERRORS_FILE)
        self.model, self.tokenizer = self.initialize_model()
        self.audio = audio

    @staticmethod
    def initialize_model():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16
        )
        return model, tokenizer

    @staticmethod
    def load_memory(file_path):
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading memory: {str(e)}")
            return []

    def save_memory(self, data, file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving memory: {str(e)}")

    def format_context(self, messages):
        context = system_prompt
        for msg in messages[-MAX_HISTORY:]:
            context += f"\n<|{msg['role']}|>{msg['content']}</s>"
        return context + "\n<|assistant|>"

    def generate_response(self, context):
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to("cuda")
        output = self.model.generate(
            input_ids,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def process_voice_input(self):
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=5)
                return recognizer.recognize_whisper(audio, language="russian")
            except sr.WaitTimeoutError:
                return ""
            except Exception as e:
                print(f"Audio error: {str(e)}")
                return ""

    def handle_command(self, command):
        if command == "стоп":
            self.save_memory(self.memory, MEMORY_FILE)
            return True
        elif command == "исправь":
            self.error_memory.extend(self.memory[-2:])
            self.save_memory(self.error_memory, ERRORS_FILE)
        return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.avatar_config = AvatarConfig(model_url="path/to/model.glb")
        self.avatar = AvatarController(self.avatar_config)
        self.setCentralWidget(self.avatar.container)
        self.bot = ChatBot()

    def start_chat_loop(self):
        print("Диалог начат...")
        try:
            while True:
                user_input = self.bot.process_voice_input()
                if not user_input:
                    continue

                if self.bot.handle_command(user_input.lower()):
                    break

                self.bot.memory.append({"role": "user", "content": user_input})
                context = self.bot.format_context(self.bot.memory)
                response = self.bot.generate_response(context)

                self.bot.memory.append({"role": "assistant", "content": response})
                self.avatar.animate_speech()
                self.bot.audio.speak(response)
                print(f"\nБот: {response}\n")

                if len(self.bot.error_memory) >= 2:
                    threading.Thread(
                        target=scheduler.run_background,
                        daemon=True
                    ).start()
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
        finally:
            self.bot.save_memory(self.bot.memory, MEMORY_FILE)
            self.bot.save_memory(self.bot.error_memory, ERRORS_FILE)
            trainer.train()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.start_chat_loop()
    sys.exit(app.exec_())
