# train.py
import os
import json
import torch
import schedule
import time
from typing import List, Dict
from datetime import datetime
import pynvml

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

class LoRATrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        """Инициализация модели с квантизацией"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=True
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        self._apply_lora()

    def _apply_lora(self):
        """Применяет LoRA адаптеры"""
        peft_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)
        self._freeze_layers()

    def _freeze_layers(self):
        """Заморозка необучаемых слоев"""
        for param in self.model.parameters():
            param.requires_grad = False
            
        for param in self.model.model.layers[-self.config["num_trainable_layers"] :].parameters():
            param.requires_grad = True

    def _load_errors(self) -> List[Dict]:
        """Загрузка и валидация ошибок"""
        try:
            with open(self.config["errors_path"], "r", encoding="utf-8") as f:
                errors = json.load(f)
                return [e for e in errors if self._validate_sample(e)]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _validate_sample(self, sample: Dict) -> bool:
        """Проверка корректности примера"""
        required_keys = {"input", "output", "corrected"}
        return all(key in sample for key in required_keys)

    def _prepare_dataset(self):
        """Подготовка данных для обучения"""
        errors = self._load_errors()
        return [self._format_example(e) for e in errors]

    def _format_example(self, example: Dict) -> str:
        """Форматирование примера в текст"""
        return (
            f"### Input: {example['input']}\n"
            f"### Incorrect Output: {example['output']}\n"
            f"### Corrected Output: {example['corrected']}"
        )

    def train(self):
        """Основной метод обучения"""
        if self._check_vram() > self.config["max_vram"]:
            raise MemoryError("Not enough VRAM for training")

        dataset = self._prepare_dataset()
        if len(dataset) < self.config["min_samples"]:
            return False

        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation"],
            learning_rate=self.config["learning_rate"],
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        trainer.train()
        self._save_adapters()
        return True

    def _save_adapters(self):
        """Сохранение только адаптеров"""
        output_dir = os.path.join(
            self.config["output_dir"],
            f"adapters_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        self.model.save_pretrained(output_dir)

    def _check_vram(self) -> int:
        """Проверка используемой VRAM"""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used // 1024**2  # MB

class TrainingScheduler:
    def __init__(self, trainer: LoRATrainer):
        self.trainer = trainer
        self._setup_schedule()

    def _setup_schedule(self):
        """Настройка расписания"""
        schedule.every().day.at("04:00").do(self._daily_training)
        schedule.every(3).hours.do(self._check_for_errors)

    def _daily_training(self):
        if self.trainer.train():
            self._clean_old_adapters()

    def _check_for_errors(self):
        if len(self.trainer._load_errors()) >= 5:
            self.trainer.train()

    def _clean_old_adapters(self):
        """Удаление старых адаптеров"""
        adapters = sorted(os.listdir(self.trainer.config["output_dir"]))
        for adapter in adapters[:-3]:
            os.remove(os.path.join(self.trainer.config["output_dir"], adapter))

    def run_background(self):
        """Запуск в фоновом режиме"""
        while True:
            schedule.run_pending()
            time.sleep(60)

# Пример конфигурации
DEFAULT_CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "errors_path": "memory/memory_errors.json",
    "output_dir": "./adapters",
    "lora": {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05
    },
    "num_trainable_layers": 2,
    "num_epochs": 1,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 3e-5,
    "max_vram": 10000,  # 10GB
    "min_samples": 5
}