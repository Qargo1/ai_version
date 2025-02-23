import os
import json
import torch
import schedule
import time
from typing import List, Dict
from datetime import datetime
import pynvml
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from safetensors.torch import load_file
from peft import PeftConfig, PeftModel


class GPTQTrainer:
    # LoRA дообучение (пример)
    def fine_tune_with_lora(self, training_data):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        # Здесь нужно добавить обучение на training_data (упрощённый пример)
        model.save_pretrained("/path/to/lora_model")
    
'''
Комментарии и предложения:
Инкрементное обучение :
Добавлена функция incremental_learning, которая позволяет обучать модель на небольших 
объемах данных (например, одном предложении), что делает процесс более легковесным 1.
Уменьшено количество эпох и размер батча для инкрементного обучения, чтобы снизить 
нагрузку на систему.
Оптимизация использования VRAM :
Добавлено проверка доступной VRAM перед началом обучения. Если доступно недостаточно 
памяти, выбрасывается исключение 1.
Распределенное обучение :
Рассмотрите использование DistributedDataParallel для распределенного обучения на 
нескольких GPU, если у вас есть такая возможность 1.
Автоматическое сохранение модели :
Модель автоматически сохраняется после каждого цикла обучения, что позволяет 
восстановить обучение в случае сбоя 7.
Чистка старых моделей :
В TrainingScheduler добавлена функция clean_old_models, которая удаляет старые 
версии модели, оставляя только последние три, что помогает экономить место на диске 1.
Постоянное улучшение модели :
Функция incremental_learning может быть вызвана при каждом новом взаимодействии с 
пользователем, что позволит модели постепенно перенимать новые знания 1.
Легковесность :
Для снижения нагрузки на систему можно использовать меньшие значения параметров, 
таких как batch_size и num_train_epochs, а также уменьшить шаг обучения (learning_rate) 
для инкрементного обучения 1.
'''

'''
Основные предложения:
Инкрементное обучение :
Для инкрементного обучения можно использовать не только Trainer из transformers, 
но и более легковесные методы, такие как torch.no_grad() для обновления весов модели 
без необходимости полного цикла обучения.
Можно использовать технику Experience Replay , где новые примеры сохраняются в буфере, 
а модель обучается на случайной выборке из этого буфера, чтобы избежать "забывания" 
старых знаний.
Оптимизация VRAM :
Использование gradient_checkpointing может помочь снизить использование памяти.
Можно использовать torch.cuda.empty_cache() для очистки кэша GPU после каждого шага обучения.
Модель :
Qwen2.5-0.5B-Instruct-GPTQ-Int8 — это хорошая модель для начала, особенно если 
вы хотите минимизировать использование ресурсов. Однако, если вы хотите экспериментировать 
с другими моделями, рассмотрите варианты, такие как Llama 2 7B (4-bit quantized) или 
Mistral 7B (4-bit quantized) . Эти модели также достаточно легковесны и могут быть 
адаптированы для домашнего использования.
Автоматическое сохранение и чистка моделей :
Ваш подход к автоматическому сохранению и чистке старых моделей уже хорош, но можно 
добавить возможность сохранения только изменений (например, через дифференциальное 
сохранение).
Параллельная обработка :
Если у вас есть доступ к нескольким GPU, можно использовать DistributedDataParallel 
для распределенного обучения.
Легковесность :
Уменьшение batch_size до 1 и использование меньшего количества эпох уже реализовано, 
что отлично подходит для легковесного решения.
'''

'''
1. Использование LoRA (Low-Rank Adaptation)
LoRA — это метод тонкой настройки моделей, который позволяет обучать только небольшую 
часть параметров модели, сохраняя остальные замороженными. Это значительно снижает 
вычислительные затраты и объем памяти, необходимый для обучения.

Преимущества:
Меньше ресурсов : Вы обучаете только маленькую матрицу весов, а не всю модель.
Быстрее : Процесс обучения становится значительно быстрее.
Легче сохранять изменения : Вы можете сохранять только обновленные веса, что экономит 
место на диске.
Библиотеки:
PEFT : Библиотека от Hugging Face, которая поддерживает LoRA и другие методы тонкой 
настройки.
Lora : Оригинальная реализация LoRA.
from peft import LoraConfig, get_peft_model

# Настройка LoRA
lora_config = LoraConfig(
    r=8,  # Ранг матрицы (чем меньше, тем легче)
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Модули, которые будут обучаться
    lora_dropout=0.1,
    bias="none"
)

# Применение LoRA к модели
model = get_peft_model(model, lora_config)
'''


'''
2. Использование DPO (Direct Preference Optimization)
DPO — это метод обучения, который напрямую оптимизирует предпочтения пользователя, 
минуя этап создания наград (как в RLHF). Этот метод может быть полезен, если вы 
хотите, чтобы модель училась на ваших предпочтениях в реальном времени.

Преимущества:
Простота : Нет необходимости в сложной инфраструктуре для обучения через RLHF.
Эффективность : Модель учится напрямую на ваших предпочтениях.
Библиотеки:
TRL : Библиотека от Hugging Face, которая поддерживает DPO и другие методы.
3. Использование PPO (Proximal Policy Optimization)
PPO — это метод обучения с подкреплением, который можно использовать для тонкой 
настройки модели на основе обратной связи пользователя. Хотя этот метод требует больше ресурсов, чем LoRA, он может быть полезен, если вы хотите, чтобы модель учились на более сложных задачах.

Библиотеки:
TRL : Поддерживает PPO и другие методы RLHF.
'''

'''
4. Оптимизация градиентов
Если вы хотите еще больше снизить нагрузку на систему, можно использовать следующие 
техники:

a) Gradient Accumulation
Вы уже используете gradient_accumulation_steps, но можно экспериментировать с большими 
значениями, чтобы уменьшить размер батча до 1, не теряя при этом стабильности обучения.

b) Gradient Clipping
Ограничение градиентов может помочь избежать взрыва градиентов и сделать обучение более 
стабильным:
training_args = TrainingArguments(
    ...
    gradient_clipping=1.0,  # Ограничение градиентов
    ...
)
c) Mixed Precision Training
Вы уже используете fp16=True, но можно также попробовать bf16=True (Brain Floating Point), 
который менее точен, 
но быстрее и требует меньше памяти.
'''

'''
5. Собственноручно написанный код для легковесного обучения
Если вы хотите полностью контролировать процесс обучения, можно написать собственный цикл 
обучения. Это позволит вам точно настроить каждый шаг и минимизировать использование ресурсов.

Пример легковесного цикла обучения:
import torch
import torch.nn as nn
import torch.optim as optim

def lightweight_training(model, tokenizer, input_text, corrected_text, lr=1e-5):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Токенизация входных данных
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    labels = tokenizer(corrected_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Прямой проход
    outputs = model(**inputs, labels=labels["input_ids"])
    loss = outputs.loss
    
    # Обратный проход
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Очистка кэша GPU
    torch.cuda.empty_cache()
    
    return loss.item()

# Пример использования
user_input = "Пример входного текста."
corrected_output = "Правильный выходной текст."
loss = lightweight_training(trainer.model, trainer.tokenizer, user_input, corrected_output)
print(f"Loss: {loss}")
'''


'''
6. Использование ONNX Runtime для ускорения вывода
Если вы хотите ускорить вывод модели (inference), можно использовать ONNX Runtime . Это позволит вам выполнять вывод модели быстрее, особенно на CPU.

Преимущества:
Ускорение вывода : ONNX Runtime оптимизирует выполнение модели.
Кроссплатформенность : Может работать как на CPU, так и на GPU.
Библиотеки:
ONNX Runtime : Официальная библиотека.
Пример конвертации модели в ONNX:
from transformers import AutoTokenizer, AutoModelForCausalLM
from onnxruntime import InferenceSession

# Конвертация модели в ONNX
tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B-Instruct-GPTQ-Int8")
model = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B-Instruct-GPTQ-Int8")

# Сохранение модели в ONNX формат
torch.onnx.export(
    model,
    tokenizer("Пример текста", return_tensors="pt").input_ids,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
    opset_version=13
)

# Загрузка ONNX модели
session = InferenceSession("model.onnx")
'''

'''
7. Альтернативные модели
Если вы хотите экспериментировать с другими моделями, вот несколько вариантов:

Llama 2 7B (4-bit quantized) : Легковесная версия популярной модели.
Mistral 7B (4-bit quantized) : Еще одна мощная модель с хорошей производительностью.
Phi-2 : Очень компактная модель, которая хорошо работает на CPU.
'''

