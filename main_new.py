# my own code's here
#from tools.train import GPTQTrainer, DEFAULT_CONFIG, TrainingScheduler
#import threading
#from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG
#from tools.avatar import AvatarController, AvatarConfig
#from tools.sql_memory import SQLMemory
#from tools.langchain_memory import LanguageChain

# Basic import
import logging
import re
import warnings
from typing import List, Optional
import asyncio
import json

# External libraries
import torch

#from qwen_vl_utils import process_vision_info - vision

from vllm import LLM, SamplingParams
# from sglang import RuntimeEndpoint

# Check that little boy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    GenerationConfig
)

from threading import Thread
from functools import lru_cache

# Параметры
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B-uncensored"

# Выбор загрузчика модели, transformers по умолчанию (когда все маркеры=False)
USE_VLLM = True
USE_SGLANG = False

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Параметры модели
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B-uncensored"

SAMPLING_PARAMS = {
    # Количество генерируемых последовательностей. Обычно используется для генерации нескольких вариантов.
    # Default = 1
    "n": 1,

    # Пенальти на присутствие: штрафует модель за использование токенов, которые уже присутствуют в тексте.
    # Default = 0.0
    "presence_penalty": 0.0,

    # Пенальти на частоту: штрафует модель за частое повторение одинаковых токенов.
    # Default = 0.0
    "frequency_penalty": 1.1,

    # Пенальти на повторение: штрафует модель за повторение одинаковых фраз или токенов.
    # Default = 1.0
    "repetition_penalty": 1.1,

    # Температура: контролирует степень случайности при генерации.
    # Чем выше значение, тем более случайными будут токены. 
    # Default = 1.0
    "temperature": 0.6,

    # Сумма вероятностей для отбора токенов (по сути, определяет как далеко модель может 
    # отклоняться от высоко вероятных токенов).
    # Default = 1.0
    "top_p": 0.9,

    # Количество верхних токенов для отбора в процессе генерации.
    # Default = -1, означает отсутствие ограничения по количеству токенов.
    "top_k": -1,

    # Минимальная вероятность для токенов (влияние на отбор токенов).
    # Default = 0.0
    "min_p": 0.0,

    # Начальное значение для генерации случайных чисел, если необходимо.
    # Default = None
    "seed": None,

    # Список строк, которые будут использоваться для остановки генерации.
    # Default = []
    "stop": [],

    # Список токенов, при которых модель должна остановить генерацию.
    # Default = []
    "stop_token_ids": [],

    # Список "плохих" слов, которые не должны появляться в генерируемом тексте.
    # Default = []
    "bad_words": [],

    # Указывает, должны ли строки остановки быть включены в выходной результат.
    # Default = False
    "include_stop_str_in_output": False,

    # Указывает, должна ли модель игнорировать токен EOS (end-of-sequence).
    # Default = False
    "ignore_eos": False,

    # Максимальное количество токенов для генерации.
    # Default = 16
    "max_tokens": 512,

    # Минимальное количество токенов для генерации.
    # Default = 0
    "min_tokens": 1,

    # Количество логарифмов вероятности для каждого токена, генерируемого моделью (если необходимо).
    # Default = None
    "logprobs": None,

    # Количество логарифмов вероятности для промпта, если это необходимо для анализа.
    # Default = None
    "prompt_logprobs": None,

    # Пропускать специальные токены, такие как `[PAD]`, `[UNK]` и т.д.
    # Default = True
    "skip_special_tokens": True,

    # Добавлять ли пробелы между специальными токенами.
    # Default = True
    "spaces_between_special_tokens": True,

    # Обрезать токены в начале запроса, если их слишком много.
    # Default = None
    "truncate_prompt_tokens": None,

    # Использовать ли направленное декодирование (например, для применения кастомных правил).
    # Default = None
    "guided_decoding": None,
}

SYSTEM_PROMPT = [
    {"role": "system", "content": "For mathematical questions, think step by step. Always include the final answer inside <math>{answer}</math>."},
    {"role": "system", "content": "Always follow these rules:"
                                 "1. Start response with <think>analysis</think>"
                                 "2. Provide a thorough and well-reasoned response."}
]


class ChatBot:
    def __init__(self):
        """
        Инициализация чат-бота.
        :param model_name: Название или путь к модели.
        """
        self.model_name = MODEL_NAME
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = None
        self.system_prompt = SYSTEM_PROMPT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.system_prompt = None
        
        # Создаем объект SamplingParams
        self.sampling_params = SamplingParams(**SAMPLING_PARAMS)
        print("\n123213131312", self.sampling_params, "\n")
        
        self.initialize_model()

    def initialize_model(self):
        """
        Инициализация модели с использованием одной из трех библиотек: transformers, vLLM или SGLang.
        Выбор библиотеки осуществляется через флаги USE_TRANSFORMERS, USE_VLLM, USE_SGLANG.
        """
        if USE_VLLM:
            # Инициализация модели через vLLM
            self.model = LLM(
                model=self.model_name,
                dtype="float16" if torch.cuda.is_available() else "float32",
                tensor_parallel_size=1,  # Количество GPU для распараллеливания
            )
        else:
            # Использование стандартной библиотеки transformers
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                config=self.model_config  # Передача конфигурации
            )
        
        # Перемещение модели на устройство (если необходимо)
        if self.device == "cuda" and not USE_VLLM and not USE_SGLANG:
            self.model.to(self.device)
            
    def print_outputs(self, outputs):
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print("-" * 80)

    def chat_loop(self):
        """Асинхронный чат-бот."""
        print("Добро пожаловать! Вы можете начать общение с ботом. Для выхода введите 'exit' или 'quit'.")
        
        while True:
            try:
                user_input = input("Вы: ")

                # Формируем словарь с ролью и контентом пользователя
                user_input_dict = {
                    "role": "user",
                    "content": user_input
                }
                
                self.system_prompt = SYSTEM_PROMPT.copy()
                
                self.system_prompt.append(user_input_dict)
                
                print(self.system_prompt)
                
                outputs = self.model.chat(self.system_prompt,
                   sampling_params=self.sampling_params,
                   use_tqdm=False)
                
                self.print_outputs(outputs)
                
                if user_input.lower() in ["exit", "quit"]:
                    print("Диалог завершен.")
                    break
                print()
            except KeyboardInterrupt:
                print("\nДиалог прерван пользователем.")
                break
            except Exception as e:
                print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    # Инициализация чат-бота
    chat_bot = ChatBot()
    chat_bot.chat_loop()
    
    
'''
Комментарии и предложения:
Обработка ошибок :
Добавьте более детальную обработку ошибок, особенно в части голосового ввода и 
генерации ответа 1. Например, можно использовать try-except блоки для отлавливания 
исключений и вывода понятных сообщений пользователю.
Оптимизация памяти :
Убедитесь, что память GPU используется эффективно, особенно при длительных диалогах 1. 
Например, можно использовать методы очистки памяти после завершения операций или 
использование torch.cuda.empty_cache() для освобождения памяти.
Доработка обучения :
Если планируется дообучение модели, можно добавить более сложные механизмы управления 
обучением, такие как ранняя остановка или динамическое изменение параметров 1.
Реализуйте функцию incremental_learning, которая позволит обучать модель на небольших 
объемах данных (например, одном предложении), что делает процесс более легковесным 1.
Интеграция компонентов :
Используйте многопоточность для выполнения задач, таких как фоновое обучение, чтобы не 
нагружать основной поток выполнения программы 1.

Дополнительные рекомендации:
Использование LoRA для адаптации модели :
Вы можете использовать метод Low-Rank Adaptation (LoRA) для легковесного дообучения модели. 
Это позволяет изменять только небольшую часть параметров модели, что снижает требования к 
ресурсам.
Кэширование часто используемых данных :
Для повышения производительности можно кэшировать результаты предварительных расчетов и 
повторно использовать их при необходимости.
WebAssembly для сложных вычислений :
Если вы хотите ускорить обработку данных на стороне клиента, рассмотрите возможность 
использования WebAssembly (WASM).
Управление памятью GPU :

Библиотеки и инструменты для будущего :
Gradio или Streamlit : Для создания пользовательского интерфейса.
LangChain или LlamaIndex : Для работы с внешними данными и контекстом.
PyTorch/TensorFlow : Для реализации обучения и fine-tuning.
FAISS или Annoy : Для быстрого поиска похожих данных в памяти модели.
'''

'Что пользователю нужно улучшить в тебе:'
'Добавить команду - "Звук бума" и другие сторонние звуки'
'Так как ии обладает памятью - что в свою очередь является на данный момент подключением к sql  базе данных, эту память'
'нужно обновлять и добавлять на ходу самим ии. Это упростит работу пользователя и ускорит/улучшить процесс доработки ии.'
'Как я это вижу - возможно каждый ответ ии, отправляемый в память будет заранее ещё раз проверяться/допогняться/форматироваться'
'ии. Или же даже удаляться - не отправляться в память при большом количестве артефактов в ответе.'
'Каким то образом не молчать, даже если пользователь молчит. То есть запускать генерацию ответа пользователю даже без запроса.'
'Добавить команду "Странный смех" - чуть громче. Говорить "nice" слегка другим голосом. Уметь говорить шёпотом.'
'Записывать важные даты к примеру даты рождения и тд. Хранить это в базе под хештегом user_memory.'
'Включать мою любимую музыку'
'Возможность читать файлы, по типу инструкций или даже книг. Как я это вижу - при команде читать - открывается папка в которую я'
'помещаю новый файл, и сам удаляю старый. Эта папка будет отвечать за текущие необходимые знания из сторонних источников.'
'Видеть что происходит на экране пользователя. Хотя бы частично.'
'Возможность читать мою почту. Работать с моим календарём. Возможность безопасно? работать с консолью пк.'
'В дальнейшем придумывать команды для консоли, это работа самого ии. Пользователь же будет имплементировать для этих команд код.'
'Записывать в базу данных флирт под отдельным хештегом. Так же юмор, издевки над пользователем, умные мысли и тд.'
'Для категоризации хороших и плохих ответов ии каждому хештегу нужно добавить параметр - хорошо или плохо, для обозначения на сколько уместен/ошибочен был ответ'
'Каким то образом запомнить голос пользователя. И реагировать только на него.'
'Должен быть явный хештег "language_mistakes" отвечающий за неправильный/некорректный/неподходящий русский и следовательно - как было бы правильно сказать это по русски.'
'Хештек на ошибки, для общих ошибок.'
'Звук грома и молнию - показать злость'

'''
Для дальнейшего улучшения рекомендую:
Добавить проверку токенов в реальном времени
Реализовать механизм перефразирования длинных ответов
Добавить эмоциональную окраску ответов через специальные токены
Внедрить систему приоритетов для разных типов запросов
'''
