# my own code's here
#from tools.train import GPTQTrainer, DEFAULT_CONFIG, TrainingScheduler
#import threading
#from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG
#from tools.avatar import AvatarController, AvatarConfig
#from tools.sql_memory import SQLMemory
#from tools.langchain_memory import LanguageChain

# Basic import
import logging

# External libraries
import torch
from transformers import (
    AutoTokenizer, 
    pipeline
)
from functools import (
    lru_cache
)
#from langchain.cache import InMemoryCache

import asyncio
from tqdm import tqdm
from colorama import Fore

#from dotenv import load_dotenv
#load_dotenv()

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
'''

# Basic import
import logging
import re
import warnings
from typing import List

# External libraries
import torch

import mindspore

# Check that little boy
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer

from threading import Thread

#from optimum.onnxruntime import ORTModelForSequenceClassification

from functools import (
    lru_cache
)

"""
Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs.
Avoid adding a system prompt; all instructions should be contained within the user prompt.
To ensure that the model engages in thorough reasoning, we recommend enforcing the model to initiate its response with "<think>\n" at the beginning of every output.
"""

# Инициализация компонентов
#trainer = GPTQTrainer(DEFAULT_CONFIG)  # Инициализация тренера для обучения GPTQ
#scheduler = TrainingScheduler(trainer)  # Планировщик для управления обучением
#audio = AudioManager(DEFAULT_VOICE_CONFIG)  # Менеджер аудио для воспроизведения речи

# Параметры
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B-uncensored"

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)

# Параметры модели
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B-uncensored"

GENERATION_CONFIG = {
    "bos_token_id": 151646,
    "eos_token_id": 151646,
    "pad_token_id": 11,
    "temperature": 0.6, 
    "do_sample": True, 
    "early_stopping": True, 
    "num_return_sequences": 1, 
    "max_new_tokens": 256,
    "use_cache": True,  # Модель может использовать кэш для ускорения генерации
    "repetition_penalty": 1.7
}

system_prompt = "Please provide a thorough and well-reasoned response."


class ChatBot:
    def __init__(self):
        """
        Инициализация чат-бота.
        :param model_name: Название или путь к модели.
        """
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        
        self.initialize_tokenizer()
        self.initialize_model()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize_tokenizer(self):
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            ms_dtype=mindspore.float16
            )
        
    def initialize_model(self):
        """
        Инициализация модели с использованием transformers.
        """
        try:
            # Создаем пайплайн для генерации текста
            self.generator = pipeline("text-generation", model=self.model_name)
            logging.info("Модель успешно загружена с помощью transformers.")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {str(e)}")
            self.generator = None

    def preprocess_prompt(self, prompt):
        """
        Предварительная обработка входного запроса для улучшения качества ответа.
        :param prompt: Входной запрос пользователя.
        :return: Обработанный запрос.
        """
        # Добавляем инструкции для модели
        processed_prompt = (
            f"<think>\n{prompt}\n</think>",
            system_prompt
        )
        return processed_prompt
    
    def build_input_from_chat_history(self, chat_history, msg: str):
        messages = [{'role': 'system', 'content': system_prompt}]
        for user_msg, ai_msg in chat_history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
        messages.append({'role': 'user', 'content': msg})
        return messages
    
    # Function to generate model predictions.
    @lru_cache(maxsize=1000)  # Кэшируем результаты для повторяющихся запросов
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.6, top_p=0.95):
        """
        Генерация ответа с использованием transformers.
        :param prompt: Входной запрос пользователя.
        :param max_length: Максимальная длина ответа.
        :param temperature: Температура для генерации.
        :param top_p: Параметр nucleus sampling (top-p).
        :return: Сгенерированный текст.
        """
        try:
            if not self.generator:
                raise ValueError("Модель не загружена!")
            
            # Генерация ответа
            response = self.generator(
                prompt,
                max_new_tokens = max_new_tokens,
                truncation=True,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            return response[0]["generated_text"].strip()
        except Exception as e:
            logging.error(f"Ошибка при генерации ответа: {str(e)}")
            return "Извините, произошла ошибка."

    def start_chat_loop(self):
        """
        Основной цикл диалога.
        """
        print("Диалог начат...")
        while True:
            try:
                user_input = input(Fore.WHITE + "USER: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Диалог завершен.")
                    break

                # Предварительная обработка запроса (можно закомментировать)
                processed_input = self.preprocess_prompt(user_input)

                # Генерация ответа
                response = self.generate_response(user_input)
                print(Fore.LIGHTGREEN_EX + f"BOT: {response}")
            except KeyboardInterrupt:
                print("\nДиалог прерван пользователем.")
                break
            except Exception as e:
                logging.error(f"Ошибка в диалоге: {str(e)}")
                print("Произошла ошибка. Попробуйте снова.")

if __name__ == "__main__":
    # Инициализация чат-бота
    chat_bot = ChatBot(MODEL_NAME)

    # Запуск основного цикла диалога
    chat_bot.start_chat_loop()
    

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