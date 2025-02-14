# my own code's here
#from tools.train import GPTQTrainer, DEFAULT_CONFIG, TrainingScheduler
#import threading
#from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG
#from tools.avatar import AvatarController, AvatarConfig
#from tools.sql_memory import SQLMemory
#from tools.langchain_memory import LanguageChain

# Basic import
import logging

'''
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (
    AutoTokenizer, 
    pipeline
)
from functools import (
    lru_cache
)
from langchain.cache import InMemoryCache

import asyncio
from tqdm import tqdm
from colorama import Fore

from dotenv import load_dotenv
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            ms_dtype=mindspore.float16
            )
        
        self.generation_config = self.model.generation_config
        print("Действующие параметры модели", self.generation_config)
        
        '''
        self.generation_config.bos_token_id = GENERATION_CONFIG['bos_token_id']
        self.generation_config.eos_token_id = GENERATION_CONFIG['eos_token_id']
        self.generation_config.pad_token_id = GENERATION_CONFIG['pad_token_id']
        self.generation_config.temperature = GENERATION_CONFIG['temperature']
        self.generation_config.num_return_sequences = GENERATION_CONFIG['num_return_sequences']
        self.generation_config.max_new_tokens = GENERATION_CONFIG['max_new_tokens']
        self.generation_config.repetition_penalty = GENERATION_CONFIG['repetition_penalty']
        
        print("Действующие параметры модели", self.generation_config)
        '''

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
    def predict(self, message, history):
        history_tuple = tuple(tuple(x) for x in history)  # Конвертируем список истории в кортеж
        history_transformer_format = history + [[message, ""]]

        # Formatting the input for the model.
        messages = self.build_input_from_chat_history(history, message)
        
        # Предварительная обработка запроса (можно закомментировать)
        messages = lambda message: self.preprocess_prompt(message)
        
        input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="ms",
                tokenize=True
            )
        
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            timeout=300, 
            skip_prompt=True, 
            skip_special_tokens=True
            )
        
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
            num_beams=1,
        )
        
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()  # Starting the generation in a separate thread.
        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
                break
            yield partial_message
            
    @lru_cache(maxsize=1000)  # Кэшируем результаты для повторяющихся запросов
    def start_chat_loop(self):
        """
        Основной цикл диалога для взаимодействия с пользователем через терминал.
        """
        print("Добро пожаловать! Вы можете начать общение с ботом. Для выхода введите 'exit' или 'quit'.")
        
        # История диалога
        chat_history = []
        
        while True:
            try:
                # Получаем ввод от пользователя
                user_input = input("Вы: ").strip()
                
                # Проверяем условие выхода
                if user_input.lower() in ["exit", "quit"]:
                    print("Диалог завершен.")
                    break
                
                # Генерируем ответ от бота
                print("Бот: ", end="")
                response = ""
                for partial_response in self.predict(user_input, chat_history):
                    print(partial_response[len(response):], end="", flush=True)
                    response = partial_response
                
                print()  # Переход на новую строку после завершения ответа
                
                # Обновляем историю диалога
                chat_history.append((user_input, response))
            
            except KeyboardInterrupt:
                print("\nДиалог прерван пользователем.")
                break
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                continue
            

if __name__ == "__main__":
    # Инициализация чат-бота
    chat_bot = ChatBot()
    
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