# my own code's here
#from tools.train import GPTQTrainer, DEFAULT_CONFIG, TrainingScheduler
#import threading
#from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG
#from tools.avatar import AvatarController, AvatarConfig
#from tools.sql_memory import SQLMemory
#from tools.langchain_memory import LanguageChain

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

from tqdm import tqdm
from colorama import Fore

from dotenv import load_dotenv
import gradio as gr

from optimum.onnxruntime import ORTModelForSequenceClassification

Инициализация компонентов
trainer = GPTQTrainer(DEFAULT_CONFIG)  # Инициализация тренера для обучения GPTQ
scheduler = TrainingScheduler(trainer)  # Планировщик для управления обучением
audio = AudioManager(DEFAULT_VOICE_CONFIG)  # Менеджер аудио для воспроизведения речи

mindnlp.
'''

# Basic import
import logging
import re
import warnings
from typing import List
import asyncio

# External libraries
import torch

# Check that little boy
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

from threading import Thread
from functools import (
    lru_cache
)


'''
Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent 
endless repetitions or incoherent outputs.
Avoid adding a system prompt; all instructions should be contained within the user prompt.
To ensure that the model engages in thorough reasoning, we recommend enforcing the model 
to initiate its response with "<think>\n" at the beginning of every output.
'''

# Параметры
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B-uncensored"
# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)

# Параметры модели
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-1.5B-uncensored"

MAX_HISTORY_LENGTH = 5  # Ограничиваем историю диалога

GENERATION_CONFIG = {
    "bos_token_id": 1,
    "eos_token_id": 111,
    "pad_token_id": 1111,
    "temperature": 0.6, 
    "do_sample": True, 
    "early_stopping": True, 
    "num_return_sequences": 1, 
    "max_new_tokens": 512,
    "use_cache": True,  # Модель может использовать кэш для ускорения генерации
    "repetition_penalty": 1.7,
    "top_p": 0.95
}

SYSTEM_PROMPT = (
    "Every response should start with '<think>' to ensure the model engages in thorough reasoning."
    "Please provide a thorough and well-reasoned response."
)


class StopOnEOS(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.eos_token_id  # Останавливаем генерацию при `eos_token_id`


class ChatBot:
    def __init__(self):
        """
        Инициализация чат-бота.
        :param model_name: Название или путь к модели.
        """
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.system_prompt = SYSTEM_PROMPT
        self.streamer = None
        
        self.chat_history = []  # Храним историю сообщений
        self.chat_history.append([{'role': 'system', 'content': self.system_prompt}])
        
        self.initialize_tokenizer()
        self.initialize_model()
        self.initialize_streamer()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)  # Перемещаем модель на GPU, если доступно
        
    def initialize_tokenizer(self):
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
            )
            
        # Убедитесь, что pad_token установлен
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Безопасный вариант
            self.tokenizer.padding_side = "left"  # Обрезаем слева для совместимости с transformers

    def initialize_model(self):
        """
        Инициализация модели с использованием transformers.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        #self.model.to(self.device) - doesn't work
        
    def initialize_streamer(self):
        self.streamer = TextIteratorStreamer(self.tokenizer, 
                                        skip_prompt=True, 
                                        skip_special_tokens=True)
    
    def build_input_from_chat_history(self, msg: str):
        """Формируем историю сообщений в виде строки для модели."""
        self.chat_history.append({'role': 'user', 'content': msg})
        return f"\n{self.chat_history}"

    async def predict(self, user_input):
        """Асинхронная генерация ответа с streamer."""
        prompt = self.build_input_from_chat_history(user_input)

        # Токенизируем ввод (ФИКС ошибки attention_mask)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Перенос на cuda

        # Запускаем генерацию в отдельном потоке
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self.model.generate, kwargs={"streamer": streamer, **inputs})
        thread.start()

        response = ""
        async for new_token in self.stream_response(streamer, response):
            print(new_token[len(response):], end="", flush=True)
            response = new_token

        return response
        
        '''
        Version 1.0.0
        messages = self.build_input_from_chat_history(history, message)
        
        # Токенизируем входные данные
        input_ids = self.tokenizer(
            "\n".join([msg['content'] for msg in messages]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).input_ids.to(self.device)  # Перемещаем тензоры на GPU

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self.model.generate, args=(input_ids,), kwargs={"streamer": streamer})
        thread.start()

        response = ""
        async for new_token in self.stream_response(streamer, response):
            print(new_token[len(response):], end="", flush=True)
            response = new_token
        return response
        '''
    
    async def stream_response(self, streamer, response):
        """Асинхронный поток вывода ответа в реальном времени."""
        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message
            await asyncio.sleep(0.01)  # Даем время для асинхронного обновления UI

    async def chat_loop(self):
        """Асинхронный чат-бот."""
        print("Добро пожаловать! Вы можете начать общение с ботом. Для выхода введите 'exit' или 'quit'.")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "Вы: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Диалог завершен.")
                    break

                print("Бот: ", end="")
                response = await self.predict(user_input)
                print()

            except KeyboardInterrupt:
                print("\nДиалог прерван пользователем.")
                break
            except Exception as e:
                print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    # Инициализация чат-бота
    chat_bot = ChatBot()
    
    # Запуск основного цикла диалога
    asyncio.run(chat_bot.chat_loop())
    

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