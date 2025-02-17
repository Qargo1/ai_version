from long_term import LongTermMemory

# Basic imports
import time
import logging
import re
import warnings
from typing import List, Optional
import asyncio
import json

# @lru_cache(maxsize=128) # Можно использовать для кеширования unhashable types
from functools import lru_cache

from collections import deque

from colorama import Fore

# External libraries
import torch

# Different_Model_loaders
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer,
    StoppingCriteria,
    pipeline
)
from gptqmodel import GPTQModel
from awq import AutoAWQForCausalLM
from vllm import LLM, SamplingParams
#from qwen_vl_utils import process_vision_info - vision

from threading import Thread


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class StopOnEOS(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.eos_token_id  # Останавливаем генерацию при `eos_token_id`


class HelperForChatBot:
    def __init__(self):
        pass
    
    def create_model_config(self):
        with open(self.model_config_path, "w", encoding="utf-8") as json_file:
            json.dump(self.model_config, json_file, indent=4, ensure_ascii=False)

    def compare_configs(self):
        """
        Сравнивает конфигурации модели до и после запуска,
        выявляет изменения, удалённые и новые параметры.
        """

        # Загружаем конфигурации
        before_config = self.basic_config
        after_config = json.loads(self.model.config.to_json_string())

        # Контейнеры для различий
        changed_params = {}
        removed_params = {}
        new_params = {}

        # Проверяем изменения и удалённые параметры
        for key, value in before_config.items():
            if key not in after_config:
                removed_params[key] = value
            elif value != after_config[key]:
                changed_params[key] = (value, after_config[key])

        # Проверяем новые параметры
        for key, value in after_config.items():
            if key not in before_config:
                new_params[key] = value

        # 📜 **Формируем отчёт**
        report = ["🔍 **Сравнение конфигураций модели (до и после запуска)**\n"]

        if changed_params:
            report.append("🔄 **Изменённые параметры:**")
            for key, (old, new) in changed_params.items():
                report.append(f"  - `{key}`: **{old} → {new}**")

        if removed_params:
            report.append("\n❌ **Удалённые параметры:**")
            for key, value in removed_params.items():
                report.append(f"  - `{key}`: **{value}**")

        if new_params:
            report.append("\n🆕 **Новые параметры:**")
            for key, value in new_params.items():
                report.append(f"  - `{key}`: **{value}**")

        # 📁 **Сохранение отчёта**
        with open("config_diff.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print("\n".join(report))
        print("\n📁 Итог сохранён в `config_diff.txt`")
        
        
    def calculate_token_length(self, text: str) -> int:
        """Вычисляет длину текста в токенах (оптимизированная версия)"""
        return self.tokenizer(text, return_length=True)["length"][0]
    
    def safe_softmax(self, logits):
        """Преобразуем логиты в нормальный softmax (избегаем inf/nan)"""
        #print("🔥 Raw logits:", logits[:10])  # Вывод первых 10 логитов
        #print("🔥 Min logit:", logits.min().item(), "Max logit:", logits.max().item())
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)  # Убираем NaN
        logits = torch.where(torch.isinf(logits), torch.full_like(logits, -1e4), logits)  # Убираем Inf default = -1e9
        #print('logits: ', logits)
        return torch.nn.functional.softmax(logits, dim=-1)
    
    def adjust_parameters_based_on_context(self, user_input: str) -> dict:
        """Динамическая настройка параметров генерации на основе контекста"""
        creative_keywords = {"imagine", "try", "joke", "creative", "story", "hypothetical", "funny"}
        factual_keywords = {"fact", "clear", "truth", "accurate", "precise", "detail", "explain"}
        
        input_lower = user_input.lower()
        params = {}
        
        # Проверка креативных ключевых слов
        if any(keyword in input_lower for keyword in creative_keywords):
            params.update({
                "temperature": min(0.9, self.generation_config["temperature"] + 0.2),
                "repetition_penalty": 1.1
            })
        
        # Проверка фактологических ключевых слов
        elif any(keyword in input_lower for keyword in factual_keywords):
            params.update({
                "temperature": max(0.3, self.generation_config["temperature"] - 0.2),
                "top_k": 20,
                "repetition_penalty": 1.5
            })
        
        return params


class ChatBot(HelperForChatBot):
    def __init__(
        self, 
        model_name=None,
        max_history_length=None, 
        model_config=None,
        model_config_path=None,
        generation_config=None,
        system_prompt=None, 
        embeddings_model=None,
        db_params=None,
        use_vllm_loader=False,
        use_gptq_loader=False,
        use_awq_loader=False,
        use_prompt_template=False
        ):
        """
        Инициализация чат-бота.
        :param model_name: Название или путь к модели.
        """
        # which loader to use? if all False => use transformer
        self.use_vllm_loader = use_vllm_loader
        self.use_gptq_loader = use_gptq_loader
        print("1234155315", self.use_gptq_loader)
        self.use_awq_loader = use_awq_loader
        
        self.use_prompt_template = use_prompt_template
        
        # Инициализация долговременной памяти
        self.system_prompt = system_prompt
        self.convo = [
            {
                'role': 'system', 
                'content': self.system_prompt
                }
            ]
        self.long_memory = LongTermMemory(
            convo=self.convo, 
            db_params=db_params
            )
        
        self.max_history_length = max_history_length
        
        self.model_name = model_name
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = None

        self.streamer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # self.create_model_config()
        
        self.basic_config = None
        self.model_config = model_config or AutoConfig.from_pretrained(self.model_name)
        self.model_config_path=model_config_path
        
        self.generation_config=generation_config
        
        self.embeddings_model=embeddings_model
        
        # Инициализация короткой памяти для хранения последних 10 промпт-ответов
        self.short_memory = deque(maxlen=max_history_length)
        
        self.initialize_tokenizer()
        self.initialize_model()
        self.initialize_streamer()
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
            
        self.start_background_cache_updater()

    def initialize_tokenizer(self):
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            padding_side="left"
            )
            
        # Убедитесь, что pad_token установлен
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        #logger.info("🔹 **Изначальные настройки токенизатора** 🔹")
        #logger.info(self.tokenizer)

    def initialize_model(self):
        """
        Инициализация модели с использованием одной из трех библиотек: transformers, vLLM или SGLang.
        Выбор библиотеки осуществляется через флаги USE_TRANSFORMERS, USE_VLLM, USE_SGLANG.
        """
        if self.use_gptq_loader:
            self.model = GPTQModel.load(
                self.model_name,
                device=self.device
                )
            return
        if self.use_vllm_loader:
            # Инициализация модели через vLLM
            self.model = LLM(
                self.model_name,
                dtype="float16" if torch.cuda.is_available() else "float32",
                tensor_parallel_size=1,  # Количество GPU для распараллеливания
            )
            return
        if self.use_awq_loader:
            model = AutoAWQForCausalLM.from_quantized(
                self.model_name,
                fuse_layers=True,
                trust_remote_code=False,
                safetensors=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            return
        else:
            # Использование стандартной библиотеки transformers
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                config=self.model_config  # Передача конфигурации
            )
        
        #self.basic_config = json.loads(self.model.config.to_json_string())

    def initialize_streamer(self):
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True, # пример: <｜end▁of▁sentence｜>
            timeout=60  # Увеличенное время ожидания
        )
        
    def generate_llama_prompt_template(self, messages, bos_token="<s>"):
        """
        Генерирует prompt_template для Llama на основе входных сообщений.
        
        :param messages: Список словарей с ключами 'role' и 'content'.
        :param bos_token: Токен начала последовательности (например, "<s>").
        :return: Отформатированная строка prompt_template.
        """
        if not messages:
            raise ValueError("Список сообщений пуст.")
        
        # Разделяем системное сообщение (если оно есть) и остальные сообщения
        if messages[0]['role'] == 'system':
            # Преобразуем content в строку, если это список
            system_content = messages[0]['content']
            if isinstance(system_content, list):
                system_content = ' '.join(str(item) for item in system_content)
            
            system_message = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_content.strip()}<|eot_id|>"
            )
            loop_messages = messages[1:]
        else:
            system_message = ""
            loop_messages = messages
        
        # Формируем prompt
        prompt_parts = [bos_token]
        for i, message in enumerate(loop_messages):
            # Проверяем чередование ролей
            if (message['role'] == 'user') != (i % 2 == 0):
                raise ValueError("Роли в диалоге должны чередоваться user/assistant/user/assistant...")
            
            # Добавляем системное сообщение перед первым сообщением
            if i == 0 and system_message:
                prompt_parts.append(system_message)
            
            # Преобразуем content в строку, если это список
            content = message['content']
            if isinstance(content, list):
                content = ' '.join(str(item) for item in content)
            
            # Формируем сообщение
            formatted_message = (
                f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                f"{content.strip()}<|eot_id|>"
            )
            prompt_parts.append(formatted_message)
            
            # Добавляем приглашение для генерации ответа, если это последнее сообщение от пользователя
            if i == len(loop_messages) - 1 and message['role'] == 'user' and self.use_prompt_template:
                prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        # Объединяем все части в одну строку
        return ''.join(prompt_parts)

    async def predict(self, user_input):
        """Асинхронная генерация ответа с использованием шаблона чата."""
        # Формируем сообщения для модели
        try:
            try:
                messages = self.convo + list(self.short_memory) + [{"role": "user", "content": user_input}]
            except Exception as e:
                print(f'\nException_1 in async def predict: {e}\nWe will use only user_input for generation')
                messages = [{"role": "user", "content": user_input}]
            
            # Добавляем пользовательский ввод в диалог
            self.convo.append({"role": "user", "content": user_input})
            
            try:
                # Ищем релевантные записи в долговременной памяти
                relevant_memories = self.long_memory.retrieve_relevant_memory(user_input)
                if relevant_memories:
                    print("Найдены релевантные записи из долговременной памяти:")
                    for memory in relevant_memories:
                        print(f"Prompt: {memory['prompt']}\nResponse: {memory['response']}\n")
                
                print("\nmessages", messages, "\n")
            except Exception as e:
                logging.error(f"Ошибка в predict -> retrieve_relevant_memory: {str(e)}")
            
            # Динамически настраиваем параметры генерации
            dynamic_params = self.adjust_parameters_based_on_context(user_input)

            # @lru_cache(maxsize=128)  # Кэшируем до 128 уникальных промптов => Ошибка в tokenize: unhashable type: 'list'
            # Слишком много мороки и неизвестно есть ли смысл
            '''
            def tokenize(cache_key):
                messages_list = [dict(message) for message in cache_key]
                
                # Токенизируем ввод
                inputs = self.tokenizer.apply_chat_template(
                    messages_list,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                return inputs
            
            # Преобразуем messages в кортеж кортежей для хешируемости
            cache_key = tuple(tuple(message.items()) for message in messages)
            try:
                inputs = tokenize(cache_key)
            except Exception as e:
                logging.error(f"Ошибка в tokenize: {str(e)}")
            '''
            if self.use_prompt_template:
                # Генерируем prompt_template с помощью нашей функции
                prompt_template = self.generate_llama_prompt_template(
                    messages, 
                    bos_token="<s>"
                    )
                
                inputs = self.tokenizer(
                    prompt_template, 
                    return_tensors='pt'
                    ).input_ids.cuda()
                
                print(f"Использован кастомный chat_template")
            else:
                # Токенизируем ввод
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True, 
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            
            # Создаем attention_mask
            attention_mask = inputs.ne(self.tokenizer.pad_token_id).int().to(self.device)
            inputs = inputs.to(self.device)
            
            # Формируем параметры генерации
            generation_kwargs = {
                "input_ids": inputs,  # Явно указываем ключ для входных данных
                "attention_mask": attention_mask,  # Добавляем attention_mask
                "streamer": self.streamer,
                # "stopping_criteria": StoppingCriteriaList([StopOnEOS(self.tokenizer.eos_token_id)]),
                **self.generation_config,
                **dynamic_params
            }
            
            # self.model.to(self.device) - You shouldn't move a model that is dispatched using accelerate hooks.

            # Запускаем генерацию в отдельном потоке
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            logger.debug("Параметры генерации: %s", generation_kwargs)
            logger.debug("Параметры thread: %s", thread)
            thread.start()

            response = ""
            async for new_token in self.stream_response():
                print(new_token[len(response):], end="", flush=True)
                if response.strip() in ["<think>\n</think>", "<think></think>"]:
                    print("⚠️ Бот сгенерировал пустой ответ, перезапускаем генерацию...")
                    return await self.predict(user_input)  # 🔥 Перегенерация
                response = new_token
                
            # Убираем все от <think> до </think> (включая теги) перед сохранением в память
            cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                
            self.convo.append({"role": "girlfriend", "content": cleaned_response})
            
            # Сохраняем диалог в долговременной памяти
            self.long_memory.add_to_long_memory(user_input, cleaned_response)
                
            self.short_memory.append({"role": "user", "content": user_input})
                
            return response
        except Exception as e:
            logging.error(f"Ошибка в predict: {str(e)}")
    
    async def stream_response(self):
        """Асинхронный поток вывода ответа в реальном времени."""
        partial_message = ""
        try:
            for new_token in self.streamer:
                if new_token is None:  # Если поток завершен
                    break
                partial_message += new_token
                yield partial_message
                await asyncio.sleep(0.005)
        except Exception as e:
            print(f"Ошибка при потоковом выводе: {e}")
            yield partial_message  # Возвращаем частичный результат
            
    def start_background_cache_updater(self):
        """Запускает фоновую задачу для обновления кэша user_preferences."""
        try:
            def update_cache():
                while True:
                    time.sleep(300)  # Обновляем по таймеру
                    self.long_memory.load_user_preferences()
                    logging.info("Кэш user_preferences обновлен.")

            Thread(target=update_cache, daemon=True).start()
        except Exception as e:
            logging.error(f"Ошибка в start_background_cache_updater: {str(e)}")
            
    async def chat_loop(self):
        """Асинхронный чат-бот."""
        print("Добро пожаловать! Вы можете начать общение с ботом. Для выхода введите 'exit' или 'quit'.")

        while True:
            try:
                try:
                    prompt = await asyncio.to_thread(input, "Вы: ")
                
                    # Обработка команд
                    command_pattern = re.compile(r'^/(\w+)\s*(.*)', re.IGNORECASE)
                    match = command_pattern.match(prompt.strip())
                except Exception as e:
                    logging.error(f"Exception_1 chat_loop: {e}")
                
                if prompt.lower() in ["exit", "quit"]:
                    print("Диалог завершен.")
                    break
                
                if match:
                    command, args = match.groups()
                    args = args.strip() if args else None
                    
                    if command.lower() == 'recall':
                        if not args:
                            print("Введите аргумент для /recall")
                            continue
                        self.long_memory.recall(prompt=args)
                        self.stream_response(prompt=args)
                        
                    elif command.lower() == 'forget':
                        self.long_memory.remove_last_conversation()
                        self.convo = self.convo[:-2]
                        print('\n')
                        
                    elif command.lower() == 'preference':
                        if not args or ":" not in args:
                            print("Пожалуйста, укажите предпочтение в формате 'ключ: значение'.")
                            continue
                        prompt = args.split(":", 1)[0].strip() if ":" in args else "general"
                        response = args.split(":", 1)[1].strip() if ":" in args else args.strip()
                        self.long_memory.save_user_preference(prompt=prompt, value=response)
                        logging.info(f"Сохранено предпочтение: {prompt} -> {response}")
                        self.stream_response(prompt=args)
                        
                    elif command.lower() == 'training':
                        if not self.convo or self.convo[-1]['role'] != 'user':
                            print("Нет предыдущего промпта для обучения.")
                            continue
                        original_prompt = self.convo[-1]['content']
                        print("Промпт сохранен. Введите корректный ответ:")
                        correct_response = input(Fore.WHITE + 'CORRECT RESPONSE: \n').strip()
                        self.long_memory.store_training_data(prompt=original_prompt, response=correct_response, quality="good")
                        logging.info(f"Сохранены данные для обучения: prompt={original_prompt}, response={correct_response}")
                    
                    elif command.lower() == 'reward':
                        # Сохраняем последний ответ как "хороший"
                        last_response = self.convo[-1]['content']
                        self.long_memory.store_training_data(prompt=self.convo[-2]['content'], response=last_response, quality="good")
                        print("Спасибо за обратную связь! Я запомню этот ответ как хороший.")

                    elif command.lower() == 'penalty':
                        # Сохраняем последний ответ как "плохой"
                        last_response = self.convo[-1]['content']
                        self.long_memory.store_training_data(prompt=self.convo[-2]['content'], response=last_response, quality="bad")
                        print("Спасибо за обратную связь! Я постараюсь улучшить этот ответ.")
                    
                    elif command.lower() == 'backup_database':
                        # Сохраняем последний ответ как "плохой"
                        self.long_memory.backup_database()
                        print("База данных сохранена")
                    
                    elif command.lower() == 'memorize':
                        try:
                            self.long_memory.store_conversations(prompt=args, response='Memory stored')
                        except Exception as e:
                            print(f"Ошибка при сохранении памяти: {str(e)}")
                        finally:
                            print('\n')
                    else:
                        print(f"Неизвестная команда: /{command}")
                else:
                    print("Бот: ", end="")
                    response = await self.predict(prompt)
                    print()

            except KeyboardInterrupt:
                print("\nДиалог прерван пользователем.")
                break
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                error_type = "crush"
                message = str(e)
                self.long_memory.store_errors(error_type, message)
                
                
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