# my own code's here
#from tools.train import GPTQTrainer, DEFAULT_CONFIG, TrainingScheduler
#import threading
#from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG
#from tools.avatar import AvatarController, AvatarConfig
#from tools.sql_memory import SQLMemory
#from tools.langchain_memory import LanguageChain

# Оптимизация инференса
from optimum.onnxruntime import ORTModelForCausalLM
from gptqmodel import GPTQModel
#import vllm  # Импорт vllm для возможной будущей интеграции

# Basic import
import logging
import re
import warnings
from typing import List, Optional
import asyncio
import json

# External libraries
import torch

# Check that little boy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    GenerationConfig,
    GPT2LMHeadModel, 
    GPT2Tokenizer
)

from threading import Thread
from functools import lru_cache


'''
Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent 
endless repetitions or incoherent outputs.
Avoid adding a system prompt; all instructions should be contained within the user prompt.
To ensure that the model engages in thorough reasoning, we recommend enforcing the model 
to initiate its response with "<think>\n" at the beginning of every output.
'''

# Параметры
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
USE_ONNX = False  # Включить для использования ONNX Runtime
USE_GPTQ = True  # Включить при использовании GPTQ-квантизированной модели

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Параметры модели
MODEL_NAME = "models/llm/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"

MAX_HISTORY_LENGTH = 5  # Ограничиваем историю диалога

MODEL_CONFIG = {
    # Автоматическая настройка реализации внимания (если включено, будет автоматически настроена реализация внимания)
    "_attn_implementation_autoset": True, 

    # Путь или имя модели
    "_name_or_path": "models/llm/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2", 

    # Архитектура модели
    "architectures": [
        "Qwen2ForCausalLM"
    ], 

    # Выпадение вероятности внимания (dropout) для предотвращения переобучения
    "attention_dropout": 0.0, 

    # ID токена начала строки (BOS)
    "bos_token_id": 151643, 

    # ID токена конца строки (EOS)
    "eos_token_id": 151643, 

    # Функция активации для скрытых слоев (например, "silu" — это активация SiLU)
    "hidden_act": "silu", 

    # Размер скрытого слоя (количество нейронов в слое)
    "hidden_size": 3584, 

    # Диапазон для инициализации весов (как сильно будут инициализированы веса)
    "initializer_range": 0.02, 

    # Размер промежуточного слоя (для некоторых моделей может быть больше, чем скрытый слой)
    "intermediate_size": 18944, 

    # Максимальная длина входной последовательности (включая токены BOS и EOS)
    "max_position_embeddings": 131072, 

    # Максимальное количество слоев окон
    "max_window_layers": 28, 

    # Тип модели (это Qwen2)
    "model_type": "qwen2", 

    # Количество голов внимания в слое
    "num_attention_heads": 28, 

    # Количество скрытых слоев (глубина сети)
    "num_hidden_layers": 28, 

    # Количество голов для ключей и значений
    "num_key_value_heads": 4, 

    # Конфигурация квантования модели
    "quantization_config": {
        # Количество бит на параметр модели (4 бита на вес)
        "bits": 4, 

        # Формат контрольной точки
        "checkpoint_format": "gptq", 

        # Применять описание активации (если True, описания будут применяться)
        "desc_act": True, 

        # Динамическое квантование (параметр динамической квантованности)
        "dynamic": None, 

        # Размер групп для квантования
        "group_size": 32, 

        # Использование головы языка (обычно для генеративных моделей)
        "lm_head": False, 

        # Метаинформация квантования
        "meta": {
            "damp_auto_increment": 0.0025,  # Параметры для изменения веса в процессе квантования
            "damp_percent": 0.1,  # Параметр изменения коэффициента
            "quantizer": [
                "gptqmodel:1.7.4"  # Версия квантователя
            ], 
            "static_groups": False,  # Использование статических групп
            "true_sequential": True,  # Должна ли модель использовать истинно последовательное квантование
            "uri": "https://github.com/modelcloud/gptqmodel"  # Ссылка на репозиторий квантователя
        }, 

        # Метод квантования
        "quant_method": "gptq", 

        # Симметричное квантование (если True, квантование будет симметричным)
        "sym": True  
    }, 

    # Параметры для нормализации RMS
    "rms_norm_eps": 1e-06, 

    # Масштабирование для использования ROPE (если используется)
    "rope_scaling": None, 

    # Параметр для масштаба ROPE (ротационное позиционное кодирование)
    "rope_theta": 10000, 

    # Скользящее окно для позиционного кодирования (если используется)
    "sliding_window": None, 

    # Привязать эмбеддинги слов (если False, эмбеддинги слов не будут привязаны)
    "tie_word_embeddings": False, 

    # Тип данных для PyTorch (например, bfloat16 для использования меньшего объема памяти)
    "torch_dtype": "bfloat16", 

    # Версия библиотеки transformers
    "transformers_version": "4.48.3", 

    # Использовать кэш для ускорения генерации
    "use_cache": True, 

    # Использовать ROPE (ротационное позиционное кодирование)
    "use_mrope": False, 

    # Использовать скользящее окно
    "use_sliding_window": False, 

    # Размер словаря (количество токенов)
    "vocab_size": 152064  
}

MODEL_CONFIG_PATH="model_config.json"

GENERATION_CONFIG = {
    # Максимальная длина последовательности, включая токены начала и конца
    # Both `max_new_tokens` (=512) and `max_length`(=20) seem to have been set. `max_new_tokens` 
    # will take precedence. Please refer to the documentation for more information. 
    # (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    # Default = None
    "max_length": None, 

    # Количество новых токенов, которые будут сгенерированы (None — это означает, что не задано)
    "max_new_tokens": 512, 

    # Минимальная длина генерируемой последовательности, default = 0
    "min_length": 5, 

    # Минимальное количество новых токенов, default = None
    # Both `min_new_tokens` (=5) and `min_length`(=5) seem to have been set. `min_new_tokens` 
    # will take precedence. Please refer to the documentation for more information. 
    # (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    "min_new_tokens": None, 

    # Остановить генерацию, если достигнут конец строки
    "early_stopping": True, 

    # Время, через которое генерация будет остановлена (если задано), default = None
    "max_time": 5, 

    # Строки, по которым генерация будет остановлена, default = None
    # ValueError: There are one or more stop strings, either in the arguments to `generate` or 
    # in the model's generation config, but we could not locate a tokenizer. When generating 
    # with stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`.
    # Default = None
    "stop_strings": None, 

    # Флаг, который управляет выбором случайных токенов (по умолчанию False, то есть без сэмплинга)
    # `diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. 
    # In this generation mode, `do_sample` must be set to `False`
    "do_sample": False, 

    # Количество использованных "лучей" для beam search (1 — это жадный поиск) Должно быть > 1
    # `streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1.
    "num_beams": 1, 

    # Количество групп лучей в beam search (error if not 1)
    "num_beam_groups": 1, 

    # Коэффициент для штрафа на длину ответа
    "penalty_alpha": None, 

    # Количество слоев для доли `dola_layers` (неясно что это)
    "dola_layers": None, 

    # Использовать кэш для ускорения генерации (по умолчанию — False)
    "use_cache": False, 

    # Конфигурация кэширования (если используется)
    "cache_implementation": None, 

    # Конфигурация кэша (если используется)
    "cache_config": None, 

    # Вернуть устаревший кэш (если используется)
    "return_legacy_cache": None, 

    # Температура для контроля случайности в выборке (1 — стандартное значение, больше — более случайно)
    "temperature": 0.6, 

    # Количество токенов, сгенерированных до обрезки
    "top_k": 30, 

    # Использовать top-p sampling (например, top_p=1.0 — это значит, что мы не ограничиваем выбор)
    "top_p": 0.96, 

    # Минимальная вероятность для фильтрации токенов, default = None
    "min_p": None, 

    # Параметр, регулирующий случайность выборки
    #🔥 Помогает модели избегать странных паттернов (экспериментально), default = 1
    "typical_p": 0.9, 

    # Порог для исключения токенов с вероятностью меньше этого значения
    "epsilon_cutoff": 0.0, 

    # Порог для cutoff (например, для исключения слабых токенов)
    "eta_cutoff": 0.0, 

    # Штраф на разнообразие сгенерированных строк
    # `diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering 
    # group beam search. In this generation mode, `num_beams` should be divisible by `num_beam_groups`
    "diversity_penalty": 0.0,

    # Штраф за повторение слов или фраз в строках, default = 1
    "repetition_penalty": 1.1, 

    # Штраф за повторение слов на уровне энкодера, default = 1
    "encoder_repetition_penalty": 1, 

    # Штраф на длину генерируемой строки, default = 1
    "length_penalty": 1.1, 

    # Запрещает повторение фраз размером n-грамм
    "no_repeat_ngram_size": 0, 

    # Список "плохих" токенов, которые не должны быть использованы
    "bad_words_ids": None, 

    # Список "обязательных" токенов, которые должны быть использованы
    "force_words_ids": None, 

    # Нормализует логи перед применением softmax для стабильности
    "renormalize_logits": False, 

    # Обязательные условия для генерируемой строки
    "constraints": None, 

    # ID токена начала строки (например, BOS токен)
    "forced_bos_token_id": None, 

    # ID токена конца строки (например, EOS токен)
    "forced_eos_token_id": None, 

    # Удалять некорректные значения, например, NaN, , default = False
    "remove_invalid_values": True, 

    # Использовать экспоненциальное уменьшение штрафа на длину, default = None
    "exponential_decay_length_penalty": None, 

    # Список токенов, которые будут подавлены (не использовать)
    "suppress_tokens": None, 

    # Список токенов для начала подавления
    "begin_suppress_tokens": None, 

    # ID для обязательного использования декодера
    "forced_decoder_ids": None, 

    # Бонус или штраф для продолжений в генерации
    "sequence_bias": None, 

    # Использовать восстановление токенов (для задач восстановления текста)
    "token_healing": False, 

    # Мощность Guidance (управляющий параметр для моделей с guidance)
    "guidance_scale": None, 

    # Уменьшение использования памяти, если включено
    "low_memory": None, 

    # Конфигурация для водяных знаков, если это нужно
    "watermarking_config": None, 

    # Количество генерируемых последовательностей
    "num_return_sequences": 1, 

    # Возвращать внимание модели для каждой позиции
    "output_attentions": False, 

    # Возвращать скрытые состояния модели
    "output_hidden_states": False, 

    # Возвращать оценки вероятностей токенов
    "output_scores": False, 

    # Логиты сгенерированных токенов
    "output_logits": None, 

    # Вернуть результат генерации как словарь (по умолчанию False)
    "return_dict_in_generate": False, 

    # ID токена паддинга
    "pad_token_id": None, 

    # ID токена начала строки
    "bos_token_id": 151643, 

    # ID токена конца строки
    "eos_token_id": 151643, 

    # Запрещает повторение фраз в энкодере
    "encoder_no_repeat_ngram_size": 0, 

    # Стартовый токен для декодера
    "decoder_start_token_id": None, 

    # Это помощник для асистентов (если True, это значит, что будет другая модель)
    "is_assistant": False, 

    # Количество токенов, которые будут использованы для задач ассистента
    "num_assistant_tokens": 20, 

    # Как изменяется количество ассистентных токенов
    "num_assistant_tokens_schedule": 'constant', 

    # Порог для уверенности ассистента
    "assistant_confidence_threshold": 0.4, 

    # Количество токенов, которые нужно посмотреть назад для поиска шаблонов
    "prompt_lookup_num_tokens": None, 

    # Максимальный размер n-грамм для сопоставлений
    "max_matching_ngram_size": None, 

    # Ранний выход для ассистента (если включено, он завершит процесс быстрее)
    "assistant_early_exit": None, 

    # Количество токенов назад, которые нужно смотреть для ассистента
    "assistant_lookbehind": 10, 

    # Количество токенов назад, которые нужно смотреть для целевого текста
    "target_lookbehind": 10, 

    # Дополнительные аргументы для генерации
    "generation_kwargs": {}, 

    # Использовать конфигурацию модели
    "_from_model_config": True, 

    # Версия библиотеки transformers
    # pip show transformers
    "transformers_version": '4.48.3'
}

SYSTEM_PROMPT = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
    {"role": "system", "content": "For mathematical questions, think step by step. Always include the final answer inside <math>{answer}</math>."},
    {"role": "system", "content": "Always follow these rules:\n"
                                 "1. Start response with <think>analysis</think>\n"
                                 "2. Provide a thorough and well-reasoned response.\n"
                                 "3. Answer and think only in ENGLISH\n"
                                 "Example:\n"
                                 "<think>User asked about... I need to think...</think>\n"
                                 "**Answer:** Full answer here..."}
]
    

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
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = None
        self.system_prompt = SYSTEM_PROMPT
        self.streamer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.basic_config = None
        self.model_config_path = MODEL_CONFIG_PATH
        self.model_config = MODEL_CONFIG
        
        self.create_model_config()
        
        self.initialize_tokenizer()
        self.initialize_model()
        self.initialize_streamer()
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
            
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
        Инициализация модели с использованием transformers.
        """
        if USE_GPTQ:
            # Чтение конфигурации из файла
            if self.model_config_path:
                # Загрузка конфигурации из файла
                with open(self.model_config_path, "r", encoding="utf-8") as json_file:
                    config_data = json.load(json_file)
                self.model_config = config_data
            else:
                # Загрузка конфигурации из директории модели
                self.model_config = AutoConfig.from_pretrained(self.model_name)
                
            self.model = GPTQModel.load(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float16,
                device_map="auto",  # Автоматическое распределение по GPU/CPU
                config=self.model_config  # Передача конфигурации
                )
        elif USE_ONNX:
            self.model = ORTModelForCausalLM.from_pretrained(
                self.model_name,
                provider="CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider",
                export=not USE_ONNX  # Автоматическая конвертация при первом запуске
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
        if self.device == "cuda" and not USE_ONNX:
            self.model.to(self.device)
            
        #self.basic_config = json.loads(self.model.config.to_json_string()) 

    def initialize_streamer(self):
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60  # Увеличенное время ожидания
        )
    
    def calculate_token_length(self, text: str) -> int:
        """Вычисляет длину текста в токенах (оптимизированная версия)"""
        return self.tokenizer(text, return_length=True)["length"][0]
    
    def safe_softmax(self, logits):
        """Преобразуем логиты в нормальный softmax (избегаем inf/nan)"""
        #print("🔥 Raw logits:", logits[:10])  # Вывод первых 10 логитов
        #print("🔥 Min logit:", logits.min().item(), "Max logit:", logits.max().item())
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)  # Убираем NaN
        logits = torch.where(torch.isinf(logits), torch.full_like(logits, -1e9), logits)  # Убираем Inf
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
                "temperature": min(0.9, GENERATION_CONFIG["temperature"] + 0.2),
                "repetition_penalty": 1.1
            })
        
        # Проверка фактологических ключевых слов
        elif any(keyword in input_lower for keyword in factual_keywords):
            params.update({
                "temperature": max(0.3, GENERATION_CONFIG["temperature"] - 0.2),
                "top_k": 20,
                "repetition_penalty": 1.5
            })
        
        return params

    async def predict(self, user_input):
        """Асинхронная генерация ответа с использованием шаблона чата."""
        # Формируем сообщения для модели
        messages = SYSTEM_PROMPT + [{"role": "user", "content": user_input}]

        # Токенизируем ввод (ФИКС ошибки attention_mask)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        inputs = inputs.to(self.device)
        
        # Создаем attention_mask
        attention_mask = inputs.ne(self.tokenizer.pad_token_id).int().to(self.device)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs).logits[:, -1, :].to(torch.bfloat16)
            logits = torch.clamp(logits, min=-10, max=10)  # 🔥 Ограничиваем диапазон значений
            logits = self.safe_softmax(logits)
            
            next_token_id = torch.multinomial(logits, num_samples=1)

        # Динамически настраиваем параметры генерации
        dynamic_params = self.adjust_parameters_based_on_context(user_input)
        
        generation_kwargs = dict(
            input_ids=inputs,
            attention_mask=attention_mask,  # Добавляем attention_mask
            streamer=self.streamer,
            stopping_criteria=StoppingCriteriaList([StopOnEOS(self.tokenizer.eos_token_id)]),
            **GENERATION_CONFIG,
            **dynamic_params
        )
        
        '''
        check = True
        if check:
            self.compare_configs()
            logger.info(self.model.config.to_json_string())
            print(f'Параметры генерации модели:\n{self.model.generation_config.to_dict()}')
            check = False
        '''

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
            
        return response
    
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

'''
Для дальнейшего улучшения рекомендую:
Добавить проверку токенов в реальном времени
Реализовать механизм перефразирования длинных ответов
Добавить эмоциональную окраску ответов через специальные токены
Внедрить систему приоритетов для разных типов запросов
'''

'''
Изначальные настройки токенизатора
BASIC_TOKENIZER_CONFIGURATION = LlamaTokenizerFast(
    name_or_path='models/llm/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2', 
    vocab_size=151643, 
    model_max_length=2048, 
    is_fast=True, 
    padding_side='left', 
    truncation_side='right', 
    special_tokens={
        'bos_token': '<｜begin▁of▁sentence｜>', 
        'eos_token': '<｜end▁of▁sentence｜>', 
        'pad_token': '<|image_pad|>'}, 
    clean_up_tokenization_spaces=False, 
    added_tokens_decoder={
        151643: AddedToken(
            "<｜end▁of▁sentence｜>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151644: AddedToken(
            "<｜User｜>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151645: AddedToken(
            "<｜Assistant｜>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151646: AddedToken(
            "<｜begin▁of▁sentence｜>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151647: AddedToken(
            "<|EOT|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151648: AddedToken(
            "<think>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151649: AddedToken(
            "</think>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151650: AddedToken(
            "<|quad_start|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151651: AddedToken(
            "<|quad_end|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151652: AddedToken(
            "<|vision_start|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151653: AddedToken(
            "<|vision_end|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151654: AddedToken(
            "<|vision_pad|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151655: AddedToken(
            "<|image_pad|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151656: AddedToken(
            "<|video_pad|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=True
            ),
        151657: AddedToken(
            "<tool_call>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151658: AddedToken(
            "</tool_call>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151659: AddedToken(
            "<|fim_prefix|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151660: AddedToken(
            "<|fim_middle|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151661: AddedToken(
            "<|fim_suffix|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151662: AddedToken(
            "<|fim_pad|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151663: AddedToken(
            "<|repo_name|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
        151664: AddedToken(
            "<|file_sep|>", 
            rstrip=False, 
            lstrip=False, 
            single_word=False, 
            normalized=False, 
            special=False
            ),
}
)
'''