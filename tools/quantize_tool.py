# my own code's here
#from tools.train import GPTQTrainer, DEFAULT_CONFIG, TrainingScheduler
#import threading
#from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG
#from tools.avatar import AvatarController, AvatarConfig
#from tools.sql_memory import SQLMemory
#from tools.langchain_memory import LanguageChain

# Basic import
#import os
#import sys
import logging
import psutil

# External libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset

import torch

# Инициализация компонентов
#trainer = GPTQTrainer(DEFAULT_CONFIG)  # Инициализация тренера для обучения GPTQ
#scheduler = TrainingScheduler(trainer)  # Планировщик для управления обучением
#audio = AudioManager(DEFAULT_VOICE_CONFIG)  # Менеджер аудио для воспроизведения речи

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Параметры
MODEL_NAME = "models/llm/Qwen2.5-1.5B"


class ChatBot:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.quantized_model_id = f"{self.model_name}-4bit"
        self.model = None  # Модель для генерации текста
        self.tokenizer = None  # Токенизатор для обработки текста
        self.load_model()  # Загрузка модели и токенизатора
        
    # os.makedirs(quantized_model_dir, exist_ok=True)
    def get_wikitext2(self, tokenizer, nsamples, seqlen):
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
            lambda x: len(x["text"]) >= seqlen)

        return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]

    @torch.no_grad()
    def calculate_avg_ppl(self, model, tokenizer):
        from gptqmodel.utils import Perplexity

        ppl = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="train",
            text_column="text",
        )

        all = ppl.calculate(n_ctx=512, n_batch=512)

        # average ppl
        avg = sum(all) / len(all)

        return avg

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        logging.info("Токенизатор успешно загружен.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        """Проверяем если квантизируемая модель уже существует"""
        if os.path.exists(self.quantized_model_id):  # Проверяем, существует ли папка с квантизованной моделью
            logging.info("Квантизованная модель найдена, загружаем...")
            self.model = GPTQModel.from_quantized(self.quantized_model_id, device=device)
        else:
            """Загружаем или квантизирует модель."""
            try:
                traindataset = self.get_wikitext2(self.tokenizer, nsamples=256, seqlen=1024)
                
                quantize_config = QuantizeConfig(
                    bits=4,  # Квантизация в 4 бита
                    group_size=128,
                    desc_act=True
                )

                # load un-quantized model, the model will always be force loaded into cpu
                self.model = GPTQModel.load(self.model_name, quantize_config)
                logging.info("basic модель успешно загружена.")

                # Загружаем неквантизованную модель
                model = GPTQModel.load(self.model_name, quantize_config)
                logging.info("Неквантизированная модель загружена.")

                # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
                # with value under torch.LongTensor type.
                model.quantize(traindataset)
                logging.info("Модель успешно квантизована.")

                # Сохраняем квантизованную модель
                model.save(self.quantized_model_id)
                logging.info("Модель успешно сохранена.")

            except Exception as e:
                logging.error(f"Ошибка загрузки модели: {str(e)}")
                self.model = None
                self.tokenizer = None
            
        # Загружаем квантизованную модель в GPU
        self.model = GPTQModel.load(self.quantized_model_id, device=device)
        logging.info("Квантизованная модель успешно загружена после квантизации.")

        #self.model = torch.compile(self.model)  # Оптимизация модели
        #logging.info(f"Модель загружена и оптимизирована на {device}.")

    def generate_response(self, context):
        """Генерирует ответ модели."""
        if self.model is None:
            return "Ошибка: Модель не загружена."

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            tokenized_test = self.tokenizer.encode(context)
            logging.info(f"Токенизированный ввод (без батчинга): {tokenized_test}")

            # Создаём input_ids + attention_mask
            inputs = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True).to(device)

            # Проверка на пустые input_ids или input_ids с нулями
            if inputs["input_ids"].numel() == 0:
                logging.error("Ошибка: `input_ids` пустой!")
                return "Ошибка: Невозможно обработать пустой ввод."

            if torch.any(inputs["input_ids"] == 0):
                logging.warning("Предупреждение: `input_ids` содержит запрещённые значения (0). Заменяем их на `pad_token_id`.")
                inputs["input_ids"][inputs["input_ids"] == 0] = self.tokenizer.pad_token_id

            # Указываем pad_token_id, чтобы избежать ошибок
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            logging.info(f"Отладка input_ids перед генерацией: {inputs['input_ids']}")
            logging.info(f"Форма input_ids: {inputs['input_ids'].shape}")
            
            hidden_states = self.model(inputs["input_ids"])
            logging.info(f"Выход скрытых слоёв: {hidden_states}")

            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],  # ✅ Теперь передаём attention mask
                max_new_tokens=64,  # ✅ Ограничиваем число токенов (фикс для CUDA)
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id  # ✅ Добавляем pad_token_id
            )

            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        except Exception as e:
            logging.error(f"Ошибка при генерации ответа: {str(e)}")
            return "Извините, произошла ошибка."


    def start_chat_loop(self):
        """Основной цикл диалога в терминале."""
        print("Чат-бот запущен. Введите сообщение или 'exit' для выхода.")

        while True:
            user_input = input("Вы: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Диалог завершен.")
                break

            response = self.generate_response(user_input)
            print(f"Бот: {response}")


if __name__ == "__main__":
    # Инициализация чат-бота
    chat_bot = ChatBot()
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