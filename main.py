'''
#from tools.train import GPTQTrainer, DEFAULT_CONFIG, TrainingScheduler
import threading
#from tools.sound import AudioManager, DEFAULT_VOICE_CONFIG
#from tools.avatar import AvatarController, AvatarConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftConfig, PeftModel
import psutil
import json
import os
import speech_recognition as sr
#from gptqmodel import GPTQModel, QuantizeConfig
'''

import sys
#from PyQt5.QtWidgets import QApplication
#from PyQt5.QtWidgets import QMainWindow
import torch
#from llama_cpp import Llama

from tools.sql_memory import SQLMemory

# Вывод информации о памяти GPU
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


# Инициализация компонентов
'''
trainer = GPTQTrainer(DEFAULT_CONFIG)  # Инициализация тренера для обучения GPTQ
scheduler = TrainingScheduler(trainer)  # Планировщик для управления обучением
audio = AudioManager(DEFAULT_VOICE_CONFIG)  # Менеджер аудио для воспроизведения речи
recognizer = sr.Recognizer()  # Распознавание речи
'''
MODEL_NAME = 'deepseek-r1:1.5b'


class ChatBot:
    def __init__(self):
        self.model = None  # Модель для генерации текста
        self.tokenizer = None  # Токенизатор для обработки текста
        
        #self.memory = self.load_memory(MEMORY_FILE)  # Загрузка истории диалога
        #self.error_memory = self.load_memory(ERRORS_FILE)  # Загрузка ошибок
        #self.audio = audio  # Менеджер аудио
        #self.use_audio = True  # Флаг для использования аудио
        
        self.load_model()  # Загрузка модели и токенизатора
        self.model_name = MODEL_NAME

    def load_model(self):
        """Загружает GPTQ-модель и токенизатор."""
        try:
            self.model_configure()
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            self.model = None
            self.tokenizer = None

    def model_configure(self):
        pass
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Токенизатор успешно загружен.")
        # Конфигурация квантования
        quant_config = QuantizeConfig(
            bits=4,  # Укажите нужное количество битов
            group_size=128,
            desc_act=True  # Можно настроить по необходимости
        )
        self.model = GPTQModel.from_quantized(
            MODEL_NAME,
            quantize_config=quant_config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = torch.compile(self.model)
        self.model.tie_weights()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        print(f"Модель успешно загружена на: {device}")
        '''

    @staticmethod
    def load_memory(file_path):
        """Загружает память из файла."""
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"Ошибка загрузки памяти: {str(e)}")
            return []
    
    def process_voice_input(self):
        """Обрабатывает голосовой ввод пользователя."""
        if not self.use_audio:
            return input("Введите текст: ")  # Если аудио отключено, используем текстовый ввод

        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=5)
                return recognizer.recognize_whisper(audio, language="russian")
            except sr.WaitTimeoutError:
                print("Голосовой ввод не распознан.")
                return ""
            except Exception as e:
                print(f"Ошибка аудио: {str(e)}")
                self.use_audio = False  # Отключаем аудио при ошибке
                return input("Введите текст: ")  # Переходим на текстовый ввод

    def save_memory(self, data, file_path):
        """Сохраняет память в файл."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения памяти: {str(e)}")

    def format_context(self, messages):
        """Формирует контекст из последних сообщений."""
        context = system_prompt
        for msg in messages[-MAX_HISTORY:]:
            context += f"\n<|{msg['role']}|>{msg['content']}</s>"
        return context + "\n<|assistant|>"

    def generate_response(self, context):
        """Генерирует ответ модели."""
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to("cuda")
        attention_mask = torch.ones_like(input_ids).to("cuda")
        output = self.model.generate(
            input_ids,
            max_length=3072,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            min_length=20,
            repetition_penalty=1.2,
            attention_mask=attention_mask
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def handle_command(self, command):
        """Обрабатывает команды пользователя."""
        if command == "стоп":
            self.save_memory(self.memory, MEMORY_FILE)
            return True
        elif command == "исправь":
            self.error_memory.extend(self.memory[-2:])
            self.save_memory(self.error_memory, ERRORS_FILE)
        elif command == "перезагрузка":
            self.memory = []
        elif command == "сохранить":
            self.save_memory(self.memory, MEMORY_FILE)
        return False

    def incremental_learning(self, new_data):
        """Осуществляет инкрементальное обучение модели на новом наборе данных."""
        try:
            #trainer.update_model_with_new_data(new_data)
            print("Модель успешно обновлена!")
        except Exception as e:
            print(f"Ошибка при инкрементальном обучении: {str(e)}")

#class MainWindow(QMainWindow):
class MainWindow():
    def __init__(self):
        super().__init__()
        '''
        # Конфигурация для аватара
        self.avatar_config = AvatarConfig(
            model_url="https://models.silero.ai/models/tts/ru/v4_ru.pt",
            viseme_map={
                'sil': 0.0, 'PP': 0.3, 'FF': 0.2,
                'TH': 0.4, 'DD': 0.35, 'kk': 0.5
            }
        )
        self.use_avatar = True  # Флаг для использования аватара
        try:
            print(type(self.avatar_config))  # Проверка типа конфигурации
            self.avatar = AvatarController(self.avatar_config)  # Инициализация аватара
            self.setCentralWidget(self.avatar.container)  # Установка аватара в главное окно
        except Exception as e:
            print(f"Ошибка загрузки аватара: {str(e)}")
            self.use_avatar = False  # Отключаем аватар при ошибке
            self.avatar = None
        '''

if __name__ == "__main__":    
    sql_memory = SQLMemory() # Инициализация sql памяти
    sql_memory.model_name = MODEL_NAME
    conversations = sql_memory.fetch_conversations()
    sql_memory.create_vector_db(conversations=conversations)
    sql_memory.start_chat_loop()
    
    #chat_bot = ChatBot()  # Инициализация чат-бота
    #app = QApplication(sys.argv)
    #chat = MainWindow()
    #window.show()
    #window.start_chat_loop()
    #sys.exit(app.exec_())
    

'''
user_input = self.bot.process_voice_input()  # Получение голосового ввода
if not user_input:
    continue
if self.bot.handle_command(user_input.lower()):  # Обработка команд
    break
self.bot.memory.append({"role": "user", "content": user_input})  # Добавление ввода в память
context = self.bot.format_context(self.bot.memory)  # Формирование контекста
response = self.bot.generate_response(context)  # Генерация ответа
self.bot.memory.append({"role": "assistant", "content": response})  # Добавление ответа в память

# Анимация речи аватара (если аватар доступен)
if self.use_avatar and self.avatar:
    try:
        self.avatar.animate_speech(response)  # Анимация речи аватара
    except Exception as e:
        print(f"Ошибка анимации аватара: {str(e)}")
        self.use_avatar = False  # Отключаем аватар при ошибке

# Воспроизведение ответа (если аудио доступно)
if self.bot.use_audio:
    try:
        self.bot.audio.speak(response)  # Воспроизведение ответа
    except Exception as e:
        print(f"Ошибка воспроизведения аудио: {str(e)}")
        self.bot.use_audio = False  # Отключаем аудио при ошибке

print(f"\nБот: {response}\n")
if len(self.bot.memory) > MAX_HISTORY:  # Ограничение истории
    self.bot.memory = self.bot.memory[-MAX_HISTORY:]

# Инкрементальное обучение
self.bot.incremental_learning([{"input": user_input, "output": response}])

except KeyboardInterrupt:
print("\nЗавершение работы...")

finally:
self.bot.save_memory(self.bot.memory, MEMORY_FILE)  # Сохранение памяти
self.bot.save_memory(self.bot.error_memory, ERRORS_FILE)  # Сохранение ошибок
trainer.train()  # Запуск обучения


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
Убедитесь, что все компоненты программы интегрированы правильно и взаимодействуют между 
собой без проблем 1.
Используйте многопоточность для выполнения задач, таких как фоновое обучение, чтобы не 
нагружать основной поток выполнения программы 1.
Расширение функциональности :
Рассмотрите возможность добавления новых функций, таких как управление освещением, 
реализация системы жестов или интеграция Faceware-подобной технологии 1.
Используйте WebAssembly для выполнения сложных вычислений на стороне клиента, что 
позволит ускорить обработку данных 7.
Да, конечно, присылайте код! Я помогу вам его проанализировать, улучшить и предложу 
идеи для дальнейшего развития. Если что-то в вашем подходе можно оптимизировать или 
исправить, я обязательно укажу на это и предложу альтернативные решения.

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
Используйте torch.cuda.empty_cache() для очистки памяти GPU после завершения операций, 
чтобы освободить ресурсы для других задач.
Добавление новых функций :
Рассмотрите возможность добавления новых функций, таких как управление освещением, система 
жестов или интеграция Faceware-подобной технологии.

Обучение модели в процессе общения — это действительно интересная задача. Однако 
стоит учитывать, что обучение нейросетей "на лету" (online learning) требует значительных 
вычислительных ресурсов, особенно если модель большая. Для локального использования с 
ограниченными ресурсами можно рассмотреть следующие подходы:
Fine-tuning через LoRA (Low-Rank Adaptation) : Это метод, который позволяет адаптировать 
модель с минимальными затратами ресурсов. Вместо полного пересчета весов модели, LoRA 
изменяет только небольшую часть параметров.
Использование памяти (memory buffer) : Модель может сохранять ключевые моменты из диалогов 
в виде текстовых фрагментов или эмбеддингов, которые затем используются для генерации ответов. Это не требует переобучения модели, но позволяет ей "запоминать" контекст.
Инкрементальное обучение : Модель обучается на новых данных постепенно, сохраняя знания из 
предыдущих этапов. Это сложнее реализовать, но может быть эффективным.
Модель Qwen2.5-0.5B-Instruct-GPTQ-Int8 : Вы выбрали компактную версию модели, что разумно 
для локального использования. Она подходит для экспериментов, но её возможности ограничены 
по сравнению с более крупными моделями. Если вы хотите добавить функциональность, например, 
работу с 3D-моделями или звуком, можно интегрировать дополнительные библиотеки и сервисы.
Легковесность системы : Чтобы минимизировать нагрузку на систему, можно:
Разделить задачи на отдельные микросервисы (например, один процесс для обработки текста, 
другой для работы со звуком).
Использовать кэширование для часто используемых данных.
Добавить проверки на работоспособность каждого компонента (например, если звуковая система 
недоступна, автоматически переключаться на текстовый режим).
Библиотеки и инструменты для будущего :
Gradio или Streamlit : Для создания пользовательского интерфейса.
LangChain или LlamaIndex : Для работы с внешними данными и контекстом.
PyTorch/TensorFlow : Для реализации обучения и fine-tuning.
FAISS или Annoy : Для быстрого поиска похожих данных в памяти модели.
SpeechRecognition и gTTS : Для работы со звуком.
Blender/Three.js : Для интеграции 3D-моделей.
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