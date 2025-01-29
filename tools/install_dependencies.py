import subprocess
import sys

# Список библиотек для установки
required_libraries = [
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",  # PyTorch
    "transformers",  # Hugging Face Transformers
    "peft",  # Parameter-Efficient Fine-Tuning (LoRA)
    "bitsandbytes",  # Quantization
    "pynvml",  # NVIDIA Management Library
    "schedule",  # Планировщик задач
    "PyQt5",  # GUI (PyQt5)
    "PyQtWebEngine",  # WebEngine для PyQt5
    "speechrecognition",  # Распознавание речи
    "pyaudio",  # Работа с аудио
    "numpy",  # Математические операции
    "vllm",  # High-throughput LLM inference
    "dataclasses",  # Упрощение создания классов данных
    "urllib3",  # Работа с URL
    "queue",  # Потокобезопасные очереди
    "threadpool_executor",  # Пул потоков
]

# Установка библиотек
def install_libraries():
    for lib in required_libraries:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"Успешно установлена библиотека: {lib}")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при установке библиотеки {lib}: {e}")

if __name__ == "__main__":
    print("Начало установки библиотек...")
    install_libraries()
    print("Все библиотеки установлены!")