from TTS.api import TTS

import asyncio
import logging
import queue
import threading
import time
import os
import wave
from typing import Dict

import numpy as np
import pyaudio
import torch
import torch.serialization
import whisper
from speechbrain.inference import SpeakerRecognition


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logging.info(f"Using device: {device}")


class SpeechSynthesizer:
    def __init__(self, config=None):
        self.config = config
        self.audio_queue = queue.Queue()
        self.playback_active = True
        self._init_tts()
        self._init_playback()
        logging.info("Initializing SpeechSynthesizer...")

    def _init_tts(self):
        """Инициализация Coqui TTS с персональным голосом"""
        try:
            if os.path.exists(self.config.tts_model_path):
                self.model = TTS(
                    model_path=self.config.tts_model_path,
                    config_path=os.path.join(self.config.tts_model_path, "config.json"),
                    progress_bar=False,
                    gpu=torch.cuda.is_available()
                )
                logging.info("Model loaded from local path.")
            else:
                logging.info("Model not found locally. Initializing default model...")
                self.model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                logging.info("Default model initialized.")
        except Exception as e:
            logging.error(f"Ошибка загрузки TTS модели: {str(e)}")
            raise

    def _init_playback(self):
        """Инициализация аудиовоспроизведения"""
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.config.sample_rate,
            output=True,
            frames_per_buffer=2048  # Увеличиваем буфер для плавности
        )
        self.thread = threading.Thread(target=self._playback_worker)
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        self.playback_active = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

    def synthesize(self, text: str):
        """Синтез текста с оптимизацией"""
        try:
            audio = self.model.tts(
                text=text,
                speaker_wav=self.config.speaker_wav,
                language=self.config.language,
                speed=1.2,  # Ускоряем на 20%
                temperature=0.6  # Меньше вариативности для стабильности
            )
            audio_data = np.array(audio, dtype=np.float32)
            self.audio_queue.put(audio_data)
        except Exception as e:
            logging.error(f"Ошибка синтеза: {str(e)}")

    def _playback_worker(self):
        """Поток воспроизведения"""
        while self.playback_active:
            try:
                audio = self.audio_queue.get(timeout=0.1)  # Уменьшаем таймаут для отзывчивости
                self.stream.write(audio.tobytes())
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Ошибка воспроизведения: {str(e)}")

class SpeechRecognizer:
    def __init__(self, config=None):
        self.config = config
        self.mic = None
        self.recognizer = None
        self.speaker_verifier = None
        self._init_recognition()
        logging.info("Initializing SpeechRecognizer...")

    def _init_recognition(self):
        """Инициализация Whisper и SpeechBrain"""
        try:
            # Загрузка Whisper
            if os.path.exists(self.config.whisper_model_path):
                self.model = whisper.load_model(self.config.whisper_model_path)
            else:
                self.model = whisper.load_model("medium", download_root=self.config.path_to_cache)

            # Загрузка SpeechBrain SpeakerRecognition локально
            self.speaker_verifier = SpeakerRecognition.from_hparams(
                source=self.config.speechbrain_model_path,  # Путь к локальной модели
                savedir=self.config.speechbrain_model_path
            )

            # Загрузка эталонного аудио твоего голоса
            self.reference_voice = self.config.reference_voice_path  # Путь к твоему голосу (WAV-файл)
            self.speaker_id = self.config.speaker_id

            self.recognizer = pyaudio.PyAudio()
            self.mic = self.recognizer.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
        except Exception as e:
            logging.error(f"Ошибка инициализации STT: {str(e)}")
            raise

    def __del__(self):
        if hasattr(self, 'mic'):
            self.mic.stop_stream()
            self.mic.close()
        if hasattr(self, 'recognizer'):
            self.recognizer.terminate()

    def listen_once(self):
        """Однократное прослушивание с проверкой голоса"""
        try:
            audio_data = []
            for _ in range(int(self.config.sample_rate / 1024 * 10)):  # 5 секунд записи
                data = self.mic.read(1024, exception_on_overflow=False)
                audio_data.append(data)
            audio = b''.join(audio_data)

            # Сохранение временного файла для анализа
            with wave.open("temp.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio)

            # Проверка голоса с помощью SpeechBrain
            score, prediction = self.speaker_verifier.verify_files(self.reference_voice, "temp.wav")
            speaker_detected = prediction.item()  # True/False, является ли голос твоим
            confidence = score.item()  # Уверенность (0-1)

            if not speaker_detected or confidence < 0.2:  # Порог уверенности можно настроить
                logging.info(f"Голос не распознан как твой (уверенность: {confidence:.2f}).")
                return None

            # Распознавание текста с помощью Whisper
            result = self.model.transcribe("temp.wav", language="ru")
            text = result["text"].strip()
            return text if text else None

        except Exception as e:
            logging.error(f"Ошибка распознавания: {str(e)}")
            return None

class AudioManager:
    def __init__(self, config):
        #self.synthesizer = SpeechSynthesizer(config)
        #self.recognizer = SpeechRecognizer(config)
        self.is_listening = False

    def listen_and_recognize(self):
        if self.is_listening:
            return None
        self.is_listening = True
        try:
            text = self.recognizer.listen_once()
            return text
        finally:
            self.is_listening = False

    async def speak(self, text):
        """Асинхронное воспроизведение"""
        self.synthesizer.synthesize(text)
        # Ожидаем начала воспроизведения
        await asyncio.sleep(0.05)  # Минимальная задержка для старта


'''
Основные предложения:
Шумоподавление :
Вы уже предусмотрели параметр noise_reduction в конфигурации, но не реализовали его 
использование. Для шумоподавления можно использовать библиотеку noisereduce.
Логирование :
Добавьте систему логирования для отслеживания ошибок и событий. Это поможет вам легче 
находить проблемы и отлаживать код.
Оптимизация производительности :
Используйте многопоточность и асинхронное программирование для повышения производительности.
Убедитесь, что все ресурсы (например, аудиопотоки) правильно освобождаются.
Интеграция с облачными сервисами :
Если вы хотите использовать облачные сервисы для распознавания речи, рассмотрите 
интеграцию с Google Speech-to-Text или Amazon Transcribe.
Обработка исключений :
Некоторые блоки кода могут вызывать исключения, которые не обрабатываются должным 
образом. Добавьте более детальную обработку ошибок.
Кодирование аудио :
Убедитесь, что формат аудио (например, .ogg) поддерживается вашей системой. Если нет, 
используйте универсальные форматы, такие как .wav.
'''

