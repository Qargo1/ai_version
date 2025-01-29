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


import os
import queue
import threading
import torch
import pyaudio
import numpy as np
import wave
from typing import Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torchaudio
import speech_recognition as sr
import librosa  # Добавляем библиотеку для анализа аудио <button class="citation-flag" data-index="7">
import torchvision  # Если потребуется работа с визуальными данными
import noisereduce as nr  # Библиотека для шумоподавления
import logging  # Логирование

from silero import silero_stt, silero_tts, silero_te
import zipfile
from glob import glob


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class VoiceConfig:
    speaker: str = 'kseniya'
    model_id: str = 'v4_ru'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_rate: int = 48000
    language: str = 'ru'
    put_accent: bool = True
    put_yo: bool = True
    volume: float = 0.9
    speech_rate: int = 160
    soundbank_dir: str = "sounds"
    sound_format: str = "wav"
    default_volume: float = 0.9
    noise_reduction: bool = True  # Новый параметр для шумоподавления

# Конфигурация по умолчанию
DEFAULT_VOICE_CONFIG = VoiceConfig(
    speaker='kseniya',
    model_id='v4_ru',
    sample_rate=48000,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    language='ru',
    volume=0.9,
    speech_rate=160,
    soundbank_dir="sounds",
    sound_format="wav",
    noise_reduction=True  # Включаем шумоподавление
)

class SpeechSynthesizer:
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.playback_active = True
        self.soundbank = self._load_soundbank(config.soundbank_dir, config.sound_format)
        self._init_tts()
        self._init_playback()

    def _init_tts(self):
        """Инициализация модели TTS"""
        try:
            self.model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=self.config.language,
                speaker=self.config.model_id
            )
            self.model.to(torch.device(self.config.device))
        except Exception as e:
            logging.error(f"Error loading TTS model: {str(e)}")
            raise

    def _init_playback(self):
        """Инициализация аудиовоспроизведения"""
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.config.sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        self.thread = threading.Thread(target=self._playback_worker)
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        if hasattr(self, 'stream'):
            self.playback_active = False
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

    def _load_soundbank(self, sound_dir: str, fmt: str) -> Dict[str, np.ndarray]:
        """Загрузка пользовательских звуков из папки"""
        soundbank = {}
        if os.path.exists(sound_dir):
            for file in os.listdir(sound_dir):
                if file.endswith(f".{fmt}"):
                    key = os.path.splitext(file)[0].lower()
                    try:
                        with wave.open(os.path.join(sound_dir, file), 'rb') as wav:
                            audio_data = np.frombuffer(wav.readframes(-1), dtype=np.int16)
                            soundbank[key] = audio_data.astype(np.float32) / 32768.0
                    except Exception as e:
                        logging.error(f"Error loading {file}: {str(e)}")
        return soundbank

    def play_sound(self, sound_name: str) -> None:
        """Воспроизведение пользовательского звука"""
        audio = self.soundbank.get(sound_name.lower())
        if audio is not None:
            self.audio_queue.put(audio)
        else:
            logging.warning(f"Sound {sound_name} not found!")

    def synthesize(self, text: str) -> None:
        """Синтез речи"""
        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=self.config.speaker,
                sample_rate=self.config.sample_rate
                #put_accent=self.config.put_accent,
                #put_yo=self.config.put_yo
            )
            self.audio_queue.put(audio.numpy())
        except Exception as e:
            logging.error(f"TTS Error: {str(e)}")

    def _playback_worker(self):
        """Рабочий поток для воспроизведения"""
        while self.playback_active:
            try:
                audio = self.audio_queue.get(timeout=0.5)
                if isinstance(audio, np.ndarray):
                    self.stream.write(audio.tobytes())
                else:
                    logging.error("Invalid audio format!")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Playback error: {str(e)}")


class SpeechRecognizer:
    def __init__(self, config: VoiceConfig, model_size: str = 'tiny'):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.listening_paused = False
        self._init_recognition(model_size)
        self._adjust_noise()

    def _init_recognition(self, model_size: str):
        """Инициализация модели распознавания"""
        self.model = sr.Recognizer()

    def _adjust_noise(self):
        """Калибровка фонового шума"""
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
    def pause_listening(self):
        """Приостановка прослушивания"""
        self.listening_paused = True
        
    def resume_listening(self):
        """Возобновление прослушивания"""
        self.listening_paused = False

    def continuous_listen(self, callback):
        def listen_loop():
            with self.mic as source:
                while True:
                    if self.listening_paused:
                        continue  
                    try:                     
                        audio = self.recognizer.listen(source, timeout=3)
                        if self.config.noise_reduction:
                            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
                            reduced_noise = nr.reduce_noise(y=audio_data, sr=self.config.sample_rate, prop_decrease=0.8, stationary=False)
                            audio.frame_data = reduced_noise.astype(np.int16).tobytes()
                        text = self.recognizer.recognize_google(audio, language=self.config.language)
                        if text and text != self.last_text:  # Проверка на повторения
                            self.last_text = text
                            callback(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        logging.warning("Не удалось распознать речь")
                    except Exception as e:
                        logging.error(f"Ошибка ASR: {str(e)}")

        self.executor.submit(listen_loop)


    def shutdown(self):
        """Завершение работы распознавателя"""
        self.executor.shutdown(wait=False)


class AudioManager:
    def __init__(self, config: VoiceConfig):
        self.synthesizer = SpeechSynthesizer(config)
        self.recognizer = SpeechRecognizer(config)
        self.command_queue = queue.Queue()
        self.config = config
        self._init_handlers()

    def _init_handlers(self):
        """Инициализация обработчиков аудиособытий"""
        self.recognizer.continuous_listen(self._process_voice_command)

    def _process_voice_command(self, text: str):
        """Обработка распознанной команды"""
        self.command_queue.put(text.strip().lower())

    def get_command(self) -> Optional[str]:
        """Получение последней команды"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def speak(self, text: str) -> None:
        """Приоритет пользовательских звуков"""
        sound_key = text.strip().lower()
        if sound_key in self.synthesizer.soundbank:
            self.synthesizer.play_sound(sound_key)
        else:
            self.recognizer.pause_listening()  # Отключаем микрофон
            try:
                self.synthesizer.synthesize(text)
            finally:
                self.recognizer.resume_listening()  # Включаем микрофон


if __name__ == "__main__":
    # Инициализация AudioManager с конфигурацией по умолчанию
    audio_manager = AudioManager(DEFAULT_VOICE_CONFIG)

    print("Говорите что-нибудь! Для выхода скажите 'стоп'.")

    try:
        while True:
            # Получение команды из голосового ввода
            command = audio_manager.get_command()

            if command:
                print(f"Вы сказали: {command}")

                # Если пользователь сказал "стоп", завершаем программу
                if command.lower() in ["стоп", "stop"]:
                    print("Завершение работы...")
                    break

                # Воспроизведение ответа
                response = f"Вы сказали: {command}"
                audio_manager.speak(response)

    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем.")
    finally:
        # Очистка ресурсов
        audio_manager.recognizer.shutdown()