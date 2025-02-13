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

from silero import silero_stt
import zipfile
from glob import glob
import asyncio
from vosk import Model, KaldiRecognizer
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import signal
import keyboard
import json
#import deepspeech
from whisper import load_model as load_whisper
#from coqui_stt import Model as CoquiModel


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(f'\n{sr.Microphone.list_microphone_names()}\n')

@dataclass
class VoiceConfig:
    speaker: str = 'kseniya'
    model_id: str = 'v4_ru'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_rate: int = 8000 #16000
    language: str = 'ru'
    put_accent: bool = True
    put_yo: bool = True
    volume: float = 0.9
    speech_rate: int = 160
    soundbank_dir: str = "sounds"
    sound_format: str = "wav"
    default_volume: float = 0.9
    noise_reduction: bool = True  # Новый параметр для шумоподавления
    energy_threshold: int = 400
    use_silero: bool = False  # Включаем Silero
    use_vosk: bool = False  # Включаем Vosk
    use_deepspeech: bool = False
    use_whisper: bool = True
    use_coqui: bool = False
    vosk_model_path: str = "models/sound/vosk-model-ru-0.42" #качаем отдельно с https://alphacephei.com/vosk/models

# Конфигурация по умолчанию
DEFAULT_VOICE_CONFIG = VoiceConfig()

def handle_exception(logger, message, exception):
    logger.error(f"{message}: {str(exception)}")
    raise

class SpeechSynthesizer:
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.playback_active = True
        self.soundbank = self._load_soundbank(config.soundbank_dir, config.sound_format)
        self._init_tts()
        self._init_playback()
        logging.info("Initializing SpeechSynthesizer...")

    def _init_tts(self):
        """Инициализация TTS"""
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=self.config.language,
                speaker=self.config.model_id
            )
            self.model.to(torch.device(self.config.device))

            try:
                example_text = "Привет, мир!"
                traced_model = torch.jit.trace(self.model.apply_tts, example_text)
                self.model = traced_model
                logging.info("TTS модель успешно скомпилирована с JIT (trace).")
            except Exception as e:
                logging.error(f"Ошибка при компиляции модели с JIT: {str(e)}")
                self.model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=self.config.language,
                    speaker=self.config.model_id
                    )
                self.model.to(torch.device(self.config.device))

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
            logging.info(f"Loading sounds from directory: {sound_dir}")
            for file in os.listdir(sound_dir):
                if file.endswith(f".{fmt}"):
                    key = os.path.splitext(file)[0].lower()
                    try:
                        with wave.open(os.path.join(sound_dir, file), 'rb') as wav:
                            # Проверяем параметры файла
                            sample_width = wav.getsampwidth()
                            channels = wav.getnchannels()
                            sample_rate = wav.getframerate()
                            logging.debug(f"File: {file}, Channels: {channels}, Sample Rate: {sample_rate}, Width: {sample_width}")

                            if channels != 1 or sample_rate != self.config.sample_rate:
                                logging.warning(f"Skipping {file}: unsupported format.")
                                continue

                            audio_data = np.frombuffer(wav.readframes(-1), dtype=np.int16)
                            soundbank[key] = audio_data.astype(np.float32) / 32768.0
                            logging.info(f"Loaded sound: {key}")
                    except Exception as e:
                        logging.error(f"Error loading {file}: {str(e)}")
        else:
            logging.warning(f"Sound directory not found: {sound_dir}")
        return soundbank
    
    def test_synthesize(self, text: str):
        """Test TTS without adding to the queue."""
        try:
            with torch.no_grad():
                audio = self.model.apply_tts(
                    text=text,
                    speaker=self.config.speaker,
                    sample_rate=self.config.sample_rate
                )
                logging.debug(f"Speaker: {self.config.speaker}, Sample Rate: {self.config.sample_rate}")
            return audio.numpy()
        except Exception as e:
            handle_exception(logging, "TTS Test Error", e)
            
    def test_speech_to_text(self, audio_file: str):
        """Тестирует распознавание речи из файла"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_vosk(audio)
            logging.info(f"Recognized text: {text}")
            return text
        except Exception as e:
            handle_exception(logging, "STT Test Error", e)
    
    def close(self):
        """Explicitly release resources."""
        self.playback_active = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

    def play_sound(self, sound_name: str) -> None:
        """Воспроизведение пользовательского звука"""
        audio = self.soundbank.get(sound_name.lower())
        if audio is not None:
            self.audio_queue.put(audio)
        else:
            logging.warning(f"Sound {sound_name} not found!")

    def synthesize(self, text: str) -> None:
        try:
            with torch.no_grad():
                audio = self.model.apply_tts(
                    text=text,
                    speaker=self.config.speaker,
                    sample_rate=self.config.sample_rate
                )
            logging.info(f"Synthesized audio for text: '{text}'")
            self.audio_queue.put(audio.numpy())
        except Exception as e:
            handle_exception(logging, "TTS Error", e)
            
    def _playback_worker(self):
        """Рабочий поток для воспроизведения"""
        while self.playback_active:
            try:
                audio = self.audio_queue.get(timeout=0.5)
                logging.info(f"Audio retrieved from queue. Length: {len(audio)}")
                if isinstance(audio, np.ndarray):
                    self.stream.write(audio.tobytes())
                else:
                    logging.error("Invalid audio format!")
            except queue.Empty:
                logging.debug("Queue is empty. Waiting for audio...")
                continue
            except Exception as e:
                logging.error(f"Playback error: {str(e)}")


class SpeechRecognizer:
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.listening_paused = False
        self.model = None
        self._init_recognition()
        logging.info("Initializing SpeechRecognizer...")

    def _init_recognition(self):
        if self.config.use_whisper:
            logging.info("Используется Whisper")
            self.model = load_whisper("medium")
        elif self.config.use_silero and self.config.device == 'cuda':
            logging.info("Используется Silero STT на GPU")
            self.model, self.decoder, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_stt',
                language='en', 
                device=torch.device(self.config.device))
            (self.read_batch, self.split_into_batches,
            self.read_audio, self.prepare_model_input) = self.utils  # see function signature for details
        elif self.config.use_vosk:
            logging.info("Используется Vosk STT")
            self.model = Model(self.config.vosk_model_path)
            self.recognizer_vosk = KaldiRecognizer(self.model, self.config.sample_rate)
        elif self.config.use_deepspeech:
            logging.info("Используется DeepSpeech")
            self.model = deepspeech.Model(self.config.deepspeech_model_path)
        elif self.config.use_coqui:
            logging.info("Используется Coqui STT")
            self.model = CoquiModel(self.config.coqui_model_path)
        
    def __del__(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'pyaudio_instance'):
            self.pyaudio_instance.terminate()
            
    def close(self):
        """Explicitly release resources."""
        if hasattr(self, 'mic') and self.mic:
            self.mic.__exit__(None, None, None)
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def test_recognition(self, audio_file: str):
        """Test STT with a pre-recorded audio file."""
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_vosk(audio)
            return text
        except Exception as e:
            handle_exception(logging, "STT Test Error", e)
            
    def _process_audio(self, audio):
        if self.config.use_vosk:
            return self.recognizer_vosk.AcceptWaveform(audio.get_wav_data()) 
        elif self.config.use_whisper:
            return self.model.transcribe(audio.get_wav_data())['text'].strip()

    async def listen_loop(self, callback):
        with self.mic as source:
            while True:
                try:
                    #logging.info("Жду аудио...")
                    self.recognizer.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening
                    audio = self.recognizer.listen(source)#, timeout=5, phrase_time_limit=100)

                    if self.config.use_whisper:
                        try:
                            text = self.recognizer.recognize_whisper(audio, language="russian")
                            #logging.info(f"Текст передан{text}")
                        except sr.UnknownValueError:
                            print("Whisper could not understand audio")
                        except sr.RequestError as e:
                            print(f"Could not request results from Whisper; {e}")
                    elif self.config.use_vosk:
                        if self.recognizer_vosk.AcceptWaveform(audio.get_wav_data()):
                            result = json.loads(self.recognizer_vosk.Result())
                            text = result.get("text", "").strip()
                        else:
                            text = None
                    elif self.config.use_silero:
                        batches = self.split_into_batches(audio.get_wav_data(), batch_size=10)
                        input = self.prepare_model_input(self.read_batch(batches[0]),
                            device=device)
                        text = self.model(input)
                    elif self.config.use_deepspeech:
                        text = self.model.stt(audio.get_wav_data())
                    elif self.config.use_coqui:
                        text = self.model.stt(audio.get_wav_data())
                    else:
                        text = self._process_audio(audio)
                    
                    if text:
                        callback(text)
                except KeyboardInterrupt:
                    logging.info("Listening interrupted by user. Exiting gracefully...")
                    break
                except Exception as e:
                    handle_exception(logging, "ASR Error", e)

    def continuous_listen(self, callback):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.create_task(self.listen_loop(callback))
        if not loop.is_running():
            loop.run_until_complete(asyncio.sleep(0))
            
    def listen_once(self):
        """Однократное прослушивание микрофона"""
        with self.mic as source:
            try:
                logging.info("Жду аудио...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, phrase_time_limit=4)# ,timeout=10)
                
                # Распознавание текста
                if self.config.use_whisper:
                    text = self.recognizer.recognize_whisper(audio, language="russian")
                elif self.config.use_vosk:
                    if self.recognizer_vosk.AcceptWaveform(audio.get_wav_data()):
                        result = json.loads(self.recognizer_vosk.Result())
                        text = result.get("text", "").strip()
                    else:
                        text = None
                else:
                    text = self._process_audio(audio)
                
                return text.strip().lower() if text else None
            except sr.UnknownValueError:
                logging.warning("Could not understand audio.")
                return None
            except sr.RequestError as e:
                logging.error(f"Could not request results; {e}")
                return None
            except Exception as e:
                handle_exception(logging, "ASR Error", e)
                return None

    def shutdown(self):
        self.executor.shutdown(wait=False)


class AudioManager:
    def __init__(self, config: VoiceConfig):
        try:
            self.synthesizer = SpeechSynthesizer(config)
            self.recognizer = SpeechRecognizer(config)
            self.command_queue = queue.SimpleQueue()
            self.config = config
            self.is_listening = False  # Флаг для контроля состояния прослушивания
        except Exception as e:
            logging.error(f"Ошибка инициализации AudioManager: {str(e)}")
            raise
        
    def _init_handlers(self):
        """Инициализация обработчиков аудиособытий"""
        #В качестве callback-функции передается _process_voice_command
        self.recognizer.continuous_listen(self._process_voice_command)
        
    def _process_voice_command(self, text: str):
        """Обработка распознанной команды"""
        self.command_queue.put(text.strip().lower())
    
    def start_listening(self):
        """Запуск прослушивания микрофона"""
        if not self.is_listening:
            self._init_handlers()
            self.is_listening = True
            logging.info("Microphone listening started.")
            
    def process_command(self, text: str):
        """Обработка команды: генерация и воспроизведение ответа"""
        if text in self.synthesizer.soundbank:
            self.synthesizer.play_sound(text)
        else:
            self.synthesizer.synthesize(text)

    def get_command(self) -> Optional[str]:
        """Получение последней команды"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            logging.info("There are no commands in the queue. Waiting for new commands...")
            return None
        
    def shutdown(self):
        """Завершение работы AudioManager"""
        self.recognizer.shutdown()
        self.synthesizer.__del__()  # Очистка ресурсов синтезатора

    def speak(self, text: str) -> None:
        self.play_sound_on_key_press('space', 'beep') #new
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
                
    async def run(self):
        """Основной цикл программы"""
        while True:
            print("Говорите что-нибудь! Для выхода скажите 'стоп'.")
            
            # Шаг 1: Прослушивание микрофона
            command = self.recognizer.listen_once()
            if not command:
                logging.info("No command recognized. Listening again...")
                continue
            
            print(f"Вы сказали: {command}")
            
            # Шаг 2: Проверка на завершение
            if command.lower() in ["стоп", "stop"]:
                print("Завершение работы...")
                break
            
            # Шаг 3: Генерация и воспроизведение ответа
            response = f"Вы сказали: {command}"
            self.process_command(response)
            
            # Ждем завершения воспроизведения
            while not self.synthesizer.audio_queue.empty():
                await asyncio.sleep(0.1)  # Ждем, пока очередь аудио не опустеет


if __name__ == "__main__":
    # Инициализация AudioManager с конфигурацией по умолчанию
    audio_manager = AudioManager(DEFAULT_VOICE_CONFIG)

    try:
        # Запуск асинхронного цикла через asyncio.run()
        asyncio.run(audio_manager.run())
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем.")
    finally:
        # Очистка ресурсов
        logging.info("Shutting down AudioManager...")
        audio_manager.shutdown()
        
        
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

