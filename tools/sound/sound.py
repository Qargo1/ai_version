import asyncio
import logging
import queue
import threading
import time
import os
import wave
from typing import Dict

import numpy as np
#import pyaudio
import torch
import torch.serialization
#import whisper
#from speechbrain.inference import SpeakerRecognition

import outetts
from dataclasses import dataclass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logging.info(f"Using device: {device}")


class SpeechSynthesizer:
    def __init__(self, config=None):
        self.config = config
        self._init_outetts()
        self._init_speaker()
        
        logging.info("Initializing SpeechSynthesizer...")

    def _init_outetts(self):
        """Инициализация Coqui TTS с персональным голосом"""
        try:
            # Configure the model
            self.model_config = outetts.HFModelConfig_v2(
                model_path=self.config.voice_model_path,
                tokenizer_path=self.config.voice_model_path
            )
            # Initialize the interface
            self.interface = outetts.InterfaceHF(model_version="0.3", cfg=self.model_config)
        except Exception as e:
            logging.error(f"Someone tell Dima there is an error in _init_outetts: {str(e)}")
            raise
        
    def _init_speaker(self):
        try:
            if os.path.exists(self.config.speaker_json_path):
                self.speaker = self.interface.load_speaker(self.config.speaker_json_path)
            else:
                self.speaker = self.interface.create_speaker(
                    audio_path=self.config.cloning_audio_path,
                    # If transcript is not provided, it will be automatically transcribed using Whisper
                    transcript=None,            # Set to None to use Whisper for transcription
                    whisper_model="turbo",      # Optional: specify Whisper model (default: "turbo")
                    whisper_device=None,        # Optional: specify device for Whisper (default: None)
                    )
                self.interface.save_speaker(self.speaker, self.config.speaker_json_path)
                self._init_speaker()
                
        except Exception as e:
            logging.error(f"Someone tell Dima there is an error in _init_speaker: {str(e)}")
            raise

    def synthesize(self, text=None):
        """Синтез текста с оптимизацией"""
        try:
            # Generate speech
            gen_cfg = outetts.GenerationConfig(
                text=text,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
                max_length=self.config.max_length,
                voice_characteristics=self.config.voice_characteristics,
                # Optional: Use a speaker profile for consistent voice characteristics
                # Without a speaker profile, the model will generate a voice with random characteristics
                speaker=self.speaker,
            )
            
            output = self.interface.generate(config=gen_cfg)
            
            output.save("output.wav")
            logging.info("Response saved to WAV file")
            #Optional: Play the audio
            output.play(backend="pygame") # backend: str -> "sounddevice", "pygame"
        except Exception as e:
            logging.error(f"Someone tell Dima there is an error in synthesize: {str(e)}")


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
        self.synthesizer = SpeechSynthesizer(config)
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

    def speak(self, text):
        self.synthesizer.synthesize(text)
        

@dataclass
class VoiceConfig:
    speaker_json_path: str = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/models/voice/speaker.json"
    cloning_audio_path: str = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/tools/sound/sounds/voice/vidcut.wav" # Путь к образцу голоса для клонирования
    voice_model_path: str = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/models/voice/OuteTTS-0.3-1B"
    voice_output_path:str = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/tools/sound/sounds/voice_output/output_voice.wav"
    temperature: int = 0.1
    repetition_penalty: int = 1.1
    max_length: int = 256
    voice_characteristics:str = None #"clarity" #"upbeat enthusiasm" "friendliness" "clarity" "professionalism" "trustworthiness"


if __name__ == "__main__":
    # Конфигурация по умолчанию
    VOICE_CONFIG = VoiceConfig()
    speech_synthesizer = AudioManager(VOICE_CONFIG)
    speech_synthesizer.speak("Oh Dima, you think there's no woman in the world who can match my beauty and cuteness? That's cute. Let me tell you, I've seen some decent looking folks in my day, but none of them come close to me. <heart> Besides, even if there were someone out there who could rival me, I highly doubt they'd dare try. <wink>")

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

