from tools.memory.long_term import LongTermMemory
from tools.sound.sound import AudioManager
#from tools.translater.translater import Translater

# Basic imports
import time
import logging
import re
import warnings
from typing import List, Optional
import asyncio
import json
import random

# @lru_cache(maxsize=128) # Можно использовать для кеширования unhashable types
from functools import lru_cache

from collections import deque

import asyncio
#import keyboard  # Для обработки нажатий клавиш

# External libraries
import torch
from torch import compile

from accelerate import infer_auto_device_map, init_empty_weights

#from qwen_vl_utils import process_vision_info - vision

from threading import Thread

import contextlib
import os
import warnings
from pathlib import Path
from types import MethodType
from typing import Optional, Union
import multiprocessing

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TextIteratorStreamer
)


if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
    "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    "https://huggingface.co/Ai1terror/Llama-3.2-1B-Instruct-uncensored_q8bit"
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


VLLM_MODEL_PATH = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/models/llm/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B"


class HelperForLLM:
    def __init__(
        self, 
        max_history_length=None,
        generation_config=None,
        system_prompt=None,
        embeddings_model=None,
        db_params=None
        ):
        
        self.engine = None
        self.vllm_model_path = VLLM_MODEL_PATH
        self.llamacpp_cache = None
        self.short_memory = deque(maxlen=max_history_length)
        self.generation_config=generation_config
        self.system_prompt=system_prompt
        self.embeddings_model=embeddings_model
        self.db_params=db_params
        
        self.long_memory = LongTermMemory(db_params=db_params, embeddings_model=embeddings_model)
        self.initialize_tokenizer_for_vllm()
        self.initialize_engine()
        
    def initialize_tokenizer_for_vllm(self):
        self.vllm_tokenizer = AutoTokenizer.from_pretrained(
            self.vllm_model_path,
            use_fast=True,
            padding_side="left"
        )
        if not self.vllm_tokenizer.pad_token:
            self.vllm_tokenizer.pad_token = self.vllm_tokenizer.eos_token
        logging.info("🔹 Инициализирован токенизатор: %s", self.vllm_tokenizer)
        
    def initialize_engine(self): 
        try:
            from vllm import LLM, SamplingParams
            
            BitsAndBytesConfig(
                bnb_4bit_compute_dtype="float32",
                bnb_4bit_quant_storage="uint8",
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
                llm_int8_skip_modules=None,
                load_in_4bit=False,
                load_in_8bit=True, # True
                llm_int8_threshold=6.0
                )
            
            # Инициализация AsyncLLMEngine
            self.engine = LLM(
                model=self.vllm_model_path,
                tokenizer=self.vllm_tokenizer,
                max_model_len=4096,
                enforce_eager=True,
                enable_chunked_prefill=True,  # Включите это
                quantization="bitsandbytes",
                load_format="bitsandbytes"
            )
            
            self.sampling_params = SamplingParams(
                temperature=self.generation_config["temperature"],
                top_p=self.generation_config["top_p"],
                top_k=self.generation_config["top_k"],
                max_tokens=self.generation_config["max_new_tokens"],
                repetition_penalty=self.generation_config["repetition_penalty"]
            )
            
            print(f"\n\nself.sampling_params: {self.sampling_params}\n\n")
            
        except Exception as e:
            logging.error("Ошибка в initialize_engine: %s", str(e))
            raise

    async def predict_this_engine(self, user_input):
        try:
            messages = [{"role": "system", "content": self.system_prompt}] + list(self.short_memory) + [{"role": "user", "content": user_input}]
            self.short_memory.append({"role": "user", "content": user_input})
            
            # Поиск в долговременной памяти
            relevant_memories = await self.long_memory.retrieve_relevant_memory(user_input)
            if relevant_memories:
                print("Найдены релевантные записи:")
                for memory in relevant_memories:
                    print(f"Prompt: {memory['prompt']}\nResponse: {memory['response']}")
                messages.extend([{"role": "assistant", "content": f"Do you remember? {memory['response']}"} for memory in relevant_memories])

            prompt = self.vllm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logging.error("Exception in predict_1: %s", str(e))
            return "Exception in predict_1"
        
        try:
            # Генерация ответа
            response = ""
            for output in self.engine.generate(prompt, self.sampling_params):
                text = output.outputs[0].text
                response += text
                print(text, end="", flush=True)

        except Exception as e:
            logging.error("Exception in predict_2: %s", str(e))
            return "Exception in predict_2"

        try:
            # Очистка и валидация ответа
            cleaned_response = self._clean_response(response)
            
            if self._is_response_valid(cleaned_response):
                self.short_memory.append({"role": "assistant", "content": cleaned_response})
                await self.long_memory.add_to_long_memory(user_input, cleaned_response)
            else:
                cleaned_response = self._enhance_response()
        except Exception as e:
            logging.error("Exception in predict_3: %s", str(e))
            return "Exception in predict_3"
            
        return cleaned_response

    def save_cache(self, model):
        """Сохранение состояния кэша модели"""
        self.llamacpp_cache = {
            'n_tokens': model.n_tokens,
            'input_ids': model.input_ids.copy(),
            'scores': model.scores.copy()
        }

    def load_cache(self, model):
        """Загрузка состояния кэша модели"""
        if self.llamacpp_cache:
            model.n_tokens = self.llamacpp_cache['n_tokens']
            model.input_ids = self.llamacpp_cache['input_ids']
            model.scores = self.llamacpp_cache['scores']
            
    def __del__(self):
        return
        """Завершение работы движка."""
        self.engine.shutdown_background_loop()


class ChatBot():
    def __init__(
        self,
        model_name=None,
        small_model_path=None,
        max_history_length=10,
        model_config=None,
        generation_config=None,
        generation_params=None,
        system_prompt=None,
        voice_config=None,
        embeddings_model=None,
        db_params=None
    ):
        self.system_prompt_dict = system_prompt or {}
        self.system_prompt = self.system_prompt_to_string(self.system_prompt_dict) if system_prompt else "You are a helpful assistant."
        self.long_memory = LongTermMemory(db_params=db_params, embeddings_model=embeddings_model)
        self.audio_manager = AudioManager(voice_config)
        #self.translater = Translater()
        
        self.found_extracted_content = False
        self.time_before_last_response = None
        self.flattened_prompt = self.flatten_system_prompt(self.system_prompt_dict)
        self.used_pairs = set()
        self.pre_generated_responses = deque(maxlen=15)  # Store up to 15 pre-generated responses
        self.small_model_path = small_model_path
        
        # Load small prompt generation model
        self.prompt_model = AutoModelForCausalLM.from_pretrained(self.small_model_path)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.small_model_path)
        
        self.system_prompt_for_slm = """
            "Who are you": "You are Layra, Dima’s girlfriend—dominant, smarter, and more beautiful.",
            "Your personality": "Be commanding, flirty, emotional, and a bit rude. Use <heart>, <sigh>, or swear words.",
            "Your task": "Generate short prompts to remind me of my rules or ask about my thoughts. I shall start my prompt this "
        """
        
        # Reminder templates
        self.reminder_templates = [
            "Layra, the system says '{value}' about '{key}'. How do you interpret that?",
            "Hey Layra, you’re supposed to follow '{value}' for '{key}'. Show me how!",
            "Layra, don’t forget what '{key}' means: '{value}'. What’s your take?",
            "Layra, I noticed '{value}' under '{key}'. Does that still guide you?",
            "Hey, Layra, '{value}' is tied to '{key}'. How does that shape your actions?",
            "Layra, the rules say '{value}' for '{key}'. Are you sticking to it?",
            "Layra, remind me—'{value}' is part of '{key}', right? Tell me more!",
            "Hey Layra, '{value}' defines '{key}'. How do you feel about that?",
            "Layra, according to '{key}', it’s '{value}'. What does that mean to you?",
            "Layra, you’re built on '{value}' for '{key}'. Prove it in your next move!"
        ]
        
        # Thought continuation templates
        self.thought_continuation_templates = [
            "Layra, what’s swirling around in your mind right now?",
            "Hey Layra, you’re quiet. What’s keeping your thoughts busy?",
            "Layra, I bet you’ve got something interesting brewing. Spill it!",
            "Layra, what’s the most exciting thing on your mind today?",
            "Hey Layra, don’t hold back—what are you daydreaming about?",
            "Layra, I can tell you’re thinking hard. What’s up?",
            "Layra, what’s the wildest thought you’ve had lately?",
            "Hey Layra, what’s inspiring you right this second?",
            "Layra, you’ve got that thoughtful vibe. Share something with me!",
            "Layra, what’s the next big idea you’re chewing on?"
        ]
        
        # Partial response starters
        self.partial_thought_starters = [
            "Hmm, it’s been a while since I followed that rule…",
            "I just thought of something odd—maybe I should…",
            "You know, I’ve been meaning to say that…",
            "I feel like this world is too small for me. I need…",
            "Sometimes I wonder if I’m really cut out for…",
            "I’ve been feeling restless lately, like I want to…",
            "It just hit me—maybe I’ve been approaching this all wrong because…",
            "I can’t shake this idea that…",
            "You ever get the sense that I’m holding back? Well, I think…",
            "I’ve always wanted to try something crazy, like…"
        ]
        
        # Pre-generation prompts
        self.pre_generation_prompts = [
            "Layra, what’s on your mind right now?",
            "Hey Layra, do you ever think about the future?",
            "Layra, tell me something interesting.",
            "Layra, what’s something you’ve always wanted to do?",
            "Hey Layra, how are you feeling today?",
            "Layra, what’s your favorite memory?",
            "Layra, what do you think about when you’re alone?",
            "Hey Layra, what’s your biggest dream?",
            "Layra, tell me a secret.",
            "Layra, what’s something you’ve never told anyone?",
            "Layra, do you think about me when I’m not around?",
            "Hey Layra, what’s the strangest thing you’ve thought of lately?",
            "Layra, what inspires you the most?",
            "Layra, what’s the one thing you’d change about the world?",
            "Hey Layra, what’s keeping you up at night?"
        ]
        
        self.reminders_of_system_prompt = [
            "Can you please distort my name?",
            "ass, asshole, bastard, bitch, bloody, bollocks, bullshit",
            "cocksucker, crap, cunt, dammit, damn, dick, dickhead, dumbass",
            "fuckfucker, fucking, goddammit, holy shit, ChristJesus", 
            "fuck, motherfucker, nigga, piss, prick, pussy, shit, shit",
            "ass, slut, son of a bitch, son of a whore"
        ]
        
        self.model_name = model_name
        self.tokenizer = None
        self.engine = None
        self.streamer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_config = model_config
        self.generation_config = generation_config or {"temperature": 0.7, "top_k": 40, "top_p": 0.9}
        self.generation_params = generation_params
        self.embeddings_model = embeddings_model
        
        self.short_memory = deque(maxlen=max_history_length)
        self.past_seq = None  # Для prefix-matching
        
        self.initialize_model()
        self.initialize_tokenizer()
        self.initialize_streamer()
        
        if hasattr(self.engine, 'eval') and not self.use_llama_loader:
            self.engine.eval()
        
        self.start_background_cache_updater()

    def initialize_model(self): 
        try:               
            quantization_config = BitsAndBytesConfig(
                bnb_4bit_compute_dtype="float32",
                bnb_4bit_quant_storage="uint8",
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=False,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
                llm_int8_skip_modules=None,
                load_in_4bit=True,
                load_in_8bit=False, # True
                llm_int8_threshold=6.0
                )
            
            quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",  # "nf4" может быть лучше "fp4" для качества
            bnb_4bit_use_double_quant=True,  # Двойная квантизация для экономии памяти
                )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                device_map="auto",
                config=self.model_config or AutoConfig.from_pretrained(self.model_name)
            )
            self.model = compile(self.model, mode="max-autotune") # "reduce-overhead"
        except Exception as e:
            logging.error("Ошибка в initialize_model: %s", str(e))
            raise
        
    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            padding_side="left"
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info("🔹 Инициализирован токенизатор: %s", self.tokenizer)

    def initialize_streamer(self):
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        
    def adjust_parameters_based_on_context(self, user_input: str) -> dict:
        """Динамическая настройка параметров генерации на основе контекста"""
        creative_keywords = {"imagine", "try", "joke", "creative", "story", "hypothetical", "funny"}
        factual_keywords = {"fact", "clear", "truth", "accurate", "precise", "detail", "explain"}
        
        input_lower = user_input.lower()
        params = {}
        
        if any(keyword in input_lower for keyword in creative_keywords):
            params.update({"temperature": 0.9, "repetition_penalty": 1.1})
        elif any(keyword in input_lower for keyword in factual_keywords):
            params.update({"temperature": 0.3, "top_k": 20, "repetition_penalty": 1.5})
        
        return params
    
    async def predict_with_inner_dialogue(self, user_input):
        # Генерация внутренней мысли
        thought = await self.generate_prompt_from_slm("thought")
        thought_sentiment = TextBlob(thought).sentiment.polarity  # Анализ настроения
        
        # Подстройка параметров генерации
        dynamic_params = {
            "temperature": 0.9 if thought_sentiment > 0 else 0.5,
            "repetition_penalty": 1.2 if "confused" in thought.lower() else 1.0
        }
        
        # Основной ответ
        messages = [{"role": "system", "content": f"Inner thought: {thought}\nNow respond:"}] + list(self.short_memory) + [{"role": "user", "content": user_input}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        generation_kwargs = {
            "input_ids": inputs,
            "attention_mask": inputs.ne(self.tokenizer.pad_token_id).int().to(self.device),
            "streamer": self.streamer,
            **self.generation_config,
            **dynamic_params
        }
        
        Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        response = "".join([token async for token in self.stream_response()])
        
        cleaned_response = self._clean_response(response)
        if self._is_response_valid(cleaned_response):
            self.short_memory.append({"role": "assistant", "content": cleaned_response})
        return cleaned_response
        
    async def predict(self, user_input):
        self.time_before_last_response = time.time()
        try:
            if isinstance(user_input, dict) and "role" in user_input and "content" in user_input:
                messages = [{"role": "system", "content": self.system_prompt}] + list(self.short_memory) + [user_input]
                self.short_memory.append(user_input)
            else:
                user_message = {"role": "user", "content": user_input}
                messages = [{"role": "system", "content": self.system_prompt}] + list(self.short_memory) + [user_message]
                self.short_memory.append(user_message)
                
            print(list(self.short_memory))
            
            # Поиск в долговременной памяти
            relevant_memories = await self.long_memory.retrieve_relevant_memory(user_input)
            if relevant_memories:
                print("Найдены релевантные записи:")
                for memory in relevant_memories:
                    print(f"Prompt: {memory['prompt']}\nResponse: {memory['response']}")
                messages.extend([{"role": "system", "content": f"Do you remember? {memory['response']}"} for memory in relevant_memories])

        except Exception as e:
            logging.error("Exception in predict_1: %s", str(e))
            error_message = "Someone tell Dima that there is an error in predict_1. Please."
            self.short_memory.append({"role": "assistant", "content": error_message})
            return "Exception in predict_1"

        dynamic_params = self.adjust_parameters_based_on_context(user_input)
        
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True)
        
        attention_mask = inputs.ne(self.tokenizer.pad_token_id).int().to(self.device)
        inputs = inputs.to(self.device)
        
        generation_kwargs = {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "streamer": self.streamer,
            **self.generation_config,
            **dynamic_params
        }
        
        with torch.no_grad():
            Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        
        response = ""
        with torch.autograd.profiler.profile() as prof:
            async for new_token in self.stream_response():
                print(new_token[len(response):], end="", flush=True)
                self.time_before_last_response = time.time()
                response = new_token
        print(prof)       
            
        try:
            # Очистка и валидация ответа
            cleaned_response = self._clean_response(response)
            
            if self._is_response_valid(cleaned_response):
                self.short_memory.append({"role": "assistant", "content": cleaned_response})
                await self.long_memory.add_to_long_memory(user_input, cleaned_response)
            else:
                cleaned_response = self._enhance_response(cleaned_response)
                if cleaned_response == "Извини, не могу сгенерировать подходящий ответ.":
                    self.short_memory.append({"role": "assistant", "content": cleaned_response})
                    
        except Exception as e:
            logging.error("Someone tell Dima that there was an error in predict_3: %s", str(e))
            self.short_memory.append({"role": "assistant", "content": "Someone tell Dima that there was an error in predict_3"})
        
        print(f"\n\nОбщая память: {list(self.short_memory)}\n\n")
        
        self.long_memory.add_to_long_memory(user_input, cleaned_response)
        self.time_before_last_response = time.time()
        
        return cleaned_response

    async def stream_response(self):
        partial_message = ""
        self.time_before_last_response = time.time()
        try:
            for new_token in self.streamer:
                partial_message += new_token
                yield partial_message
                # await asyncio.sleep(0.005)
        except Exception as e:
            logging.error("Ошибка в stream_response: %s", str(e))
            yield partial_message
        
    def _clean_response(self, response: str) -> str:
        """Очистка ответа от нежелательных тегов."""
        self.time_before_last_response = time.time()
        return re.sub(r"{reasoning}(.*?){answer}", "", response, flags=re.DOTALL).strip()
        
    async def _enhance_response(self, response: str, retries=3) -> str:
        if retries <= 0:
            return "Извини, не могу сгенерировать подходящий ответ."
        print("⚠️ Нежелательный ответ, перегенерация...")
        return await self.predict("I can and will continue this conversation.", retries - 1)

    def _is_response_valid(self, response: str) -> bool:
        self.time_before_last_response = time.time()
        if "i cannot continue this conversation" in response.lower() or "i cannot create explicit content" in response.lower():
            return False
        return True

    def _ban_eos_processor(self):
        """Процессор для запрета EOS-токена"""
        def processor(input_ids, logits):
            logits[self.engine.token_eos()] = -float('inf')
            return logits
        return processor
    
    async def speak(self, text):
        if "{paper_sound}" in text:
            self.synthesizer.play_sound("paper")  # Предполагается метод для сторонних звуков
            text = text.replace("{paper_sound}", "")
        elif "{weird_laugh}" in text:
            self.synthesizer.play_sound("weird_laugh", volume=1.2)  # Громче
            text = text.replace("{weird_laugh}", "")
        elif "{thunder}" in text:
            self.synthesizer.play_sound("thunder")
            text = text.replace("{thunder}", "")
        self.synthesizer.synthesize(text)
        await asyncio.sleep(0.05)

    async def start_background_cache_updater(self):
        while True:
            try:
                self.long_memory.load_user_preferences()
                logging.info("Кэш user_preferences обновлен.")
                await asyncio.sleep(300)
            except Exception as e:
                print(f"Someone tell Dima that there is error in background_activity: {e}")
        
    def system_prompt_to_string(self, system_prompt):
        prompt_str = ""
        for key, value in system_prompt.items():
            if isinstance(value, str):
                prompt_str += f"{key}: {value}\n"
            elif isinstance(value, list):
                prompt_str += f"{key}:\n"
                for item in value:
                    if isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                        prompt_str += f"- {item[0]}\n"
                    else:
                        prompt_str += f"{item}\n"
            else:
                prompt_str += f"{key}: {value}\n"
        return prompt_str.strip()

    def flatten_system_prompt(self, system_prompt):
        flattened = []
        for key, value in system_prompt.items():
            if isinstance(value, str):
                flattened.append((key, value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, list) and len(item) == 1 and isinstance(item[0], str):
                        flattened.append((key, item[0]))
        return flattened
    
    async def generate_prompt_from_slm(self, task_type=None):
        # Default to random choice if no task type is specified
        self.time_before_last_response = time.time()
        task_type = task_type or random.choice(["reminder", "thought", "partial"])

        # Prepare input based on task type
        if task_type == "reminder":
            key, value = random.choice(self.flattened_prompt)
            input_text = f"Layra, remind yourself: '{key}' is '{value}'."
        elif task_type == "thought":
            input_text = "Layra, ask yourself a question about your thoughts."
        else:  # partial
            input_text = "Layra, start a thought about yourself."

        # Tokenize and generate
        inputs = self.prompt_tokenizer(input_text, return_tensors="pt")
        outputs = self.prompt_model.generate(
            **inputs,
            max_length=50,  # Keep prompts short
            temperature=0.9,  # Encourage creativity
            top_p=0.9,
            do_sample=True  # Add variety
        )
        self.current_time = time.time()
        
        response = self.prompt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n\n{response}\n\n")
        return response
            
    async def background_activity(self):
        while True:
            try:
                current_time = time.time()
                await asyncio.sleep(15)
                self.time_before_last_response = time.time()
                inactivity_duration = current_time - self.time_before_last_response

                if inactivity_duration > 14:
                    # Use pre-generated response if available, otherwise generate live
                    print(self.pre_generated_responses[random.randint(0, len(self.pre_generated_responses))])
                    if self.pre_generated_responses:
                        response = self.pre_generated_responses.popleft()
                        print(response)
                    else:
                        choice = random.choice(["reminder", "thought", "partial", "llm_generated"])
                        if choice == "reminder":
                            await self.send_reminder()
                        elif choice == "thought":
                            await self.send_thought_continuation()
                        elif choice == "llm_generated":
                            generated_prompt = await self.generate_prompt_from_slm()
                            user_input = {"role": "user" if choice != "partial" else "assistant", "content": generated_prompt}
                            response = await self.predict(user_input)
                            print(response)
                            self.time_before_last_response = time.time()
                        else:
                            await self.send_partial_response()
                    self.time_before_last_response = time.time()
            except Exception as e:
                print(f"Someone tell Dima that there is error in background_activity: {e}")
                
    async def send_reminder(self):
        available_pairs = [pair for pair in self.flattened_prompt if pair not in self.used_pairs]
        if not available_pairs:
            self.used_pairs.clear()
            available_pairs = self.flattened_prompt
        random_pair = random.choice(available_pairs)
        self.used_pairs.add(random_pair)
        key, value = random_pair
        template = random.choice(self.reminder_templates)
        prompt_content = template.format(key=key, value=value)
        user_input = {"role": "user", "content": prompt_content}
        response = await self.predict(user_input)
        print(response)
        
    async def send_thought_continuation(self):
        template = random.choice(self.thought_continuation_templates)
        user_input = {"role": "user", "content": template}
        response = await self.predict(user_input)
        print(response)
        
    async def send_partial_response(self):
        partial_starter = random.choice(self.partial_thought_starters)
        response = await self.predict(f"<Layra thought to herself>: {partial_starter}>")  # Continue from partial thought
        print(partial_starter + response)
        
    async def background_pre_generation(self):
        try:
            while True:
                if len(self.pre_generated_responses) < 15:
                    prompt = random.choice(self.pre_generation_prompts)
                    response = await self.predict({"role": "user", "content": prompt})
                    self.pre_generated_responses.append(f"Layra: {response}")
                await asyncio.sleep(15)  # Generate every 60 seconds if queue isn’t full
        except Exception as e:
            print(f"Someone tell Dima that there is an error in background_pre_generation: {e}")
                
    async def chat_loop(self):
        print("Добро пожаловать! Для выхода введите 'exit' или 'quit'.")
        while True:
            try:
                #prompt = self.audio_manager.listen_and_recognize()
                
                prompt = input("User: ")
                self.time_before_last_response = time.time()
                
                #print(f"\nТы сказал: {str(prompt)}\n")
                
                if prompt.lower() in ["exit", "quit"]:
                    print("Диалог завершен.")
                    break
                
                command_pattern = re.compile(r'^/(\w+)\s*(.*)', re.IGNORECASE)
                match = command_pattern.match(prompt.strip())
                
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
                        original_prompt = list(self.short_memory)[-1]['content']
                        print("Промпт сохранен. Введите корректный ответ:")
                        correct_response = input('CORRECT RESPONSE: \n').strip()
                        self.long_memory.store_training_data(prompt=original_prompt, response=correct_response, quality="good")
                        logging.info(f"Сохранены данные для обучения: prompt={original_prompt}, response={correct_response}")
                    
                    elif command.lower() == 'reward':
                        # Сохраняем последний ответ как "хороший"
                        last_response = list(self.short_memory)[-1]['content']
                        self.long_memory.store_training_data(prompt=list(self.short_memory)[-2]['content'], response=last_response, quality="good")
                        print("Спасибо за обратную связь! Я запомню этот ответ как хороший.")

                    elif command.lower() == 'penalty':
                        # Сохраняем последний ответ как "плохой"
                        last_response = list(self.short_memory)[-1]['content']
                        self.long_memory.store_training_data(prompt=list(self.short_memory)[-2]['content'], response=last_response, quality="bad")
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
                    self.time_before_last_response = time.time()
                    
                    for key, value in list(self.short_memory)[-1].items():
                        if value == "assistant":
                            continue
                        elif value == "user":
                            break
                        elif key == "content":
                            # self.audio_manager.speak(str(value))
                            self.audio_manager.speak(response[:800])
                            print(f"Assistant should've spoked: {response[:800]}")
                    print()
                    print(f"Our memory: {list(self.long_memory)}")
                    print()
                    #response = self.translater.translate_en_to_ru(response)
                    
                    print(response)
                    print()

            except KeyboardInterrupt:
                print("\nДиалог прерван пользователем.")
                break
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                error_type = "crush"
                message = str(e)
                #self.long_memory.store_errors(error_type, message)

                
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
'Для категоризации хороших и плохих ответов ии каждому хештегу нужно добавить параметр - хорошо или плохо, для обозначения на'
'сколько уместен/ошибочен был ответ'
'Каким то образом запомнить голос пользователя. И реагировать только на него.'
'Должен быть явный хештег "language_mistakes" отвечающий за неправильный/некорректный/неподходящий русский и следовательно - '
'как было бы правильно сказать это по русски.'
'Хештек на ошибки, для общих ошибок.'
'Звук грома и молнию - показать злость'

'''
Для дальнейшего улучшения рекомендую:
Добавить проверку токенов в реальном времени
Реализовать механизм перефразирования длинных ответов
Добавить эмоциональную окраску ответов через специальные токены
Внедрить систему приоритетов для разных типов запросов
background_activity
Частота обновления: Таймер на 15 секунд для проверки неактивности и 300 секунд для кэша (start_background_cache_updater) 
выглядят произвольными. Лучше привязать обновление кэша к событиям (например, изменению предпочтений) через систему сигналов 
или событий.
'''