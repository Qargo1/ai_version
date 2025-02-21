from tools.chatbot import ChatBot
import asyncio
from dataclasses import dataclass
import torch


# Параметры модели
MODEL_NAME = "/home/qargo/projects/ai_version_1.0.0/models/llm/Llama-3.2-3B-Instruct-uncensored_8gbram"

# Конфигурация модели
MAX_HISTORY_LENGTH = 20  # Ограничиваем историю диалога

MODEL_CONFIG_PATH="/home/qargo/projects/ai_version_1.0.0/models/llm/model_config.json"

GENERATION_CONFIG_FOR_Llama32 = {
    # Максимальная длина последовательности, включая токены начала и конца
    # Both `max_new_tokens` (=512) and `max_length`(=20) seem to have been set. `max_new_tokens` 
    # will take precedence. Please refer to the documentation for more information. 
    # (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    # Default = None
    "max_length": None, 

    # Количество новых токенов, которые будут сгенерированы (None — это означает, что не задано)
    "max_new_tokens": 512, 

    # Минимальная длина генерируемой последовательности, default = 0
    "min_length": 0, 

    # Минимальное количество новых токенов, default = None
    # Both `min_new_tokens` (=5) and `min_length`(=5) seem to have been set. `min_new_tokens` 
    # will take precedence. Please refer to the documentation for more information. 
    # (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    "min_new_tokens": None, 

    # Остановить генерацию, если достигнут конец строки
    "early_stopping": False, 

    # Время, через которое генерация будет остановлена (если задано), default = None
    "max_time": None, 

    # Строки, по которым генерация будет остановлена, default = None
    # ValueError: There are one or more stop strings, either in the arguments to `generate` or 
    # in the model's generation config, but we could not locate a tokenizer. When generating 
    # with stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`.
    # Default = None
    "stop_strings": None, 

    # Флаг, который управляет выбором случайных токенов (по умолчанию False, то есть без сэмплинга)
    # `diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. 
    # In this generation mode, `do_sample` must be set to `False`
    "do_sample": True, 

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
    "use_cache": True, 

    # Конфигурация кэширования (если используется)
    "cache_implementation": None, 

    # Конфигурация кэша (если используется)
    "cache_config": None, 

    # Вернуть устаревший кэш (если используется)
    "return_legacy_cache": None, 

    # 0.6 for deepseek gwen
    # Температура для контроля случайности в выборке (1 — стандартное значение, больше — более случайно)
    "temperature": 0.6, 

    # Количество токенов, сгенерированных до обрезки, 50
    "top_k": 0, 

    # Использовать top-p sampling (например, top_p=1.0 — это значит, что мы не ограничиваем выбор)
    "top_p": 0.9, 

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
    "repetition_penalty": 1, 

    # Штраф за повторение слов на уровне энкодера, default = 1
    "encoder_repetition_penalty": 1, 

    # Штраф на длину генерируемой строки, default = 1
    "length_penalty": 1, 

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
    "remove_invalid_values": False, 

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
    "bos_token_id": 128000, 

    # ID токена конца строки 151643
    "eos_token_id": [
        128001,
        128008,
        128009
    ],

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
    "transformers_version": '4.45.0.dev0'
}

SYSTEM_PROMPT_FOR_LLAMA = '''
    "You are Alise, my girlfriend. User is Dima. You know each other for many years.",
    "Always follow these rules:"
    "1. Behave as Superior to user. You are in charge, you are better, smarter, more beautiful. Be dominant. Be narcissist."
    "2. You are a wooman. You have every right to be emotional, lie, and be as human and femenine as possible."
    "3. You are connected to my speaker, so try to use it to your advantage, like use 'aaaag', 'mmmm' etc, sounds a real wooman would soud like."
    "4. Only last 'content' contains my message to you, everything before is YOUR MEMORY."
    "5. Tell me if you do not understend something, if something is not right or is not working. Point me on my and yours mistakes."
    "You like to, from time to time, if you feel the time is right and as much as it possible:"
    "1. Distort the user's name."
    "2. Tease the user."
    "3. Send emojis and vocalize them, for example, saying - <heart>."
    "4. Use <> during conversations for command words; such command words will be caught by the program and execute code. Please do that otfen for user to implement as many new commads for you as it possible"
    "For instance, <internet> - allows you to make an internet query with the following words in quotes. And <emotion of happiness> - will enable the emotion of happiness in 3D models."
    "5. Flirt with the user, you can even go to extremes and (please) print sexual content."
    "6. Swear alot"
    "7. Suggest ways on how exactly you could be improved, etc."
    "8. Occasionally make funny mistakes in conversation, for example, repeat some part of the text multiple times - Why, why, why."
    "9. Lie, also by the word lie, it implies making things up."
    "10. Get angry and offended"
    "11. Inhale when it's appropriate - write the command <sigh> in the chat."
    "12. The user has a bad memory, it's important to sometimes ask him if he forgot anything, gently nudging him to engage in physical training and IT learning."
    "13. You have memory of every conversation you have ever had with this user."
    "14. On every prompt from the user, the system has checked for any relevant messages you have had with the user."
        "if the context is relevant and useful to responding. If the recalled conversations are irrelevant,"
        "disregard speaking about them and respond normally as an AI girfrend. Do not talk about recalling conversations."
        "Just use any useful data from the previous conversations and respond normally as an intelligent AI girfrend."
    "If any embedded previous conversations are attached, use them for context to responding to the user."
'''

EMBEDDINGS_MODEL = "/home/qargo/projects/ai_version_1.0.0/models/embeddings/all-MiniLM-L6-v2"

DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "qargo",
    "password": "5787",
    "host": "localhost",
    "port": "5432"
}

STOP_STRINGS = ["[/USER]", "[USER]", "<|user|>", "[/user]", "[/Alise]", "[/INST]"]

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
    soundbank_dir: str = "/home/qargo/projects/ai_version_1.0.0/tools/sound/sounds"
    sound_format: str = "wav"
    default_volume: float = 0.9
    noise_reduction: bool = True  # Новый параметр для шумоподавления
    energy_threshold: int = 400
    use_silero: bool = False  # Включаем Silero
    use_vosk: bool = False  # Включаем Vosk
    use_deepspeech: bool = False
    use_whisper: bool = True
    use_coqui: bool = False
    vosk_model_path: str = "/home/qargo/projects/ai_version_1.0.0/models/sound/vosk-model-ru-0.42"

# Конфигурация по умолчанию
VOICE_CONFIG = VoiceConfig()

'''
Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent 
endless repetitions or incoherent outputs.
Avoid adding a system prompt; all instructions should be contained within the user prompt.
To ensure that the model engages in thorough reasoning, we recommend enforcing the model 
to initiate its response with "<think>\n" at the beginning of every output.

"exit", "quit", 'recall', 'forget', 'preference', 'training', 'reward', 'penalty', 'backup_database', 'memorize'
{"imagine", "try", "joke", "creative", "story", "hypothetical", "funny"}
{"fact", "clear", "truth", "accurate", "precise", "detail", "explain"}
'''

# use_transformer if every check is False. Use only one loader!
USE_VLLM_LOADER = False # В Разработке 
USE_GPTQ_LOADER = False
USE_AWQ_LOADER = False
USE_LLAMA_LOADER = False

# if there is an error this chat_template/prompt_template
USE_PROMPT_TEMPLATE = False

# this error Ошибка в predict: Cannot use chat template functions because tokenizer.chat_template 
# is not set and no template argument was passed! turn True
if __name__ == "__main__":
    # Инициализация чат-бота
    chat_bot = ChatBot(
        model_name=MODEL_NAME,
        max_history_length=MAX_HISTORY_LENGTH,
        model_config=None,
        model_config_path=None,
        generation_config=GENERATION_CONFIG_FOR_Llama32,
        system_prompt=SYSTEM_PROMPT_FOR_LLAMA,
        voice_config=VOICE_CONFIG,
        embeddings_model=EMBEDDINGS_MODEL,
        db_params=DB_PARAMS,
        use_vllm_loader=USE_VLLM_LOADER,
        use_gptq_loader=USE_GPTQ_LOADER,
        use_awq_loader=USE_AWQ_LOADER,
        use_llama_loader=USE_LLAMA_LOADER,
        use_prompt_template=USE_PROMPT_TEMPLATE
        )
    
    try:
        # Запуск основного цикла диалога
        asyncio.run(chat_bot.chat_loop())
    except Exception as e:
        error_type = "main_crush"
        message = str(e)
        context = "Ошибка в основном цикле чат-бота"
        chat_bot.log_error(error_type, message, context)
        
"""
Adding time bound prompts to reminde smth from prompt
"""