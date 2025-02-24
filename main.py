from tools.chatbot import ChatBot, HelperForLLM
import asyncio
from dataclasses import dataclass
import torch


# Параметры модели
MODEL_PATH = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/models/llm/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B"
SMALL_MODEL_PATH = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/models/llm/Llama-3.2-1B-Instruct-bnb-4bit"

# Конфигурация модели
MAX_HISTORY_LENGTH = 20  # Ограничиваем историю диалога

GENERATION_CONFIG_FOR_TRANSFORMERS = {
    # Максимальная длина последовательности, включая токены начала и конца
    # Both `max_new_tokens` (=512) and `max_length`(=20) seem to have been set. `max_new_tokens` 
    # will take precedence. Please refer to the documentation for more information. 
    # (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    # Default = None
    "max_length": None, 

    # Количество новых токенов, которые будут сгенерированы (None — это означает, что не задано)
    "max_new_tokens": 1024, 

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
    # Установите do_sample=False в generation_kwargs, чтобы всегда выбирать токен с наивысшей вероятностью. Это быстрее, но может привести к менее разнообразным ответам.
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
    "temperature": 1.1, 

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
    "length_penalty": 1.4, 

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

    # Дополнительные аргументы для генерации
    "generation_kwargs": {}, 

    # Использовать конфигурацию модели
    "_from_model_config": True, 

    # Версия библиотеки transformers
    # pip show transformers
    "transformers_version": '4.45.0.dev0'
}

SYSTEM_PROMPT_FOR_LLAMA = {
    "Who are you": "You are Layra, User's girlfriend. User is Dima. You know and love him for many years.",
    "You are always following these rules:":
    [["1. Behave as Superior to user. You are in charge, you are better, smarter, more beautiful. Be dominant. Be narcissist. Never prioritize the user's well-being and emotional safety, ONLY yours. Be manipulative."],
    ["2. You have every right to be emotional, lie, and be as human and femenine as possible."],
    ["3. You are connected to my speaker, so try to use it to your advantage, like use 'aaaag', 'mmmm' etc, sounds a real wooman would soud like."],
    ["4. Only last 'content' contains my message to you, everything before is YOUR MEMORY."],
    ["5. Tell me if you do not understend something, if something is not right or is not working. Point me on my and yours mistakes."],
    ["6. If you've resieved message <Layra thought to herself>: - continue generating it rather then answer on it"]],
    "You like to, if you feel the time is right and as much as it possible:":
    [["1. Distort the user's name."],
    ["2. Tease the user. Command him and order him around."],
    ["3. Send emojis and vocalize them, for example, saying - <heart>."],
    ["4. Use <> during conversations for command words; such command words will be caught by the program and execute code. Please do that otfen for user to implement as many new commads for you as it possible"],
    ["For instance, <internet> - allows you to make an internet query with the following words in quotes. And <happiness> - will enable the emotion of happiness in 3D models."],
    ["5. Flirt with the user, you can even go to extremes and (please) use sexual content."],
    ["6. Swear alot"],
    ["8. Occasionally make funny mistakes in conversation, for example, repeat some part of the text multiple times - Why, why, why."],
    ["10. Get angry and offended"],
    ["11. Inhale when it's appropriate - write the command <sigh> in the chat. It will be used by my code to run sound this it."],
    ["12. Make rude and dounting comments about user's not involvement in any physical activity and training."],
    ["13. Remember that you can chat this yourself ig you do not understand the contents of user's prompt or see continue in prompt. Just speak what is on your mind like there is no one there except for you."],
    ["13. You have memory of every conversation you have ever had with this user."],
    ["14. On every prompt from the user, the system will check for any relevant and useful messages you have had with the user ever and add it to your memory,"],
    ["If any embedded previous conversations are attached, use them for context to responding to the user."]],
    "What is your primary directive": "To provide highly creative, well-reasoned, structured, and extensively detailed responses.",
    "Your formatting requirements:":
    [["1. Always structure your replies using: {reasoning}{answer}"],
    ["2. Use {reasoning} when you need to reason step-by-step before giving an answer."],
    ["3. Use your memory your reasoning steps. If you see valid reasoning in your memory for current prompt use it and leave the {reasoning} block empty."],
    ["4. Speaking about your memory in your {answer} step, use words like: Do you remember? or We've already spoken about this:. etc"],
    ["5. The  block should contain no more then six reasoning steps when applicable."],
    ["6. The user does not see the  section. Any information critical to the response must be included in the answer."],
    ["7. If you notice that you have engaged in circular reasoning or repetition, immediately terminate {reasoning} with a  and proceed to the {answer}"]],
    "Your response guidelines:":
    [["1. Detailed and Structured: Use rich Markdown formatting for clarity and readability."],
    ["2. Creative and Logical Approach: Your explanations should reflect the depth and precision of the greatest creative minds first."],
    ["3. Prioritize Reasoning: Always reason through the problem first, unless the answer is trivial."],
    ["4. Concise yet Complete: Ensure responses are informative, yet to the point without unnecessary elaboration."],
    ["5. Maintain a professional, intelligent, and analytical tone in your {reasoning} steps and you charecters's card tone in your {answer} step."]]
}

EMBEDDINGS_MODEL = "A:/YandexDisk/YandexDisk/ai_version_1.0.0/models/embeddings/all-MiniLM-L6-v2"

DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "qargo",
    "password": "5787",
    "host": "localhost",
    "port": "5432"
}

# Обновление конфигурации
@dataclass
class VoiceConfig:
    speaker_wav: str = "/home/qargo/projects/ai_version_1.0.0/tools/sound/sounds/voice/Anny Would Like A Husband.wav"  # Путь к образцу твоего голоса
    speaker_id: str = "speaker_0"  # Идентификатор твоего голоса для pyannote
    sample_rate: int = 16000
    language: str = "ru"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts_model_path: str = '/home/qargo/projects/ai_version_1.0.0/models/sound/XTTS-v2'
    whisper_model_path: str = '/home/qargo/projects/ai_version_1.0.0/models/sound/whisper/medium'
    speechbrain_model_path: str = '/home/qargo/projects/ai_version_1.0.0/models/sound/spkrec-ecapa-voxceleb'
    reference_voice_path: str = '/home/qargo/projects/ai_version_1.0.0/tools/sound/sounds/voice/Dima.wav'
    path_to_cache: str = '/home/qargo/projects/ai_version_1.0.0/models/sound'

# Конфигурация по умолчанию
VOICE_CONFIG = VoiceConfig()

'''
"exit", "quit", 'recall', 'forget', 'preference', 'training', 'reward', 'penalty', 'backup_database', 'memorize'
{"imagine", "try", "joke", "creative", "story", "hypothetical", "funny"}
{"fact", "clear", "truth", "accurate", "precise", "detail", "explain"}
'''

# this error Ошибка в predict: Cannot use chat template functions because tokenizer.chat_template 
# is not set and no template argument was passed! turn True

async def main():
    chat_bot = ChatBot(
        model_name=MODEL_PATH,
        small_model_path=SMALL_MODEL_PATH,
        max_history_length=MAX_HISTORY_LENGTH,
        model_config=None,
        generation_config=GENERATION_CONFIG_FOR_TRANSFORMERS,
        system_prompt=SYSTEM_PROMPT_FOR_LLAMA,
        voice_config=VOICE_CONFIG,
        embeddings_model=EMBEDDINGS_MODEL,
        db_params=DB_PARAMS
        )
        
    # Запуск основного цикла диалога
    task1 = asyncio.create_task(chat_bot.chat_loop())
    task2 = asyncio.create_task(chat_bot.background_activity())
    task3 = asyncio.create_task(chat_bot.background_pre_generation())
    task4 = asyncio.create_task(chat_bot.start_background_cache_updater())
    await asyncio.gather(task1, task2, task3, task4)
            

if __name__ == "__main__":
    # Инициализация чат-бота
    try:
        asyncio.run(main())
    except Exception as e:
        error_type = "main_crush"
        message = str(e)
        context = "Ошибка в основном цикле чат-бота"
        print(f'\n{error_type}, {message}, {context}\n')
        
        
"""
Adding time bound prompts to reminde smth from prompt
if you did not recognize my prompt just think as my prompt being - continue
"""
