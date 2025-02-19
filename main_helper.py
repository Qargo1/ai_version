
# For Deepseek Qwen
GENERATION_CONFIG_FOR_DEEPSEEK_QWEN = {
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
    "top_p": 1, 

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
    "bos_token_id": 151643, 

    # ID токена конца строки 151643
    "eos_token_id": 151643, 

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
    "transformers_version": '4.47.1'
}

MODEL_CONFIG = {
    # Автоматическая настройка реализации внимания (если включено, будет автоматически настроена реализация внимания)
    "_attn_implementation_autoset": True, 

    # Путь или имя модели
    "_name_or_path": "models/llm/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2", 

    # Архитектура модели
    "architectures": [
        "Qwen2ForCausalLM"
    ], 

    # Выпадение вероятности внимания (dropout) для предотвращения переобучения
    "attention_dropout": 0.0, 

    # ID токена начала строки (BOS)
    "bos_token_id": 128000, 

    # ID токена конца строки (EOS)
    "eos_token_id": 128001, 

    # Функция активации для скрытых слоев (например, "silu" — это активация SiLU)
    "hidden_act": "silu", 

    # Размер скрытого слоя (количество нейронов в слое)
    "hidden_size": 3584, 

    # Диапазон для инициализации весов (как сильно будут инициализированы веса)
    "initializer_range": 0.02, 

    # Размер промежуточного слоя (для некоторых моделей может быть больше, чем скрытый слой)
    "intermediate_size": 18944, 

    # Максимальная длина входной последовательности (включая токены BOS и EOS)
    "max_position_embeddings": 131072, 

    # Максимальное количество слоев окон
    "max_window_layers": 28, 

    # Тип модели (это Qwen2)
    "model_type": "qwen2", 

    # Количество голов внимания в слое
    "num_attention_heads": 28, 

    # Количество скрытых слоев (глубина сети)
    "num_hidden_layers": 28, 

    # Количество голов для ключей и значений
    "num_key_value_heads": 4, 

    # Конфигурация квантования модели
    "quantization_config": {
        # Количество бит на параметр модели (4 бита на вес)
        "bits": 4, 

        # Формат контрольной точки
        "checkpoint_format": "gptq", 

        # Применять описание активации (если True, описания будут применяться)
        "desc_act": True, 

        # Динамическое квантование (параметр динамической квантованности)
        "dynamic": None, 

        # Размер групп для квантования
        "group_size": 32, 

        # Использование головы языка (обычно для генеративных моделей)
        "lm_head": False, 

        # Метаинформация квантования
        "meta": {
            "damp_auto_increment": 0.0025,  # Параметры для изменения веса в процессе квантования
            "damp_percent": 0.1,  # Параметр изменения коэффициента
            "quantizer": [
                "gptqmodel:1.7.4"  # Версия квантователя
            ], 
            "static_groups": False,  # Использование статических групп
            "True_sequential": True,  # Должна ли модель использовать истинно последовательное квантование
            "uri": "https://github.com/modelcloud/gptqmodel"  # Ссылка на репозиторий квантователя
        }, 

        # Метод квантования
        "quant_method": "gptq", 

        # Симметричное квантование (если True, квантование будет симметричным)
        "sym": True  
    }, 

    # Параметры для нормализации RMS
    "rms_norm_eps": 1e-06, 

    # Масштабирование для использования ROPE (если используется)
    "rope_scaling": None, 

    # Параметр для масштаба ROPE (ротационное позиционное кодирование)
    "rope_theta": 10000, 

    # Скользящее окно для позиционного кодирования (если используется)
    "sliding_window": None, 

    # Привязать эмбеддинги слов (если False, эмбеддинги слов не будут привязаны)
    "tie_word_embeddings": False, 

    # Тип данных для PyTorch (например, bfloat16 для использования меньшего объема памяти)
    "torch_dtype": "bfloat16", 

    # Версия библиотеки transformers
    "transformers_version": "4.38.2", 

    # Использовать кэш для ускорения генерации
    "use_cache": False, 

    # Использовать ROPE (ротационное позиционное кодирование)
    "use_mrope": False, 

    # Использовать скользящее окно
    "use_sliding_window": False, 

    # Размер словаря (количество токенов)
    "vocab_size": 152064  
}
