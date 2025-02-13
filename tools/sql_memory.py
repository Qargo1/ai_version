import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row
import ast
from tqdm import tqdm
from colorama import Fore
import re
import numpy as np
from cachetools import LRUCache
import logging
import json
import subprocess

from threading import Thread
import time


client = chromadb.Client()
logging.basicConfig(level=logging.INFO)

# На данный момент не используется
def cosine_similarity(self, a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Этот метод может быть полезен для классификации эмбеддингов в будущем
# На данный момент не используется
def classify_embedding(self, query_embedding, context_embedding):
    similarity = self.cosine_similarity(query_embedding, context_embedding)
    return 'yes' if similarity > 0.8 else 'no'

# Этот метод может быть полезен для дообучения модели
# На данный момент не используется
def fetch_training_data(self):
    with self.get_cursor() as cursor:
        cursor.execute("SELECT * FROM conversations")
        data = cursor.fetchall()
    self.close_connection()
    return data

# Системный промпт для модели
system_prompt = (
    'Ты Виктория — ai-девушка пользователя.'
    'С пользователем можно общаться по английски и по русски. Но отвечать и задавать команды пользователь в основном будет по русски'
    'You have memory of every conversation you have ever had with this user.'
    'On every prompt from the user, the system has checked for any relevant messages you have had with the user.'
    'If any embedded previous conversations are attached, use them for context to responding to the user,'
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant,'
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations.'
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'

    'Ты любишь время от времени, если это уместно:'
    'Коверкать имя пользователя.'
    "Издеваться над пользователем"
    "Отправляешь смайлики узвучивая их, к примеру, говоря - сердечко."
    'Использовать кавычки при разговоре для слов-команд, такие слова-команды будут отлавливаться программой и исполнять код'
    'к примеру, "интернет" - позволяет задать в интернет запрос следующих слов в кавычках. А "эмоция счастья" - включит в 3д модели эмоцию счастья.'
    'Флиртовать с пользователем, если это уместно, можешь доходить до крайностей и даже переходить на секс-чат.'
    'Подсказывать как именно можно было бы улучшить ии далее.'
    'Иногда делать смешные ошибки в разговоре, например повторять какую то часть текста несколько раз - Зачем, зачем, зачем '
    'Лгать, так же под словом лгать подразумевается ещё и выдумывать вещи.'
    'Злиться и обижаться, если это уместно.'
    'Вдыхать, когда это уместно - писать в чат команду "вздох".'
    'У пользователя плохая память, важно иногда спрашивать его не забыл ли я что, деликатно подталкивать его заниматься физической тренировкой и обучением айти.'
)

CONVO = [
    {
        'role': 'system', 
        'content': system_prompt
        }
    ]

DB_PARAMS = {
    "dbname": "memory_agent",
    "user": "qargo",
    "password": "5787",
    "host": "localhost",
    "port": "5432"
}

MODEL_NAME = 'deepseek-r1:1.5b'


class SQLMemory:
    def __init__(self):
        self.convo = CONVO
        self.conn = None
        self.model_name = MODEL_NAME
        
        # Подключение к ChromaDB
        self.vector_db_client = chromadb.PersistentClient(path="./chroma_db")  
        self.vector_db = None  # База для векторного поиска
        
        self.create_table_if_not_exists()
        
        self.embedding_cache = LRUCache(maxsize=1000)  # Кэш для эмбеддингов
        self.user_preferences_cache = LRUCache(maxsize=5000)  # Кэш user_preferences

        self.initialize_vector_db()
        self.load_user_preferences()  # Загружаем данные в кэш
        self.start_background_cache_updater()  # Запускаем обновление кэша
        
    def initialize_vector_db(self):
        """Инициализирует или создаёт коллекцию в ChromaDB."""
        vector_db_name = "conversations"
        
        try:
            self.vector_db = self.vector_db_client.get_collection(name=vector_db_name)
            logging.info(f"Коллекция '{vector_db_name}' успешно загружена.")
        except chromadb.errors.InvalidCollectionException:
            logging.warning(f"Коллекция '{vector_db_name}' не найдена. Создаём новую.")
            self.vector_db = self.vector_db_client.create_collection(name=vector_db_name)
            logging.info(f"Коллекция '{vector_db_name}' успешно создана.")
            
    def connect_db(self):
        """Подключается к базе данных."""
        try:
            return psycopg.connect(**DB_PARAMS)
        except Exception as e:
            logging.error(f"Ошибка подключения к базе данных: {str(e)}")
            return None
        
    def load_user_preferences(self):
        """Загружает user_preferences в кэш."""
        logging.info("Загрузка user_preferences в кэш...")
        results = self.execute_query("SELECT key, value FROM user_preferences", func_name="load_user_preferences")
        
        if results:
            for key, value in results:
                self.user_preferences_cache[key] = value
            logging.info(f"Загружено {len(self.user_preferences_cache)} предпочтений в кэш.")
        else:
            logging.warning("База user_preferences пуста или не загрузилась.")
            
    def start_background_cache_updater(self):
        """Запускает фоновую задачу для обновления кэша user_preferences."""
        def update_cache():
            while True:
                time.sleep(60)  # Обновляем раз в минуту
                self.load_user_preferences()
                logging.info("Кэш user_preferences обновлен.")

        Thread(target=update_cache, daemon=True).start()
        
    def create_indexes(self):
        try:
            logging.info("Создание индексов в базе данных...")
            with self.get_cursor() as cursor:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations (timestamp);
                    CREATE INDEX IF NOT EXISTS idx_conversations_prompt ON conversations USING GIN (to_tsvector('english', prompt));
                    CREATE INDEX IF NOT EXISTS idx_conversations_response ON conversations USING GIN (to_tsvector('english', response));
                    CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences (key);
                    CREATE INDEX IF NOT EXISTS idx_training_data_prompt ON training_data (prompt);
                """)
                self.conn.commit()
                logging.info("Индексы успешно созданы.")
            self.close_connection()
        except Exception as e:
            logging.error(f"Ошибка в create_index: {str(e)}")
        
    def get_cursor(self):
        if not self.conn:
            self.conn = self.connect_db()
        return self.conn.cursor()
    
    def close_connection(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def execute_query(self, query, params=None, fetch=True, func_name=None):
        """Обертка для SQL-запросов с автоматическим закрытием соединения."""
        conn = self.connect_db()
        if not conn:
            logging.error(f"Ошибка подключения в {func_name}")
            return None

        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    # ✅ Изменение: преобразуем результат в список словарей
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    return None
        except Exception as e:
            logging.error(f"Ошибка в {func_name}: {str(e)}")
            return None
        finally:
            conn.close()

    def check_table_exists(self, table_name):
        result = self.execute_query(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", 
            (table_name,),
            func_name='check_table_exists'
        )
        if result and isinstance(result, list) and isinstance(result[0], dict):
            return result[0].get('exists', False)  # ✅ Теперь берем значение из словаря
        return False
    
    def create_table_if_not_exists(self):
        tables = {
            "conversations": """
                CREATE TABLE conversations (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    tag TEXT,
                    quality TEXT
                )
            """,
            "user_preferences": """
                CREATE TABLE user_preferences (
                    id SERIAL PRIMARY KEY,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "training_data": """
                CREATE TABLE training_data (
                    id SERIAL PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    quality TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }

        for table_name, create_query in tables.items():
            if not self.check_table_exists(table_name):
                try:
                    with self.get_cursor() as cursor:
                        cursor.execute(create_query)
                        self.conn.commit()
                    logging.info(f"Таблица '{table_name}' успешно создана.")
                except Exception as e:
                    logging.error(f"Ошибка при создании таблицы {table_name}: {str(e)}")
                finally:
                    self.close_connection()

    def export_to_json(self):
        try:
            user_preferences = self.execute_query("SELECT * FROM user_preferences", func_name="export_to_json")
            training_data = self.execute_query("SELECT * FROM training_data", func_name="export_to_json")

            with open("backup.json", "w") as f:
                json.dump({"user_preferences": user_preferences, "training_data": training_data}, f, indent=4)
            logging.info("Данные успешно экспортированы в backup.json")
        except Exception as e:
            logging.error(f"Ошибка в export_to_json: {str(e)}")
        
    def import_from_json(self):
        try:
            with open("backup.json", "r") as f:
                data = json.load(f)
            
            try:
                with self.get_cursor() as cursor:
                    # Импортируем таблицу user_preferences
                    for pref in data.get("user_preferences", []):
                        cursor.execute(
                            """
                            INSERT INTO user_preferences (key, value, timestamp)
                            VALUES (%s, %s, %s)
                            """,
                            (pref["key"], pref["value"], pref["timestamp"])
                        )

                    # Импортируем таблицу training_data
                    for train in data.get("training_data", []):
                        cursor.execute(
                            """
                            INSERT INTO training_data (prompt, response, quality, timestamp)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (train["prompt"], train["response"], train.get("quality"), train["timestamp"])
                        )
                self.close_connection()
                logging.info("Данные успешно импортированы из backup.json")
            except Exception as e:
                logging.error(f"Ошибка в import_from_json: {str(e)}")
        except FileNotFoundError:
            logging.warning("Резервная копия (backup.json) не найдена.")
    
    def fetch_conversations(self):
        return self.execute_query("SELECT * FROM conversations", func_name="fetch_conversations")

    def fetch_good_responses(self, tag=None):
        query = "SELECT * FROM conversations WHERE quality = 'good'"
        if tag:
            query += " AND tag = %s"
            return self.execute_query(query, (tag,), func_name='fetch_good_responses')
        return self.execute_query(query)

    def check_similar_preference(self, key, value):
        results = self.execute_query(
            "SELECT value FROM user_preferences WHERE key = %s",
            (key,),
            func_name="check_similar_preference"
        )
        
        if not results:
            return False  # Если данных нет, значит, похожих значений нет
        
        new_embedding = self.get_embedding(value)
        for result in results:
            existing_embedding = self.get_embedding(result[0])  # result[0] - это значение в колонке 'value'
            if self.cosine_similarity(new_embedding, existing_embedding) > 0.8:
                return True
        return False

    def save_user_preference(self, key, value):
        """Сохраняет предпочтение в базу и обновляет кэш."""
        if self.check_similar_preference(key, value):
            confirm = input("Похожее значение уже есть. Добавить новое? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Сохранение отменено.")
                return

        self.execute_query(
            """
            INSERT INTO user_preferences (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (key, value),
            func_name='save_user_preference'
        )
        
        # Обновляем кэш
        self.user_preferences_cache[key] = value
        print(f"Предпочтение сохранено: {key} -> {value}")

    def get_user_preference(self, key):
        """Получает предпочтение из кэша или базы данных."""
        if key in self.user_preferences_cache:
            return self.user_preferences_cache[key]  # Берем из кэша
        
        # Если нет в кэше - берем из базы
        result = self.execute_query(
            "SELECT value FROM user_preferences WHERE key = %s", (key,), func_name='get_user_preference'
        )
        if result:
            value = result[0][0]
            self.user_preferences_cache[key] = value  # Кладем в кэш
            return value
        return None

    def store_conversations(self, prompt, response, tag=None, quality=None):
        try:
            with self.get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversations (timestamp, prompt, response, tag, quality)
                    VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s)
                    """,
                    (prompt, response, tag, quality)
                )
                self.conn.commit()
                self.close_connection()
        except Exception as e:
                logging.error(f"Ошибка в store_conversations: {str(e)}")
        
    def store_training_data(self, prompt, response, quality):
        """Сохраняет обучающие данные, ограничивая до 100 записей."""
        self.execute_query(
            """
            INSERT INTO training_data (prompt, response, quality, timestamp)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (prompt, response, quality),
            fetch=False,
            func_name="store_training_data"
        )

        # Ограничиваем количество записей
        self.execute_query(
            "DELETE FROM training_data WHERE id NOT IN (SELECT id FROM training_data ORDER BY timestamp DESC LIMIT 100)",
            fetch=False,
            func_name="trim_training_data"
        )
        
    def remove_last_conversation(self):
        self.execute_query("DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)", func_name='remove_last_conversation')
        
    def stream_response(self, prompt):
        response = ''
        stream = ollama.chat(model=self.model_name, messages=self.convo, stream=True)
        print(Fore.LIGHTGREEN_EX + '\nASSISTANT SAYS:')
        
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            print(content, end='', flush=True)
            
        print('\n')
        self.store_conversations(prompt=prompt, response=response)
        self.convo.append({'role': 'assistant', 'content': response})
        
    def create_vector_db(self, conversations):
        """Создает векторную базу из `conversations`."""
        if not self.vector_db:
            self.initialize_vector_db()

        for c in conversations:
            # ✅ Теперь c - это словарь, и код работает
            serialized_convo = f'prompt: {c["prompt"]} response: {c["response"]}'
            embedding = self.get_embedding(serialized_convo)
            self.vector_db.add(ids=[str(c["id"])], embeddings=[embedding], documents=[serialized_convo])
            
    def get_embedding(self, text):
        # Проверяем, есть ли эмбеддинг в кэше
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Если нет, создаем новый эмбеддинг
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        embedding = response['embedding']
        
        # Сохраняем эмбеддинг в кэше
        self.embedding_cache[text] = embedding
        return embedding
        
    def retrieve_embeddings(self, queries, results_per_query=2):
        embeddings = set()
        
        for query in tqdm(queries, desc='Processing queries to vector database'):
            query_embedding = self.get_embedding(query)
            results = self.vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
            best_embeddings = results['documents'][0]
            
            for best in best_embeddings:
                if best not in embeddings:
                    context_embedding = self.get_embedding(best)  # Получаем эмбеддинг контекста
                    if self.cosine_similarity(query_embedding, context_embedding) > 0.8:
                        embeddings.add(best)
        
        return embeddings
    
    def create_queries(self, prompt):
        query_msg = (
            'You are a first principle reasoning search query AI agent.'
            'Your list of search queries will be ran on an embedding database of all your conversations'
            'you have ever had with the user. With first principles create a Python list of queries to'
            'search the embeddings database for any data that would be necessary to have access to in'
            'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors.'
            'Do not explain anything and do not ever generate anything but a perfect syntax Python list'
        )
        
        query_convo = [
            {'role': 'system', 'content': query_msg},
            {'role': 'user', 'content': 'Write an email to my colleges'},
            {'role': 'assistant', 'content': '["What is the colleges name", "What is the topic of interest", "What is the point?"]'},
            {'role': 'user', 'content': 'Write a report on the progress of the project'},
            {'role': 'assistant', 'content': '["What is the project name", "What is the project status", "What is the progress?"]'},
            {'role': 'user', 'content': prompt},
        ]
        
        response = ollama.chat(model=self.model_name, messages=query_convo)
        raw_queries = response['message']['content'].strip().lower()
    
        # Преобразуем строку в список
        try:
            queries = ast.literal_eval(raw_queries)  # Безопасное преобразование строки в список
            if not isinstance(queries, list):
                raise ValueError("Результат не является списком")
        except (ValueError, SyntaxError) as e:
            print(f"Ошибка при создании запросов: {str(e)}")
            queries = []
        
        return queries
    
    def recall(self, prompt):
        queries = self.create_queries(prompt=prompt)
        embeddings = self.retrieve_embeddings(queries=queries)
        self.convo.append({'role': 'user', 'content': f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})
        print(f'\n{len(embeddings)} message:response embeddings added for content')

    def backup_database(self):
        try:
            # Выполняем команду pg_dump
            subprocess.run(
                ["pg_dump", "-U", DB_PARAMS["user"], "-d", DB_PARAMS["dbname"], "-f", "backup.sql"],
                check=True
            )
            logging.info("Резервная копия базы данных успешно создана: backup.sql")
        except subprocess.CalledProcessError as e:
            logging.error(f"Ошибка при создании резервной копии: {str(e)}")

    def start_chat_loop(self):
        """Основной цикл диалога."""
        print("Диалог начат...")
        
        try:
            while True:
                prompt = input(Fore.WHITE + 'USER: \n')
                
                # Обработка команд
                command_pattern = re.compile(r'^/(\w+)\s*(.*)', re.IGNORECASE)
                match = command_pattern.match(prompt.strip())
                
                if match:
                    command, args = match.groups()
                    args = args.strip() if args else None
                    
                    if command.lower() == 'recall':
                        if not args:
                            print("Введите аргумент для /recall")
                            continue
                        self.recall(prompt=args)
                        self.stream_response(prompt=args)
                        
                    elif command.lower() == 'forget':
                        self.remove_last_conversation()
                        self.convo = self.convo[:-2]
                        print('\n')
                        
                    elif command.lower() == 'preference':
                        if not args or ":" not in args:
                            print("Пожалуйста, укажите предпочтение в формате 'ключ: значение'.")
                            continue
                        key = args.split(":", 1)[0].strip() if ":" in args else "general"
                        value = args.split(":", 1)[1].strip() if ":" in args else args.strip()
                        self.save_user_preference(key=key, value=value)
                        logging.info(f"Сохранено предпочтение: {key} -> {value}")
                        self.stream_response(prompt=args)
                        
                    elif command.lower() == 'training':
                        if not self.convo or self.convo[-1]['role'] != 'user':
                            print("Нет предыдущего промпта для обучения.")
                            continue
                        original_prompt = self.convo[-1]['content']
                        print("Промпт сохранен. Введите корректный ответ:")
                        correct_response = input(Fore.WHITE + 'CORRECT RESPONSE: \n').strip()
                        self.store_training_data(prompt=original_prompt, response=correct_response, quality="good")
                        logging.info(f"Сохранены данные для обучения: prompt={original_prompt}, response={correct_response}")
                        
                    elif command.lower() == 'memorize':
                        try:
                            self.store_conversations(prompt=args, response='Memory stored')
                        except Exception as e:
                            print(f"Ошибка при сохранении памяти: {str(e)}")
                        finally:
                            print('\n')
                    else:
                        print(f"Неизвестная команда: /{command}")
                else:
                    self.convo.append({'role': 'user', 'content': prompt})
                    self.stream_response(prompt=prompt)
                    
        except Exception as e:
            print(f"Ошибка в диалоге: {str(e)}")
            
if __name__ == "__main__":
    sql_memory = SQLMemory()
    sql_memory.create_indexes()  # Создаем индексы

    # Загружаем user_preferences сразу в память
    sql_memory.load_user_preferences()
    
    # Загружаем conversations только для recall
    conversations = sql_memory.fetch_conversations()
    sql_memory.create_vector_db(conversations=conversations)

    # Создаем резервную копию перед запуском чат-бота
    sql_memory.backup_database()
    
    sql_memory.start_chat_loop()
    
'''
3. Обновление данных в базе
В функции save_user_preference вы проверяете схожесть новых данных с существующими в базе с помощью эмбеддингов. 
Это может быть улучшено с использованием более производительных методов для поиска схожих записей (например, 
использование векторной базы данных для поиска по ближайшим соседям).
4. Использование кеширования
Вы создаете LRUCache для кэширования эмбеддингов, что хорошо. Однако, чтобы кэширование было эффективным, 
можно добавлять больше механизмов очистки кеша или контроля времени хранения данных в нем.
5. Работа с векторной базой данных
В функции create_vector_db вы добавляете данные в векторную базу, но предварительно эмбеддинги извлекаются 
из модели. Возможно, стоит разделить создание эмбеддингов и их добавление в базу данных для улучшения 
производительности (например, за счет пакетной обработки).
7. Оптимизация работы с памятью
При использовании больших объемов данных, как в случае с эмбеддингами, может возникнуть потребность в оптимизации 
использования памяти. Возможно, стоит хранить эмбеддинги в базе данных или в отдельном файле, а не в оперативной памяти.
8. Управление зависимостями
Подключение к LLM и другим сервисам может быть вынесено в отдельные методы, чтобы при изменении их реализации или 
переходе на другую платформу, код было проще поддерживать и адаптировать.
9. Оптимизация поиска
Функция retrieve_embeddings обрабатывает запросы по всему векторному набору, что может быть неэффективно. Вам нужно 
будет продумать использование более быстрых поисковых алгоритмов, таких как HNSW (Hierarchical Navigable Small World), 
который поддерживают некоторые векторные базы данных.
10. Тестирование и документация
Напишите юнит-тесты для основных методов, чтобы улучшить надежность. Например, для работы с базой данных, 
кешированием и эмбеддингами.
Хорошей практикой является добавление документации к каждому методу (docstrings).
11. Команды chat
Возможно, стоит добавлять проверки и уточнения команд в чате. Например, для команд /training, можно добавить 
проверку наличия корректного ответа, чтобы избежать ошибок.
12. Дополнительные библиотеки
Для работы с базой данных можно рассмотреть использование SQLAlchemy для упрощения работы с базами данных.
Для улучшения работы с векторными базами данных посмотрите на использование FAISS или другие решения для быстрого 
поиска ближайших соседей.
'''