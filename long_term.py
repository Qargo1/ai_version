import re
import logging
import subprocess
#from tqdm import tqdm
import shutil

from threading import Thread

import chromadb

import psycopg2
#from psycopg2 import sql

import numpy as np
from cachetools import LRUCache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch


class HelperForSQL:
    def __init__(self):
        pass
    
    def connect_db(self):
        """Подключается к базе данных."""
        try:
            if not self.conn:
                self.conn = psycopg2.connect(
                    dbname=self.db_params["dbname"],
                    user=self.db_params["user"],
                    password=self.db_params["password"],
                    host=self.db_params["host"],
                    port=self.db_params["port"]
                )
                logging.info("Подключение к базе данных успешно установлено.")
            return self.conn
        except Exception as e:
            logging.error(f"Ошибка подключения к базе данных в connect_db: {str(e)}")
            return None
        
    def get_cursor(self):
        if not self.conn:
            self.conn = self.connect_db()
            if not self.conn:
                logging.error("Не удалось получить курсор: нет соединения.")
                return None
        return self.conn.cursor()
    
    def close_connection(self):
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def check_table_exists(self, table_name):
        result = self.execute_query(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", 
            (table_name,),
            func_name='check_table_exists'
        )
        if result and isinstance(result, list) and isinstance(result[0], dict):
            return result[0].get('exists', False)  # ✅ Теперь берем значение из словаря
        return False
                    
    def create_indexes(self):
        conn = self.connect_db()
        if not conn:
            logging.error("Не удалось подключиться к базе данных.")
            return

        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations (timestamp);
                    CREATE INDEX IF NOT EXISTS idx_conversations_prompt ON conversations USING GIN (to_tsvector('english', prompt));
                    CREATE INDEX IF NOT EXISTS idx_conversations_response ON conversations USING GIN (to_tsvector('english', response));
                    CREATE INDEX IF NOT EXISTS idx_user_preferences_role ON user_preferences (prompt);
                    CREATE INDEX IF NOT EXISTS idx_training_data_prompt ON training_data (prompt);
                    CREATE INDEX IF NOT EXISTS idx_training_data_bad_response ON training_data (bad_response);
                    CREATE INDEX IF NOT EXISTS idx_training_data_right_response ON training_data (right_response);
                    CREATE INDEX IF NOT EXISTS idx_bot_error_message ON bot_errors (message);
                """)
                conn.commit()
                logging.info("Индексы успешно созданы.")
        except Exception as e:
            logging.error(f"Ошибка в create_index: {str(e)}")
        finally:
            self.close_connection()
            
    def execute_query(self, query, params=None, fetch=True, func_name=None, reuse_connection=False):
        """Обертка для SQL-запросов с автоматическим закрытием соединения."""
        if reuse_connection and self.conn:
            conn = self.conn
        else:
            conn = self.connect_db()
            if not conn:
                logging.error(f"Ошибка подключения в {func_name}")
                return None
            
        if not conn:
            conn = self.connect_db()
            if not conn:
                logging.error("Не удалось подключиться к базе данных в execute_query.")
                return False

        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    return None
        except Exception as e:
            logging.error(f"Ошибка в {func_name}: {str(e)}")
            return None
        finally:
            if not reuse_connection and conn:
                self.close_connection()
                    
    def backup_database(self):
        if not shutil.which("pg_dump"):
            logging.error("Команда pg_dump не найдена. Убедитесь, что PostgreSQL установлен.")
            return

        try:
            subprocess.run(
                ["pg_dump", "-U", self.db_params["user"], "-d", self.db_params["dbname"], "-f", "backup.sql"],
                check=True
            )
            logging.info("Резервная копия базы данных успешно создана: backup.sql")
        except subprocess.CalledProcessError as e:
            logging.error(f"Ошибка при создании резервной копии: {str(e)}")


class LongTermMemory(HelperForSQL):
    def __init__(self, db_params=None, convo=None, embeddings_model=None):
        self.long_memory = []
        
        self.convo = convo or []  # Если convo не передан, используем пустой список
        
        self.db_params = db_params
        
        self.db_params = db_params
        
        self.conn = None
        
        try:
            self.client = chromadb.Client()
            logging.basicConfig(level=logging.INFO)
            
            self.embeddings_model = embeddings_model
            
            # Загрузка модели для генерации эмбеддингов
            self.model = SentenceTransformer(self.embeddings_model)
            
            # Подключение к ChromaDB
            self.vector_db = None  # База для векторного поиска
            
            self.embedding_cache = LRUCache(maxsize=1000)  # Кэш для эмбеддингов
            self.user_preferences_cache = LRUCache(maxsize=5000)  # Кэш user_preferences

            self.load_user_preferences()  # Загружаем данные в кэш
        except Exception as e:
            print(f"Exception in LongTermMemory.__init__: {e}")
            
    def get_embedding(self, text):
        """
        Генерирует эмбеддинг текста с использованием модели из transformers.
        """
        pass
        '''
        try:
            if text in self.embedding_cache:
                return self.embedding_cache[text]
            
            # Токенизация текста
            outputs = self.model(text)
            
            # Используем среднее значение по последнему скрытому состоянию
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Сохраняем эмбеддинг в кэше
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Exception in get_embedding: {e}")
        '''
    
    def retrieve_embeddings(self, queries, results_per_query=2):
        pass
        '''
        try:
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
        except Exception as e:
            print(f"Exception in retrieve_embeddings: {e}")
        '''

    def clean_response(self, response):
        """
        Удаляет всё содержимое между <think> и </think>, включая сами теги.
        """
        try:
            return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"Exception in clean_response: {e}")
    
    def fetch_conversations(self):
        try:
            return self.execute_query("SELECT * FROM conversations", func_name="fetch_conversations")
        except Exception as e:
            print(f"Exception in fetch_conversations: {e}")

    def fetch_good_responses(self, tag=None):
        try:
            query = "SELECT * FROM conversations WHERE quality = 'good'"
            if tag:
                query += " AND tag = %s"
                return self.execute_query(query, (tag,), func_name='fetch_good_responses')
            return self.execute_query(query)
        except Exception as e:
            print(f"Exception in fetch_good_responses: {e}")

    def check_similar_preference(self, key, value):
        pass
        '''
        try:
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
        except Exception as e:
                print(f"Exception in check_similar_preference: {e}")
        '''

    def save_user_preference(self, prompt, response):
        """Сохраняет предпочтение в базу и обновляет кэш."""
        try:
            if self.check_similar_preference(prompt, response):
                confirm = input("Похожее значение уже есть. Добавить новое? (yes/no): ").strip().lower()
                if confirm != "yes":
                    print("Сохранение отменено.")
                    return

            self.execute_query(
                """
                INSERT INTO user_preferences (prompt, response)
                VALUES (%s, %s)
                ON CONFLICT (prompt) DO UPDATE SET response = EXCLUDED.response
                """,
                (prompt, response),
                func_name='save_user_preference'
            )
            
            # Обновляем кэш
            self.user_preferences_cache[prompt] = response
            print(f"Предпочтение сохранено: {prompt} -> {response}")
            
            def trim_user_preferences():
                """Удаляет старые записи из таблицы user_preferences, оставляя только последние 100."""
                try:
                    self.execute_query(
                        """
                        DELETE FROM user_preferences
                        WHERE id NOT IN (
                            SELECT id FROM user_preferences
                            ORDER BY id DESC
                            LIMIT 100
                        )
                        """,
                        fetch=False,
                        func_name="trim_user_preferences"
                    )
                    logging.info("Старые записи из таблицы user_preferences успешно удалены.")
                except Exception as e:
                    logging.error(f"Ошибка при очистке таблицы user_preferences: {str(e)}")
            
            # Проверяем и удаляем лишние записи
            trim_user_preferences()
            
            # Логируем предупреждение, если количество записей превышает лимит
            count = self.execute_query("SELECT COUNT(*) FROM user_preferences", func_name="count_user_preferences")
            if count and count[0][0] > 100:
                logging.warning("Количество записей в user_preferences превышает лимит (100).")
            
            # Логируем предупреждение, если количество записей превышает лимит
            count = self.execute_query("SELECT COUNT(*) FROM user_preferences", func_name="count_user_preferences")
            if count and count[0][0] > 100:
                logging.warning("Количество записей в user_preferences превышает лимит (100).")
        except Exception as e:
                print(f"Exception in save_user_preference: {e}")

    def get_user_preference(self, prompt):
        """Получает предпочтение из кэша или базы данных."""
        try:
            if prompt in self.user_preferences_cache:
                return self.user_preferences_cache[prompt]  # Берем из кэша
            
            # Если нет в кэше - берем из базы
            response = self.execute_query(
                "SELECT response FROM user_preferences WHERE prompt = %s", (prompt,), func_name='get_user_preference'
            )
            if response:
                response = response[0][0]
                self.user_preferences_cache[prompt] = response  # Кладем в кэш
                return response
            return None
        except Exception as e:
            print(f"Exception in get_user_preference: {e}")

    def store_conversations(self, prompt, response, role=None, quality=None):
        try:
            # Очищаем ответ от <think>...</think>
            cleaned_response = self.clean_response(response)
            
            def trim_conversations():
                """Удаляет старые записи из таблицы conversations, оставляя только последние 5000."""
                try:
                    self.execute_query(
                        """
                        DELETE FROM conversations
                        WHERE id NOT IN (
                            SELECT id FROM conversations
                            ORDER BY timestamp DESC
                            LIMIT 5000
                        )
                        """,
                        fetch=False,
                        func_name="trim_conversations"
                    )
                    logging.info("Старые записи из таблицы conversations успешно удалены.")
                except Exception as e:
                    logging.error(f"Ошибка при очистке таблицы conversations: {str(e)}")
                    
            # Логируем предупреждение, если количество записей превышает лимит
            count = self.execute_query("SELECT COUNT(*) FROM conversations", func_name="count_conversations")
            if count and count[0][0] > 5000:
                logging.warning("Количество записей в conversations превышает лимит (5000).")

            trim_conversations()
            
            with self.get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversations (timestamp, role, prompt, response, quality)
                    VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s)
                    """,
                    (role, prompt, cleaned_response, quality)
                )
                self.conn.commit()
                self.close_connection()
        except Exception as e:
                logging.error(f"Ошибка в store_conversations: {str(e)}")
                
    def store_errors(self, error_type, message):
        """Логирует ошибку в таблицу bot_errors."""
        try:
            self.execute_query(
                """
                INSERT INTO bot_errors (error_type, message)
                VALUES (%s, %s)
                """,
                (error_type, message),
                fetch=False,
                func_name="store_errors"
            )
            logging.info("Ошибка успешно записана в базу данных.")
        except Exception as e:
            logging.error(f"Ошибка в store_errors: {str(e)}")
        
    def store_training_data(self, prompt, bad_response, right_response):
        """Сохраняет обучающие данные, ограничивая до 5000 записей."""
        try:
            # Очищаем ответ от <think>...</think>
            cleaned_response = self.clean_response(bad_response)
            
            self.execute_query(
                """
                INSERT INTO training_data (prompt, bad_response, right_response)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (prompt, cleaned_response, right_response),
                fetch=False,
                func_name="store_training_data"
            )

            # Ограничиваем количество записей
            self.execute_query(
                "DELETE FROM training_data WHERE id NOT IN (SELECT id FROM training_data ORDER BY timestamp DESC LIMIT 5000)",
                fetch=False,
                func_name="trim_training_data"
            )
        except Exception as e:
            logging.error(f"Ошибка в store_conversations: {str(e)}")
        
    def remove_last_conversation(self):
        try:
            self.execute_query("DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)", func_name='remove_last_conversation')
        except Exception as e:
            logging.error(f"Ошибка в remove_last_conversation: {str(e)}")
        
    def create_vector_db(self, conversations):
        """Создает векторную базу из `conversations`."""
        pass
        '''
        vector_db_name = 'conversations'
        
        try:
            self.client.delete_collection(name=vector_db_name)
        except ValueError:
            pass
        
        try:
            self.vector_db = self.client.create_collection(name=vector_db_name)
            
            for c in conversations:
                # ✅ Теперь c - это словарь, и код работает
                # Очищаем текст перед сериализацией
                serialized_convo = f'prompt: {c["prompt"]} response: {self.clean_response(c["response"])}'
                embedding = self.get_embedding(serialized_convo)
                
                self.vector_db.add(
                    ids=[str(c["id"])], 
                    embeddings=[embedding], 
                    documents=[serialized_convo]
                    )
        except Exception as e:
            logging.error(f"Ошибка в create_vector_db => create_collection: {str(e)}")
    

        version 1.0.0
        def cosine_similarity(self, embedding1, embedding2):
            """
            Вычисляет косинусное сходство между двумя эмбеддингами.
            """
            return cosine_similarity([embedding1], [embedding2])[0][0]
        '''

    def cosine_similarity(self, embedding1, embedding2):
        """Вычисляет косинусное сходство между двумя эмбеддингами."""
        # Преобразуем эмбеддинги в массивы NumPy
        try:
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            # Вычисляем косинусное сходство
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return similarity
        except Exception as e:
            logging.error(f"Ошибка в cosine_similarity: {str(e)}")
    
    def add_to_long_memory(self, user_input, response):
        """Добавляет пару (вопрос, ответ) в долговременную память."""
        # Очищаем ответ от <think>...</think>
        try:
            cleaned_response = self.clean_response(response)
            self.long_memory.append({"prompt": user_input, "response": cleaned_response})
        except Exception as e:
            logging.error(f"Ошибка в add_to_long_memory: {str(e)}")
    
    def load_user_preferences(self):
        """Загружает user_preferences в кэш."""
        try:
            logging.info("Загрузка user_preferences в кэш...")
            results = self.execute_query("SELECT prompt, response FROM user_preferences", func_name="load_user_preferences")
            
            if results:
                for prompt, response in results:
                    self.user_preferences_cache[prompt] = response
                logging.info(f"Загружено {len(self.user_preferences_cache)} предпочтений в кэш.")
            else:
                logging.warning("База user_preferences пуста или не загрузилась.")
        except Exception as e:
            logging.error(f"Ошибка в load_user_preferences: {str(e)}")
            
    def retrieve_relevant_memory(self, query, threshold=0.8):
        """Ищет релевантные записи в долговременной памяти на основе запроса."""
        pass
        '''
        try:
            query_embedding = self.get_embedding(query)
            relevant_memories = []
            
            for entry in self.long_memory:
                prompt_embedding = self.get_embedding(entry["prompt"])
                similarity = self.cosine_similarity(query_embedding, prompt_embedding)
                
                if similarity > threshold:
                    relevant_memories.append(entry)
            
            return relevant_memories
        except Exception as e:
            logging.error(f"Ошибка в retrieve_relevant_memory: {str(e)}")
        '''
    
    def recall(self, prompt):
        """Вспоминает релевантные данные из долговременной памяти."""
        try:
            relevant_memories = self.retrieve_relevant_memory(prompt)
            print(f"Найдено {len(relevant_memories)} релевантных записей.")
            for memory in relevant_memories:
                print(f"Prompt: {memory['prompt']}\nResponse: {memory['response']}\n")
        except Exception as e:
            logging.error(f"Ошибка в recall: {str(e)}")


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