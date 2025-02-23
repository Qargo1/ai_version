
import re
import logging
import subprocess
#from tqdm import tqdm
import shutil

from threading import Thread

import psycopg2

import numpy as np
from cachetools import LRUCache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

import logging
import psycopg2
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from peft import LoraConfig, get_peft_model
import asyncio

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class SQLHelper:
    def __init__(self):
        pass
    
    def connect_db(self):
        try:
            self.conn = psycopg2.connect(**self.db_params)
            logging.info("Подключение к базе данных успешно установлено.")
        except Exception as e:
            logging.error(f"Ошибка подключения к базе данных: {str(e)}")
            
    def execute_query(self, query, params=None, fetch=True):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    self.conn.commit()
                    return None
        except Exception as e:
            logging.error(f"Ошибка в execute_query: {str(e)}")
            return None
        
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
            
    def store_conversations(self, prompt, response, role=None, quality=None):
        try:
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
    
    def load_user_preferences(self):
            results = self.execute_query("SELECT prompt, response FROM user_preferences")
            if results:
                for row in results:
                    self.user_preferences_cache[row["prompt"]] = row["response"]
                logging.info(f"Загружено {len(self.user_preferences_cache)} предпочтений в кэш.")
    
    def recall(self, prompt):
        """Вспоминает релевантные данные из долговременной памяти."""
        try:
            relevant_memories = self.retrieve_relevant_memory(prompt)
            print(f"Найдено {len(relevant_memories)} релевантных записей.")
            for memory in relevant_memories:
                print(f"Prompt: {memory['prompt']}\nResponse: {memory['response']}\n")
        except Exception as e:
            logging.error(f"Ошибка в recall: {str(e)}")


class LongTermMemory(SQLHelper):
    def __init__(self, db_params=None, embeddings_model=None):
        self.db_params = db_params
        self.conn = None
        
        # Подключение к базе данных
        self.connect_db()

        # Модель эмбеддингов
        self.embeddings_model = SentenceTransformer(embeddings_model)
        
        # Qdrant клиент
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "conversations"
        
        # Создание коллекции, если не существует
        if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE)  # all-MiniLM-L6-v2: 384
            )
        
        # Кэш предпочтений
        self.user_preferences_cache = {}
        self.load_user_preferences()
        
    def get_embedding(self, text):
        return self.embeddings_model.encode(text, convert_to_numpy=True)
    
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

    def save_user_preference(self, prompt, response):
        self.execute_query(
            "INSERT INTO user_preferences (prompt, response) VALUES (%s, %s) ON CONFLICT (prompt) DO UPDATE SET response = EXCLUDED.response",
            (prompt, response),
            fetch=False
        )
        self.user_preferences_cache[prompt] = response
            
    async def add_to_long_memory(self, user_input, response):
        cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        embedding = self.get_embedding(f"prompt: {user_input} response: {cleaned_response}").tolist()
        
        # Генерируем уникальный ID
        id = hash(user_input + cleaned_response)
        
        # Добавляем в Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[rest.PointStruct(
                id=id,
                vector=embedding,
                payload={"prompt": user_input, "response": cleaned_response}
            )]
        )
        
        # Сохраняем в SQL
        self.execute_query(
            "INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)",
            (user_input, cleaned_response),
            fetch=False
        )
        
    async def retrieve_relevant_memory(self, query, threshold=0.8, max_results=3):
        query_embedding = self.get_embedding(query).tolist()
        
        # Поиск в Qdrant
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=max_results,
            score_threshold=threshold  # Для COSINE, threshold = 1 - distance
        )
        
        relevant = []
        for point in search_result:
            relevant.append({
                "prompt": point.payload["prompt"],
                "response": point.payload["response"]
            })
        return relevant


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