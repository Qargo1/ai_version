# Импортируем необходимые библиотеки
from langchain.chains import ConversationChain  # Цепочка для управления диалогом
from langchain.memory import ConversationBufferMemory  # Память для хранения контекста разговора
from langchain.prompts import PromptTemplate  # Шаблон для формирования запросов
from langchain.llms import HuggingFacePipeline  # Интеграция с Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Загрузка модели и токенизатора
import torch  # Работа с PyTorch

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from langchain.vectorstores import Qdrant as LangChainQdrant
from langchain.embeddings import HuggingFaceEmbeddings

# Инициализация токенизатора и модели
model_name = "gpt2"  # Замените на вашу модель
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Интеграция с LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Инициализация памяти
memory = ConversationBufferMemory()

# Шаблон для запросов
template = """You are a helpful assistant. Context: {context}\nQuestion: {question}\nAnswer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Цепочка для управления диалогом
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

        # Инициализация Qdrant
        self.qdrant_client = QdrantClient(":memory:")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = LangChainQdrant(client=self.qdrant_client, collection_name="my_collection", embedding_function=embeddings)


    def initialize_qdrant(self):
        # Добавление данных в Qdrant
        texts = ["Hello, world!", "How are you?", "This is a test."]
        vectorstore.add_texts(texts)

        # Поиск похожих текстов
        query = "Hello!"
        results = vectorstore.similarity_search(query, k=2)
        context = "\n".join([result.page_content for result in results])

        # Генерация ответа
        response = conversation.run(question=query, context=context)
        print(response)

        # Инициализация клиента Qdrant
        client = QdrantClient(":memory:")  # Для тестирования можно использовать in-memory режим
        # client = QdrantClient("localhost", port=6333)  # Для локального запуска

        # Создание коллекции
        collection_name = "my_collection"
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Размер вектора зависит от модели
        )

        # Загрузка модели для генерации embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Генерация векторов
        texts = ["Hello, world!", "How are you?", "This is a test."]
        embeddings = model.encode(texts)

        # Добавление точек в коллекцию
        points = [
            PointStruct(id=i, vector=embedding.tolist(), payload={"text": text})
            for i, (embedding, text) in enumerate(zip(embeddings, texts))
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Поиск похожих векторов
        query_vector = model.encode(["Hello!"])[0].tolist()
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=2
        )

        for hit in hits:
            print(f"Text: {hit.payload['text']}, Score: {hit.score}")
                
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
            