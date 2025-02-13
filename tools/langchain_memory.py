# Импортируем необходимые библиотеки
from langchain.chains import ConversationChain  # Цепочка для управления диалогом
from langchain.memory import ConversationBufferMemory  # Память для хранения контекста разговора
from langchain.prompts import PromptTemplate  # Шаблон для формирования запросов
from langchain.llms import HuggingFacePipeline  # Интеграция с Hugging Face
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant as LangChainQdrant
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Загрузка модели и токенизатора
import torch  # Работа с PyTorch

# === Шаг 1: Загрузка локальной модели Hugging Face ===
# Указываем путь к локальной модели (например, Qwen или другую)
model_name = "path/to/your/local/model"  # Укажите путь к вашей локальной модели

# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Токенизатор преобразует текст в токены
model = AutoModelForCausalLM.from_pretrained(model_name)  # Модель для генерации текста

# Инициализация Qdrant
client = QdrantClient("localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Создание векторного хранилища
vector_store = LangChainQdrant(client=client, collection_name="conversations", embedding_function=embeddings)

# Поиск по векторам
query = "Как зовут мою собаку?"
results = vector_store.similarity_search(query, k=2)
for result in results:
    print(result.page_content)  # Выводим найденные контексты

# Создаем pipeline для генерации текста
generate_text = pipeline(
    "text-generation",  # Тип задачи: генерация текста
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # Используем GPU, если доступен
)

# === Шаг 2: Интеграция с LangChain ===
# Создаем цепочку для управления диалогом
memory = ConversationBufferMemory()  # Память для хранения контекста разговора
conversation = ConversationChain(
    llm=HuggingFacePipeline(pipeline=generate_text),  # Используем загруженную модель
    memory=memory,  # Подключаем память для хранения контекста
    verbose=True  # Выводим отладочную информацию
)

# === Шаг 3: Использование внешних данных ===
# Предположим, у нас есть текстовые заметки, которые мы хотим использовать
external_data = """
Вот несколько полезных фактов:
- Python — это язык программирования.
- LangChain помогает интегрировать модели с внешними данными.
- Hugging Face предоставляет предобученные модели.
"""

# Создаем шаблон для запроса, включающий внешние данные
template = """
Используй следующие факты для ответа:
{external_data}

Текущий диалог:
{history}

Вопрос пользователя:
{input}
"""

prompt = PromptTemplate(
    input_variables=["external_data", "history", "input"],  # Переменные, которые будут подставляться в шаблон
    template=template  # Сам шаблон
)

# === Шаг 4: Функция для общения с ботом ===
def chat_with_bot():
    print("Диалог начат. Для выхода введите 'exit'.")
    while True:
        user_input = input("Вы: ")  # Получаем ввод от пользователя
        if user_input.lower() == "exit":
            print("Диалог завершен.")
            break
        
        # Формируем запрос с использованием шаблона
        full_prompt = prompt.format(
            external_data=external_data,  # Внешние данные
            history=memory.buffer,  # Контекст разговора
            input=user_input  # Вопрос пользователя
        )
        
        # Генерируем ответ
        response = conversation.predict(input=full_prompt)  # Передаем запрос в цепочку
        print(f"Бот: {response}")  # Выводим ответ

# === Шаг 5: Запуск программы ===
if __name__ == "__main__":
    chat_with_bot()